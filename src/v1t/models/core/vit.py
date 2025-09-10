from .core import register, Core

import math
import torch
import typing as t
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, einsum
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from v1t.models.utils import DropPath
from v1t.models.layers import scaled_dot_product_attention


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int = None,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super(MLP, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.model = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=use_bias),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=use_bias),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)


class BehaviorMLP(nn.Module):
    def __init__(
        self,
        behavior_mode: int,
        out_dim: int,
        dropout: float = 0.0,
        mouse_ids: t.List[str] = None,
        use_bias: bool = True,
    ):
        """
        behavior mode:
        - 0: do not include behavior
        - 1: concat behavior with natural image
        - 2: add latent behavior variables to each ViT block
        - 3: add latent behavior + pupil centers to each ViT block
        - 4: separate BehaviorMLP for each animal
        """
        super(BehaviorMLP, self).__init__()
        assert behavior_mode in (2, 3, 4)
        self.behavior_mode = behavior_mode
        in_dim = 3 if behavior_mode == 2 else 5
        mouse_ids = mouse_ids if behavior_mode == 4 else ["share"]
        self.models = nn.ModuleDict(
            {
                mouse_id: nn.Sequential(
                    nn.Linear(
                        in_features=in_dim,
                        out_features=out_dim // 2,
                        bias=use_bias,
                    ),
                    nn.Tanh(),
                    nn.Dropout(p=dropout),
                    nn.Linear(
                        in_features=out_dim // 2,
                        out_features=out_dim,
                        bias=use_bias,
                    ),
                    nn.Tanh(),
                )
                for mouse_id in mouse_ids
            }
        )

    def forward(self, inputs: torch.Tensor, mouse_id: str):
        mouse_id = mouse_id if self.behavior_mode == 4 else "share"
        return self.models[mouse_id](inputs)


class Attention(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        emb_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_flash_attention: bool = False,
        use_lsa: bool = False,
        use_bias: bool = True,
        use_scale: bool = True,
        grad_checkpointing: bool = False,
    ):
        super(Attention, self).__init__()
        self.grad_checkpointing = grad_checkpointing
        self.use_flash_attention = use_flash_attention

        self.layer_norm = nn.LayerNorm(emb_dim)

        self.to_qkv = nn.Linear(in_features=emb_dim, out_features=emb_dim * 3, bias=use_bias)
        self.rearrange = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        if use_scale:
            dim_head = emb_dim // num_heads
            scale = dim_head ** -0.5
        else:
            scale = 1.0

        if use_lsa:
            self.register_parameter(
                "scale",
                param=nn.Parameter(torch.full(size=(num_heads,), fill_value=scale)),
            )
            diagonal = torch.eye(num_tokens, num_tokens)
            self.register_buffer(
                "mask",
                torch.nonzero(diagonal == 1, as_tuple=False),
            )
            self.register_buffer(
                "max_value",
                torch.tensor(torch.finfo(torch.get_default_dtype()).max),
            )
        else:
            self.mask = None
            self.register_buffer("scale", torch.tensor(scale))

    def mha(self, inputs: torch.Tensor):
        inputs = self.layer_norm(inputs)
        q, k, v = torch.chunk(self.to_qkv(inputs), chunks=3, dim=-1)
        outputs = scaled_dot_product_attention(
            q=self.rearrange(q), k=self.rearrange(k), v=self.rearrange(v), dropout=self.dropout.p, use_flash_attention=self.use_flash_attention,
        )
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        return outputs

    def forward(self, inputs: torch.Tensor):
        if self.grad_checkpointing:
            outputs = checkpoint(
                self.mha, inputs, preserve_rng_state=True, use_reentrant=False
            )
        else:
            outputs = self.mha(inputs)
        return outputs


class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        emb_dim: int,
        num_blocks: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        behavior_mode: int,
        mouse_ids: t.List[str],
        use_flash_attention: bool = False,
        use_lsa: bool = False,
        drop_path: float = 0.0,
        use_bias: bool = True,
        grad_checkpointing: bool = False,
    ):
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            block = nn.ModuleDict(
                {
                    "mha": Attention(
                        num_tokens=num_tokens,
                        emb_dim=emb_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_flash_attention=use_flash_attention,
                        use_lsa=use_lsa,
                        use_bias=use_bias,
                        grad_checkpointing=grad_checkpointing,
                    ),
                    "mlp": MLP(
                        in_dim=emb_dim,
                        hidden_dim=mlp_dim,
                        dropout=dropout,
                        use_bias=use_bias,
                    ),
                }
            )
            if behavior_mode in (2, 3, 4):
                block["b-mlp"] = BehaviorMLP(
                    behavior_mode=behavior_mode,
                    out_dim=emb_dim,
                    mouse_ids=mouse_ids,
                    use_bias=use_bias,
                )
            self.blocks.append(block)
        self.drop_path = DropPath(dropout=drop_path)
        self.output_shape = (num_tokens, emb_dim) # (num_tokens, emb_dim)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
    ):
        outputs = inputs
        for block in self.blocks:
            if "b-mlp" in block:
                b_latent = block["b-mlp"](behaviors, mouse_id=mouse_id)
                b_latent = repeat(b_latent, "b d -> b 1 d")
                outputs = outputs + b_latent
            outputs = self.drop_path(block["mha"](outputs)) + outputs
            outputs = self.drop_path(block["mlp"](outputs)) + outputs
        # outputs: (1, num_tokens, emb_dim)
        return outputs


@register("vit")
class ViTCore(Core):
    def __init__(
        self,
        args,
        # input_shape: t.Tuple[int, int, int],
        # image_encoder_output_shape: t.Tuple[int, int],
        num_image_patches: int,
        num_neuron_tokens: int,
        name: str = "ViTCore",
    ):
        super(ViTCore, self).__init__(args, name=name)
        self.register_buffer("reg_scale", torch.tensor(args.core_reg_scale))
        self.behavior_mode = args.behavior_mode

        if not hasattr(args, "grad_checkpointing"):
            args.grad_checkpointing = False
        elif args.grad_checkpointing is None:
            args.grad_checkpointing = "cuda" in args.device.type
        if args.grad_checkpointing and args.verbose:
            print(f"Enable gradient checkpointing in ViT")

        self.readout = args.readout

        # self.patch_embedding = Image2Patches(
        #     image_shape=input_shape,
        #     patch_mode=args.patch_mode,
        #     patch_size=args.patch_size,
        #     stride=args.patch_stride,
        #     emb_dim=args.emb_dim_core,
        #     dropout=args.p_dropout,
        # )
        self.transformer = Transformer(
            num_tokens=num_image_patches+num_neuron_tokens,
            emb_dim=args.emb_dim_core,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            dropout=args.t_dropout,
            behavior_mode=self.behavior_mode,
            mouse_ids=list(args.num_output_neurons.keys()),
            use_flash_attention=args.amp,
            use_lsa=args.use_lsa,
            drop_path=args.drop_path,
            use_bias=not args.disable_bias,
            grad_checkpointing=args.grad_checkpointing,
        )
        self.num_image_patches = num_image_patches
        self.project_image = args.emb_dim_image != args.emb_dim_core
        if self.project_image:
            self.image_projection = nn.Linear(in_features=args.emb_dim_image, out_features=args.emb_dim_core, bias=False)
        
        # self.tokenize_neurons = args.tokenize_neurons
        if args.tokenize_neurons:
            self.use_mode_emb = args.use_mode_emb
            self.mode_embedding = nn.Embedding(args.num_modes, args.emb_dim_core)
            # nn.init.constant_(self.embedding.weight, 1.0 / args.emb_dim_core)
            self.project_neuron = args.emb_dim_input_neurons != args.emb_dim_core
            if self.project_neuron:
                self.neuron_projection = nn.Linear(args.emb_dim_input_neurons, args.emb_dim_core)
            

        # calculate latent height and width based on num_patches
        if self.readout == "gaussian2d":
            h, w = self.find_shape(num_image_patches)
            self.output_shape = (self.transformer.output_shape[-1], h, w)
            self.rearrange = Rearrange("b (h w) c -> b c h w", h=h, w=w)
        else:
            self.output_shape = (num_image_patches, self.transformer.output_shape[-1]) # self.transformer.output_shape is (num_neuron_tokens+num_iage_tokens, emb_dim)

    @staticmethod
    def find_shape(num_patches: int):
        dim1 = math.ceil(math.sqrt(num_patches))
        while num_patches % dim1 != 0 and dim1 > 0:
            dim1 -= 1
        dim2 = num_patches // dim1
        return dim1, dim2

    def regularizer(self):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(
        self,
        image_tokens: torch.Tensor, # (B, num_patches, emb_dim_images)
        neuron_tokens: t.Optional[torch.Tensor], # (B, num_neurons, emb_dim_input_neurons)
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        if self.project_image:
            image_tokens = self.image_projection(outputs)
        
        if neuron_tokens is not None:
            if self.project_neuron:
                neuron_tokens = self.neuron_projection(neuron_tokens)
            if self.use_mode_emb: 
                image_mode = torch.zeros_like(image_tokens[..., 0], dtype=torch.long, device=image_tokens.device)
                neuron_mode = torch.ones_like(neuron_tokens[..., 0], dtype=torch.long, device=neuron_tokens.device)
                image_tokens += self.mode_embedding(image_mode)
                neuron_tokens += self.mode_embedding(neuron_mode)
            outputs = torch.cat((image_tokens, neuron_tokens), dim=1)
        else:
            outputs = image_tokens
        if self.behavior_mode in (3, 4):
            behaviors = torch.cat((behaviors, pupil_centers), dim=-1)
        outputs = self.transformer(outputs, mouse_id=mouse_id, behaviors=behaviors)
        # subselect only image patches because they represent 'receptive fields' of neurons and can be optionally conditioned on input neuron tokens
        outputs = outputs[:, :self.num_image_patches, :] 
        # outputs = outputs[:, 1:, :]  # remove CLS token
        if self.readout == "gaussian2d":
            outputs = self.rearrange(outputs)
        return outputs

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
        emb_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_flash_attention: bool = False,
        use_lsa: bool = False,
        use_bias: bool = True,
        grad_checkpointing: bool = False,
    ):
        super(Attention, self).__init__()
        self.grad_checkpointing = grad_checkpointing
        self.use_flash_attention = use_flash_attention
        
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        dim_head = emb_dim // num_heads

        self.layer_norm = nn.LayerNorm(emb_dim)

        self.to_qkv = nn.Linear(in_features=emb_dim, out_features=emb_dim * 3, bias=use_bias)
        self.rearrange = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.dropout = nn.Dropout(p=dropout)

        self.projection = nn.Sequential(
            nn.Linear(in_features=dim_head*num_heads, out_features=emb_dim, bias=use_bias),
            nn.Dropout(p=dropout),
        )

        # if use_lsa:
        #     self.register_parameter(
        #         "scale",
        #         param=nn.Parameter(torch.full(size=(num_heads,), fill_value=scale)),
        #     )
        #     diagonal = torch.eye(num_tokens, num_tokens)
        #     self.register_buffer(
        #         "mask",
        #         torch.nonzero(diagonal == 1, as_tuple=False),
        #     )
        #     self.register_buffer(
        #         "max_value",
        #         torch.tensor(torch.finfo(torch.get_default_dtype()).max),
        #     )
        # else:
        #     self.mask = None
        #     self.register_buffer("scale", torch.tensor(scale))

    def mha(self, inputs: torch.Tensor, output_attn_weights: bool = False):
        inputs = self.layer_norm(inputs)
        q, k, v = torch.chunk(self.to_qkv(inputs), chunks=3, dim=-1)
        if output_attn_weights:
            q_h, k_h, v_h = self.rearrange(q), self.rearrange(k), self.rearrange(v)
            d = q_h.size(-1)
            scale = d ** -0.5

            logits = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale    # [B, H, N, N]
            attn = logits.softmax(dim=-1)                                 # [B, H, N, N]
            attn = self.dropout(attn)

            out_h = torch.matmul(attn, v_h)                               # [B, H, N, D_head]
            outputs = rearrange(out_h, "b h n d -> b n (h d)")            # [B, N, emb_dim]
            outputs = self.projection(outputs)
            attn.retain_grad()
            return outputs, attn
        outputs = scaled_dot_product_attention(
            q=self.rearrange(q), k=self.rearrange(k), v=self.rearrange(v), dropout=self.dropout.p, use_flash_attention=self.use_flash_attention,
        )
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        outputs = self.projection(outputs)
        return outputs

    def forward(self, inputs: torch.Tensor, output_attn_weights: bool = False):
        if self.grad_checkpointing and not output_attn_weights:
            outputs = checkpoint(
                self.mha, inputs, preserve_rng_state=True, use_reentrant=False
            )
            return outputs
        if output_attn_weights:
            outputs, attention_weights = self.mha(inputs, output_attn_weights)
            return outputs, attention_weights
        
        outputs = self.mha(inputs, output_attn_weights=output_attn_weights)
        return outputs


class Transformer(nn.Module):
    def __init__(
        self,
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
        # self.output_shape = (num_tokens, emb_dim) # (num_tokens, emb_dim)
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
        output_attn_weights: bool = False,
        return_layer_tokens: bool = False,
    ):
        outputs = inputs
        attn_list = []
        layer_tokens = []
        for block in self.blocks:
            if "b-mlp" in block:
                b_latent = block["b-mlp"](behaviors, mouse_id=mouse_id)
                b_latent = repeat(b_latent, "b d -> b 1 d")
                outputs = outputs + b_latent
            if output_attn_weights:
                y, attn = block["mha"](outputs, output_attn_weights=output_attn_weights)
                outputs = self.drop_path(y) + outputs       
                attn_list.append(attn)       
            else:
                outputs = self.drop_path(block["mha"](outputs)) + outputs
            outputs = self.drop_path(block["mlp"](outputs)) + outputs

            if return_layer_tokens:
                layer_tokens.append(outputs)
        # outputs: (1, num_tokens, emb_dim)
        if output_attn_weights:
            # stack along layer dim -> [B, L, H, N, N]
            attn_weights = torch.stack(attn_list, dim=1)
            if return_layer_tokens:
                return outputs, attn_weights, layer_tokens  # list of length L, each [B, S, C]
            return outputs, attn_weights
        if return_layer_tokens:
            return outputs, layer_tokens
        return outputs


@register("vit")
class ViTCore(Core):
    def __init__(
        self,
        args,
        num_image_patches: int,
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

        self.transformer = Transformer(
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
            self.image_projection = nn.Linear(in_features=args.emb_dim_image, out_features=args.emb_dim_core)
        
        if args.tokenize_neurons:
            self.use_mode_emb = args.use_mode_emb
            self.mode_embedding = nn.Embedding(args.num_modes, args.emb_dim_core)
            # nn.init.constant_(self.embedding.weight, 1.0 / args.emb_dim_core)
            self.project_neuron = args.emb_dim_input_neurons != args.emb_dim_core
            if self.project_neuron:
                self.neuron_projection = nn.Linear(args.emb_dim_input_neurons, args.emb_dim_core)
            
        self.subselect_image_tokens = args.subselect_image_tokens
        # calculate latent height and width based on num_patches
        if self.readout == "gaussian2d":
            h, w = self.find_shape(num_image_patches)
            # self.output_shape = (self.transformer.output_shape[-1], h, w)
            self.rearrange = Rearrange("b (h w) c -> b c h w", h=h, w=w)
        # else:
        #     if self.subselect_image_tokens:
        #         self.output_shape = (num_image_patches, self.transformer.output_shape[-1]) # self.transformer.output_shape is (num_neuron_tokens+num_image_tokens, emb_dim)
        #     else:
        #         self.output_shape = self.transformer.output_shape # (num_neuron_tokens+num_image_tokens, emb_dim)



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
        output_attn_weights: bool = False,
        return_layer_tokens: bool = False,
    ):
        if self.project_image:
            image_tokens = self.image_projection(image_tokens)
        
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
        if output_attn_weights:
            if return_layer_tokens:
                outputs, attn_weights, layer_tokens = self.transformer(
                    outputs, mouse_id=mouse_id, behaviors=behaviors, output_attn_weights=output_attn_weights, return_layer_tokens=return_layer_tokens
                )
            else:
                outputs, attn_weights = self.transformer(outputs, mouse_id=mouse_id, behaviors=behaviors, output_attn_weights=output_attn_weights)
        else:
            if return_layer_tokens:
                outputs, layer_tokens = self.transformer(
                    outputs, mouse_id=mouse_id, behaviors=behaviors, return_layer_tokens=return_layer_tokens
                )
            else:
                outputs = self.transformer(outputs, mouse_id=mouse_id, behaviors=behaviors, output_attn_weights=output_attn_weights)
        # subselect only image patches because they represent 'receptive fields' of neurons and can be optionally conditioned on input neuron tokens
        if self.subselect_image_tokens:
            outputs = outputs[:, :self.num_image_patches, :] 
        # outputs = outputs[:, 1:, :]  # remove CLS token
        if self.readout == "gaussian2d":
            outputs = self.rearrange(outputs)
        
        if output_attn_weights and return_layer_tokens:
            return outputs, attn_weights, layer_tokens
        elif output_attn_weights:
            return outputs, attn_weights
        elif return_layer_tokens:
            return outputs, layer_tokens
        return outputs

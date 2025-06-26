
import torch
import numpy as np
import typing as t
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, einsum
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel
import warnings

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout: float=0.0, use_flash_attention: bool = False):
    """ 
    q: [B, H, N, D_head]
    k, v: [B, H, S, D_head]
    """
    if q.device.type == "cuda" and use_flash_attention:
        # Dispatch through FlashAttention (or fall back) via PyTorch’s 
        # print("Using FlashAttention")
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout,
                is_causal=False
            )
    else:
        # print("Using standard attention")
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout,
            is_causal=False
        )            
    # out shape is [B, H, N, D_head]
    return out


class PreCoreAttention(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_bias: bool = True,
        grad_checkpointing: bool = True,
        scale: bool = False,
        use_flash_attention: bool = False,
    ):
        super(PreCoreAttention, self).__init__()

        self.grad_checkpointing = grad_checkpointing
        self.use_flash_attention = use_flash_attention
        self.use_bias = use_bias

        self.emb_dim = emb_dim
        self.heads = num_heads

        self.saved_attn_scores = []

        if scale:
            dim_head = self.emb_dim // self.heads
            self.scale = dim_head ** -0.5
        else:
            self.scale = 1.0

        # LayerNorm for queries and inputs
        self.layer_norm = nn.LayerNorm(self.emb_dim)

        self.to_qkv = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim * 3, bias=use_bias)
        
        self.heads_rearrange = Rearrange("b n (h d) -> b h n d", h=num_heads)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)


        if scale:
            self.register_buffer("scale_readout", torch.tensor(self.scale))


    def mha(self, tokens: torch.Tensor, save_scores: bool = False):
        tokens = self.layer_norm(tokens) # [B, N_input_neurons, emb_dim]
        q, k, v = torch.chunk(self.to_qkv(tokens), chunks=3, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        if save_scores:
            attn_scores = einsum("b h n d, b h m d -> b h n m", q, k) * self.scale
            attn_scores = self.attend(attn_scores)
            outputs = einsum("b h n m, b h m d -> b h n d", attn_scores, v)
            self.input_neurons_attn_scores.append(attn_scores.detach().cpu())
            

        outputs = scaled_dot_product_attention(q=q, k=k, v=v, dropout=self.dropout.p, use_flash_attention=self.use_flash_attention)
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        return outputs

    def forward(self, tokens: torch.Tensor, save_scores: bool = False):
        if self.grad_checkpointing:
            outputs = checkpoint(self.mha, tokens, preserve_rng_state=True, use_reentrant=False)
        else:
            outputs = self.mha(tokens, save_scores=save_scores)
        return outputs

class PreCoreTransformer(nn.Module):
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




    # def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    #     """ 
    #     q: [B, H, N, D_head]
    #     k, v: [B, H, S, D_head]
    #     """
    #     if q.device.type == "cuda" and self.use_flash_attention: #and q.dtype in (torch.float16,torch.bfloat16):
    #         with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    #             return F.scaled_dot_product_attention(
    #                 q, k, v,
    #                 attn_mask=self.mask,
    #                 dropout_p=self.dropout.p,
    #                 is_causal=False,
    #             )
    #     else:
    #         # print("Using standard attention in core")
    #         return F.scaled_dot_product_attention(
    #         q, k, v,
    #         attn_mask=self.mask,
    #         dropout_p=self.dropout.p,
    #         is_causal=False,
    #         )
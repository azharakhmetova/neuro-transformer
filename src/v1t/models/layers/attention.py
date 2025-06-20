
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

        if save_scores:
            attn_scores = einsum("b h n d, b h m d -> b h n m", q, k) * self.scale
            attn_scores = self.attend(attn_scores)
            outputs = einsum("b h n m, b h m d -> b h n d", attn_scores, v)
            self.input_neurons_attn_scores.append(attn_scores.detach().cpu())
            
        outputs = scaled_dot_product_attention(
            q=self.rearrange(q), k=self.rearrange(k), v=self.rearrange(v), dropout=self.dropout.p, use_flash_attention=self.use_flash_attention,
        )
        outputs = scaled_dot_product_attention(q=q, k=k, v=v, dropout=self.dropout.p, use_flash_attention=self.use_flash_attention)
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        return outputs

    def forward(self, tokens: torch.Tensor, save_scores: bool = False):
        if self.grad_checkpointing:
            outputs = checkpoint(self.mha, tokens, preserve_rng_state=True, use_reentrant=False)
        else:
            outputs = self.mha(tokens, save_scores=save_scores)
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
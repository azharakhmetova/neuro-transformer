
import torch
import numpy as np
import typing as t
from torch.nn import functional as F
from torch.nn import functional as F
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


    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """ 
        q: [B, H, N, D_head]
        k, v: [B, H, S, D_head]
        """
        if q.device.type == "cuda" and self.use_flash_attention: #and q.dtype in (torch.float16,torch.bfloat16):
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                return F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=self.mask,
                    dropout_p=self.dropout.p,
                    is_causal=False,
                )
        else:
            # print("Using standard attention in core")
            return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=self.mask,
            dropout_p=self.dropout.p,
            is_causal=False,
            )
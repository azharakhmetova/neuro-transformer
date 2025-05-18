import torch
import numpy as np
import typing as t
import math
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, einsum
from torch.utils.checkpoint import checkpoint
import warnings

class ModeTokenizer(nn.Module):
    def __init__(
        self, 
        num_modes: int, 
        emb_dim: int,
    ):
        super(ModeTokenizer, self).__init__()
        self.embedding = nn.Embedding(num_modes, emb_dim)
        nn.init.constant_(self.embedding.weight, 1.0 / emb_dim)

    def forward(self, mode: torch.Tensor):
        if mode.dtype != torch.long:
            mode = mode.long()
        return self.embedding(mode)


class PositionalEncoding(nn.Module):
    """
    1D & 2D sinusoidal or learned positional encoding for
    - 3D input: (B, C, L)
    - 4D input: (B, C, H, W)

    Compute only the required encoding (1D, 2D, or both) depending on 'mode'.

    Args:
        d_model: embedding dimension (must be even for 2D)
        dropout: dropout rate
        max_len: maximum length for 1D sequence
        height: fixed height for 2D grid (defaults to max_len)
        width: fixed width for 2D grid (defaults to max_len)
        learned: if True, use learnable parameters instead of sinusoidal
        mode: '1d', '2d', or 'both'
    """
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        height: int = None,
        width: int = None,
        learned: bool = False,
        mode: str = 'both',  # '1d', '2d', or 'both'
    ):
        super().__init__()
        if mode in ('2d', 'both') and d_model % 2 != 0:
            raise ValueError("d_model must be even for 2D positional encoding")
        if mode not in ('1d', '2d', 'both'):
            raise ValueError("mode must be '1d', '2d', or 'both'")
        self.dropout = nn.Dropout(dropout)
        self.learned = learned
        self.mode = mode

        # Determine spatial size for 2D
        H = max_len if height is None else height
        W = max_len if width is None else width
        # Learnable parameters
        if learned:
            if mode in ('1d', 'both'):
                self.pe1 = nn.Parameter(torch.randn(1, max_len, d_model))
            if mode in ('2d', 'both'):
                self.pe2 = nn.Parameter(torch.randn(1, H, W, d_model))
        else:
            # Sinusoidal buffers
            if mode in ('1d', 'both'):
                pe1 = torch.zeros(max_len, d_model)
                pos1 = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
                div1 = torch.exp(
                    torch.arange(0, d_model, 2, dtype=torch.float)
                    * (-math.log(10000.0) / d_model)
                )
                pe1[:, 0::2] = torch.sin(pos1 * div1)
                pe1[:, 1::2] = torch.cos(pos1 * div1)
                pe1 = pe1.unsqueeze(0)
                self.register_buffer('pe1', pe1)
            if mode in ('2d', 'both'):
                d_half = d_model // 2
                y_pos = torch.arange(H, dtype=torch.float).unsqueeze(1)
                x_pos = torch.arange(W, dtype=torch.float).unsqueeze(1)
                div_h = torch.exp(
                    torch.arange(0, d_half, 2, dtype=torch.float)
                    * (-math.log(10000.0) / d_half)
                )
                div_w = torch.exp(
                    torch.arange(0, d_half, 2, dtype=torch.float)
                    * (-math.log(10000.0) / d_half)
                )
                pe_y = torch.zeros(H, d_half)
                pe_x = torch.zeros(W, d_half)
                pe_y[:, 0::2] = torch.sin(y_pos * div_h)
                pe_y[:, 1::2] = torch.cos(y_pos * div_h)
                pe_x[:, 0::2] = torch.sin(x_pos * div_w)
                pe_x[:, 1::2] = torch.cos(x_pos * div_w)
                pe_y = pe_y.unsqueeze(1).expand(H, W, d_half)
                pe_x = pe_x.unsqueeze(0).expand(H, W, d_half)
                pe2 = torch.cat((pe_y, pe_x), dim=2)
                pe2 = pe2.unsqueeze(0)
                self.register_buffer('pe2', pe2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if self.mode not in ('1d', 'both'):
                raise RuntimeError("Module configured without 1D mode but got 3D input")
            L = x.size(1)
            x = x + self.pe1[:, :L, :]
        elif x.dim() == 4 and self.mode == '1d':
            L = x.size(2)
            x = x + self.pe1[:, :L, :].unsqueeze(0)
        elif x.dim() == 4:
            if self.mode not in ('2d', 'both'):
                raise RuntimeError("Module configured without 2D mode but got 4D input")
            H, W = x.size(1), x.size(2)
            x = x + self.pe2[:, :H, :W, :]
        else:
            raise ValueError("Input tensor must be 3D or 4D")
        return self.dropout(x)
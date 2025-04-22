import torch
import numpy as np
import typing as t
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, einsum
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import warnings

from v1t.models.utils import DropPath
from v1t.models.neuron_processing import NeuronIDTokenizer

REDUCTIONS = t.Literal["sum", "mean", None]

class CrossAttention(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        num_neurons: int,
        emb_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_lsa: bool = False,
        use_bias: bool = True,
        grad_checkpointing: bool = True,
        scale: bool = False,
        key_embedding: bool = False,
        value_embedding: bool = False,
        use_layer_norm: bool = False,
        use_pos_embedding: bool = True,
        temperature: tuple = (False, 1.0)
    ):
        super(CrossAttention, self).__init__()

        self.grad_checkpointing = grad_checkpointing
        self.use_pos_embedding = use_pos_embedding
        self.key_embedding = key_embedding
        self.value_embedding = value_embedding
        self.use_bias = use_bias

        self.input_shape = input_shape
        self.num_neurons = num_neurons
        self.emb_dim = emb_dim
        self.heads = num_heads

        if scale:
            dim_head = self.emb_dim // self.heads
            self.scale = dim_head ** -0.5
        else:
            self.scale = 1.0

        self.T = (
            temperature[1] if temperature[0]
            else nn.Parameter(torch.ones(self.num_neurons) * temperature[1])
        )

        # LayerNorm for queries and inputs
        self.layer_norm = nn.LayerNorm(self.emb_dim)
        self.layer_norm_inputs = nn.LayerNorm(self.input_shape[0]) #((self.input_shape[1] * self.input_shape[2],self.input_shape[0]))

        # Key/Value projection layer (if enabled)
        if self.key_embedding and self.value_embedding:
            self.to_kv = nn.Linear(in_features=self.input_shape[0], out_features=self.emb_dim * 2, bias=False)
        elif self.key_embedding:
            self.to_key = nn.Linear(in_features=self.input_shape[0], out_features=self.emb_dim, bias=False)

        # Optional positional embedding
        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.input_shape[1] * self.input_shape[2], self.input_shape[0])
        ) if use_pos_embedding else None

        # Reshaping utility for attention
        self.heads_rearrange = Rearrange("b n (h d) -> b h n d", h=num_heads)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        self.projection = nn.Sequential(
            nn.Linear(in_features=dim_head, out_features=emb_dim, bias=use_bias),
            nn.Dropout(p=dropout),
        )

        if scale:
            self.register_buffer("scale_readout", torch.tensor(self.scale))

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        outputs = einsum(attn, v, "b h n i, b h i d -> b h n d")
        return outputs

    # def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    #     """ 
    #     q: [B, H, N, D_head]
    #     k, v: [B, H, S, D_head]
    #     """
    #     # Dispatch through FlashAttention (or fall back) via PyTorch’s sdpa_kernel
    #     with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    #         out = F.scaled_dot_product_attention(
    #             q, k, v,
    #             attn_mask=None,
    #             dropout_p=self.dropout.p,
    #             is_causal=False
    #         )
    #     # out shape is [B, H, N, D_head]
    #     return out

    def mha(self, q: torch.Tensor, inputs: torch.Tensor):
        q = self.layer_norm(q)
        inputs = self.layer_norm_inputs(inputs)

        if self.use_pos_embedding:
            inputs_pos = inputs + self.positional_embedding
        else:
            inputs_pos = inputs

        b = inputs.size(0)

        # Compute keys and values depending on configuration
        if self.key_embedding and self.value_embedding:
            key, value = self.to_kv(rearrange(inputs, "b s c -> (b s) c")).chunk(2, dim=-1)
            k = rearrange(key, "(b s) (h d) -> b h s d", h=self.heads, b=b)
            v = rearrange(value, "(b s) (h d) -> b h s d", h=self.heads, b=b)
        elif self.key_embedding:
            key = self.to_key(rearrange(inputs_pos, "b s c -> (b s) c"))
            k = rearrange(key, "(b s) (h d) -> b h s d", h=self.heads, b=b)
            v = rearrange(inputs, "b s (h d) -> b h s d", h=self.heads)
        else:
            k = rearrange(inputs_pos, "b s (h d) -> b h s d", h=self.heads)
            v = rearrange(inputs, "b s (h d) -> b h s d", h=self.heads)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        outputs = self.scaled_dot_product_attention(q=q, k=k, v=v)
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        return outputs
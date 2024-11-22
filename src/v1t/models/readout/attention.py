from .readout import register, Readout

import torch
import numpy as np
import typing as t
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import math
import torch
import typing as t
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, einsum
from torch.utils.checkpoint import checkpoint

from v1t.models.utils import DropPath

REDUCTIONS = t.Literal["sum", "mean", None]


class CrossAttention(nn.Module):
    def __init__(
        self,
        num_patches: int,
        emb_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_lsa: bool = False,
        use_bias: bool = True,
        grad_checkpointing: bool = False,
        key_embedding: bool = False,
        value_embedding: bool = False,
        scale: bool = False,
    ):
        super(CrossAttention, self).__init__()
        self.grad_checkpointing = grad_checkpointing
        self.key_embedding = key_embedding
        self.value_embedding = value_embedding

        inner_dim = emb_dim * num_heads

        self.layer_norm_q = nn.LayerNorm(emb_dim)  # Normalize queries
        self.layer_norm_kv = nn.LayerNorm(emb_dim)  # Normalize keys/values

        self.to_q = nn.Linear(
            in_features=emb_dim, out_features=inner_dim, bias=False
        )  
        if self.key_embedding and self.value_embedding:
            self.to_kv = nn.Linear(
                in_features=emb_dim, out_features=inner_dim * 2, bias=False
            ) 
        elif self.key_embedding:
            self.to_k = nn.Linear(
                in_features=emb_dim, out_features=inner_dim, bias=False
            )
        else:
            self.to_kv = None

        self.rearrange = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        self.projection = nn.Sequential(
            nn.Linear(in_features=inner_dim, out_features=emb_dim, bias=use_bias),
            nn.Dropout(p=dropout),
        )

        # Scaling parameter
        self.scale = (emb_dim**-0.5) if scale else 1.0
        if scale:
            self.register_buffer("scale_readout", torch.tensor(self.scale))

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        """
        Compute scaled dot-product attention.
        """
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        outputs = einsum(attn, v, "b h n i, b h i d -> b h n d")
        return outputs

    def mha(self, q_inputs: torch.Tensor, kv_inputs: torch.Tensor):
        """
        Multi-head attention for cross-attention.
        """
        q_inputs = self.layer_norm_q(q_inputs)
        kv_inputs = self.layer_norm_kv(kv_inputs)

        q = self.to_q(q_inputs)  # [B, N_q, D]
        if self.key_embedding and self.value_embedding:
            k, v = torch.chunk(self.to_kv(kv_inputs), chunks=2, dim=-1)  # [B, P, D], [B, P, D]
        elif self.key_embedding:
            k = self.to_k(kv_inputs) 
            v = kv_inputs  
        else:
            k, v = kv_inputs, kv_inputs

        q = self.rearrange(q)  # [B, N_q, D] -> [B, H, N_q, D_h]
        k = self.rearrange(k)  # [B, P, D] -> [B, H, P, D_h]
        v = self.rearrange(v)  # [B, P, D] -> [B, H, P, D_h]

        outputs = self.scaled_dot_product_attention(q=q, k=k, v=v)  # [B, H, N_q, D_h]
        outputs = rearrange(outputs, "b h n d -> b n (h d)")  # [B, N_q, D]
        return self.projection(outputs)  # Project back to original embedding space

    def forward(self, q_inputs: torch.Tensor, kv_inputs: torch.Tensor):
        """
        Forward pass for cross-attention:
        - q_inputs: Queries (e.g., neuron-specific queries)
        - kv_inputs: Keys and values (e.g., feature maps from ViT)
        """
        if self.grad_checkpointing:
            outputs = checkpoint(
                self.mha, q_inputs, kv_inputs, preserve_rng_state=True, use_reentrant=False
            )
        else:
            outputs = self.mha(q_inputs, kv_inputs)
        return outputs



@register("attention")
class AttentionReadout(Readout):
    def __init__(
        self,
        args,
        input_shape: tuple,
        output_shape: tuple,
        ds: DataLoader,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_lsa: bool = False,
        use_bias: bool = True,
        scale: bool = False,
        key_embedding: bool = False,
        value_embedding: bool = False,
        grad_checkpointing: bool = False,
        name: str = "AttentionReadout",
    ):
        super(AttentionReadout, self).__init__(
            args,
            input_shape=input_shape,
            output_shape=output_shape,
            ds=ds,
            name=name,
        )

        emb_dim = input_shape[1] #[B, C, H, W]
        num_patches = input_shape[-1]*input_shape[-2]  

        num_neurons = self.num_neurons

        self.cross_attention = CrossAttention(
            num_patches=num_patches,
            emb_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_lsa=use_lsa,
            use_bias=use_bias,
            grad_checkpointing=grad_checkpointing,
            key_embedding=key_embedding,
            value_embedding=value_embedding,
            scale=scale,
        )

        self.neuron_projection = nn.Linear(in_features=emb_dim, out_features=num_neurons, bias=True)

    def forward(self, inputs: torch.Tensor, shifts: torch.Tensor = None):

        neuron_queries = self.generate_neuron_queries(inputs)

        outputs = self.cross_attention(
            q_inputs=neuron_queries, kv_inputs=inputs
        )  # [B, N_neurons, D]

        outputs = self.neuron_projection(outputs)  # [B, N_neurons]

        return outputs

    def generate_neuron_queries(self, inputs: torch.Tensor):
        """
        Generate neuron queries:
        - Shape: [B, N_neurons, D]
        """
        batch_size = inputs.size(0)
        emb_dim = inputs.size(-1)

        # Assuming fixed queries per neuron initialized randomly
        if not hasattr(self, "neuron_queries"):
            self.register_parameter(
                "neuron_queries", nn.Parameter(torch.randn(1, self.num_neurons, emb_dim))
            )

        # Expand neuron queries to match batch size
        return self.neuron_queries.expand(batch_size, -1, -1)

    def regularizer(self, reduction: str = "sum"):
        """
        L1 Regularization on learnable neuron queries.
        """
        l1_reg = self.neuron_queries.abs().sum()
        return self.reg_scale * l1_reg




# class CrossAttention(nn.Module):
#     def __init__(
#         self,
#         num_patches: int,
#         emb_dim: int,
#         num_heads: int = 8,
#         dropout: float = 0.0,
#         use_lsa: bool = False,
#         use_bias: bool = True,
#         grad_checkpointing: bool = False,
#     ):
#         super(CrossAttention, self).__init__()
#         self.grad_checkpointing = grad_checkpointing
#         inner_dim = emb_dim * num_heads

#         self.layer_norm_q = nn.LayerNorm(emb_dim)  
#         self.layer_norm_kv = nn.LayerNorm(emb_dim)  

#         self.to_q = nn.Linear(
#             in_features=emb_dim, out_features=inner_dim, bias=False
#         )  # For queries
#         self.to_kv = nn.Linear(
#             in_features=emb_dim, out_features=inner_dim * 2, bias=False
#         )  # For keys and values

#         self.rearrange = Rearrange("b n (h d) -> b h n d", h=num_heads)
#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(p=dropout)

#         self.projection = nn.Sequential(
#             nn.Linear(in_features=inner_dim, out_features=emb_dim, bias=use_bias),
#             nn.Dropout(p=dropout),
#         )

#         scale = emb_dim**-0.5
#         if use_lsa:
#             self.register_parameter(
#                 "scale",
#                 param=nn.Parameter(torch.full(size=(num_heads,), fill_value=scale)),
#             )
#             diagonal = torch.eye(num_patches, num_patches)
#             self.register_buffer(
#                 "mask",
#                 torch.nonzero(diagonal == 1, as_tuple=False),
#             )
#             self.register_buffer(
#                 "max_value",
#                 torch.tensor(torch.finfo(torch.get_default_dtype()).max),
#             )
#         else:
#             self.mask = None
#             self.register_buffer("scale", torch.tensor(scale))

#     def scaled_dot_product_attention(
#         self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
#     ):
#         """
#         Compute scaled dot-product attention.
#         """
#         if self.mask is None:
#             dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         else:
#             scale = repeat(self.scale, "h -> b h 1 1", b=q.size(0))
#             dots = torch.matmul(q, k.transpose(-1, -2)) * scale
#             dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -self.max_value
#         attn = self.attend(dots)
#         attn = self.dropout(attn)
#         outputs = einsum(attn, v, "b h n i, b h i d -> b h n d")
#         return outputs

#     def mha(self, q_inputs: torch.Tensor, kv_inputs: torch.Tensor):
#         """
#         Multi-head attention for cross-attention
#         """
#         # # Normalize queries and key/value inputs
#         # q_inputs = self.layer_norm_q(q_inputs)
#         # kv_inputs = self.layer_norm_kv(kv_inputs)

#         # Generate queries, keys, and values
#         q = self.to_q(q_inputs)  # [B, N_q, D]
#         k, v = torch.chunk(self.to_kv(kv_inputs), chunks=2, dim=-1)  # [B, P, D], [B, P, D]

#         # Rearrange for multi-head attention
#         q = self.rearrange(q)  # [B, N_q, D] -> [B, H, N_q, D_h]
#         k = self.rearrange(k)  # [B, P, D] -> [B, H, P, D_h]
#         v = self.rearrange(v)  # [B, P, D] -> [B, H, P, D_h]

#         # Compute attention
#         outputs = self.scaled_dot_product_attention(q=q, k=k, v=v)  # [B, H, N_q, D_h]
#         outputs = rearrange(outputs, "b h n d -> b n (h d)")  # [B, N_q, D]
#         return self.projection(outputs)  # Project back to original embedding space

#     def forward(self, q_inputs: torch.Tensor, kv_inputs: torch.Tensor):
#         """
#         Forward pass for cross-attention:
#         - q_inputs: Queries (e.g., neuron-specific queries)
#         - kv_inputs: Keys and values (e.g., feature maps from ViT)
#         """
#         if self.grad_checkpointing:
#             outputs = checkpoint(
#                 self.mha, q_inputs, kv_inputs, preserve_rng_state=True, use_reentrant=False
#             )
#         else:
#             outputs = self.mha(q_inputs, kv_inputs)
#         return outputs


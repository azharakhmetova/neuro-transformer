from .readout import register, Readout

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
        self.layer_norm_inputs = nn.LayerNorm(self.input_shape[0])

        # Key/Value projection layer (if enabled)
        if self.key_embedding and self.value_embedding:
            self.to_kv = nn.Linear(in_features=self.input_shape[0], out_features=self.emb_dim * 2, bias=False)
        elif self.key_embedding:
            self.to_key = nn.Linear(in_features=self.input_shape[0], out_features=self.emb_dim, bias=False)

        # Optional positional embedding
        self.positional_embedding = nn.Parameter(
            torch.randn(1, input_shape[1] * input_shape[2], input_shape[0])
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
    #     if q.device.type == "cuda":
    #         # Dispatch through FlashAttention (or fall back) via PyTorch’s sdpa_kernel
    #         with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    #             out = F.scaled_dot_product_attention(
    #                 q, k, v,
    #                 attn_mask=None,
    #                 dropout_p=self.dropout.p,
    #                 is_causal=False
    #             )
    #     else:
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

    def forward(self, q: torch.Tensor, inputs: torch.Tensor):
        if self.grad_checkpointing:
            outputs = checkpoint(self.mha, q, inputs, preserve_rng_state=True, use_reentrant=False)
        else:
            outputs = self.mha(q, inputs)
        return outputs


class NeuronTokenizer(nn.Module):
    def __init__(self, num_neurons: int, emb_dim: int):
        super(NeuronTokenizer, self).__init__()
        self.embedding = nn.Embedding(num_neurons, emb_dim)
        nn.init.constant_(self.embedding.weight, 1.0 / emb_dim)

    def forward(self, neuron_ids: torch.Tensor):
        return self.embedding(neuron_ids)


@register("attention")
class AttentionReadout(Readout):
    def __init__(
        self,
        args,
        input_shape: tuple,
        output_shape: tuple,
        ds: DataLoader,
        num_heads: int = 2,
        dropout: float = 0.0,
        use_lsa: bool = False,
        use_bias: bool = True,
        scale: bool = True,
        key_embedding: bool = True,
        value_embedding: bool = True,
        grad_checkpointing: bool = False,
        use_layer_norm: bool = True,
        use_pos_embedding: bool = True,
        temperature: tuple = (False, 1.0),
        name: str = "AttentionReadout",
    ):
        super(AttentionReadout, self).__init__(
            args, input_shape=input_shape, output_shape=output_shape, ds=ds, name=name
        )

        emb_dim = 160  # embedding dimension for readout

        self.cross_attention = CrossAttention(
            input_shape=input_shape,
            num_neurons=self.num_neurons,
            emb_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_lsa=use_lsa,
            use_bias=use_bias,
            grad_checkpointing=grad_checkpointing,
            key_embedding=key_embedding,
            value_embedding=value_embedding,
            scale=scale,
            temperature=temperature,
            use_pos_embedding=use_pos_embedding
        )

        self.dropout = nn.Dropout(p=dropout)
        self.neuron_tokenizer = NeuronTokenizer(num_neurons=self.num_neurons, emb_dim=emb_dim)
        self.neuron_projection = nn.Linear(in_features=emb_dim, out_features=1, bias=True)

    def forward(self, inputs: torch.Tensor, neuron_ids: torch.Tensor = None, query_neuron_subset: bool = False, shifts: torch.Tensor = None): 
        b, c, w, h = inputs.size()
        c_in, w_in, h_in = self.input_shape

        if (c_in, w_in, h_in) != (c, w, h):
            warnings.warn("Mismatch between expected and actual input shape.")

        if not query_neuron_subset:
            neuron_ids = torch.arange(self.num_neurons, device=inputs.device).unsqueeze(0).expand(b, -1)
        else:
            neuron_ids = torch.tensor(neuron_ids, device=inputs.device).unsqueeze(0).expand(b, -1)

        neuron_queries = self.neuron_tokenizer(neuron_ids)
        neuron_queries = self.dropout(neuron_queries)

        inputs = rearrange(inputs, 'b c h w -> b (h w) c')  # flatten spatial dims

        outputs = self.cross_attention(q=neuron_queries, inputs=inputs)
        outputs = self.dropout(outputs)
        outputs = self.neuron_projection(outputs).squeeze(-1)  # [B, N_neurons]

        return outputs

    def feature_l1(self, reduction: str = "sum"):
        l1 = self.neuron_tokenizer.embedding.weight.abs()
        return l1.sum() if reduction == "sum" else l1.mean()

    def regularizer(self, reduction: str = "sum"):
        reg_term = self.reg_scale * self.feature_l1(reduction=reduction)

        l1 = self.neuron_projection.weight.abs()
        l1 = l1.sum() if reduction == "sum" else l1.mean()

        if self.neuron_projection.bias is not None:
            bias = self.neuron_projection.bias.abs()
            l1 += bias.sum() if reduction == "sum" else bias.mean()

        reg_term += self.reg_scale * l1
        return reg_term

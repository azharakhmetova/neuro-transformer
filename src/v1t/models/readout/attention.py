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

from v1t.models.layers import scaled_dot_product_attention
from v1t.models.utils import DropPath

REDUCTIONS = t.Literal["sum", "mean", None]
"""
code reference: https://github.com/KonstantinWilleke/neuralpredictors/blob/4ef51533f948970e511ee6061711db25e5e52217/neuralpredictors/layers/attention_readout.py
"""
class CrossAttention(nn.Module):
    def __init__(
        self,
        # input_shape: tuple,
        num_neurons: int,
        emb_dim: int,
        emb_dim_core: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_lsa: bool = False,
        use_bias: bool = True,
        grad_checkpointing: bool = True,
        key_embedding: bool = False,
        value_embedding: bool = False,
        subselect_image_tokens: bool = False,
        use_pos_embedding: bool = False,
        use_flash_attention: bool = False
    ):
        super(CrossAttention, self).__init__()

        self.grad_checkpointing = grad_checkpointing
        self.use_flash_attention = use_flash_attention
        self.use_pos_embedding = use_pos_embedding
        self.key_embedding = key_embedding
        self.value_embedding = value_embedding
        self.use_bias = use_bias

        # self.input_shape = input_shape # (t, c)
        # print("CrossAttention input shape: ", self.input_shape)
        self.num_neurons = num_neurons
        self.emb_dim = emb_dim
        self.heads = num_heads

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        dim_head = self.emb_dim // self.heads

        # Optional positional embedding
        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.input_shape[0], self.input_shape[1])
        ) if use_pos_embedding else None

        # LayerNorm for queries and inputs
        self.layer_norm = nn.LayerNorm(self.emb_dim)
        self.layer_norm_inputs = nn.LayerNorm(emb_dim_core)

        # Key/Value projection layer (if enabled)
        if self.key_embedding and self.value_embedding:
            self.to_kv = nn.Linear(in_features=emb_dim_core, out_features=self.emb_dim * 2, bias=use_bias)
        elif self.key_embedding:
            self.to_key = nn.Linear(in_features=emb_dim_core, out_features=self.emb_dim, bias=use_bias)

        # Optional positional embedding
        # self.positional_embedding = nn.Parameter(
        #     torch.randn(1, self.input_shape[0], self.input_shape[1])
        # ) if use_pos_embedding else None
        # print("CrossAttention positional embedding: ", self.positional_embedding.shape)

        # Reshaping utility for attention
        self.heads_rearrange = Rearrange("b n (h d) -> b h n d", h=num_heads)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)


    def mha(self, q: torch.Tensor, inputs: torch.Tensor, output_attn_weights: bool = False):
        # q_res = q
        q = self.layer_norm(q) # [B, N_query_neurons, emb_dim]
        inputs = self.layer_norm_inputs(inputs) # [B, num_image_tokens, num_channels]

        if self.use_pos_embedding and self.subselect_image_tokens:
            inputs_pos = inputs + self.positional_embedding
        else:
            inputs_pos = inputs

        b = inputs.size(0)

        # Compute keys and values depending on configuration
        if self.key_embedding and self.value_embedding:
            key, value = self.to_kv(rearrange(inputs_pos, "b s c -> (b s) c")).chunk(2, dim=-1)
            k = rearrange(key, "(b s) (h d) -> b h s d", h=self.heads, b=b)
            v = rearrange(value, "(b s) (h d) -> b h s d", h=self.heads, b=b)
        elif self.key_embedding:
            key = self.to_key(rearrange(inputs_pos, "b s c -> (b s) c"))
            k = rearrange(key, "(b s) (h d) -> b h s d", h=self.heads, b=b)
            v = rearrange(inputs, "b s (h d) -> b h s d", h=self.heads)
        elif self.to_value_embedding:
            value = self.to_value(rearrange(inputs_pos, "b s c -> (b s) c"))
            k = rearrange(inputs_pos, "(b s) (h d) -> b h s d", h=self.heads, b=b)
            v = rearrange(value, "(b s) (h d) -> b h s d", h=self.heads, b=b)
        else:
            k = rearrange(inputs_pos, "b s (h d) -> b h s d", h=self.heads)
            v = rearrange(inputs, "b s (h d) -> b h s d", h=self.heads)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        outputs = scaled_dot_product_attention(q=q, k=k, v=v, dropout=self.dropout.p, use_flash_attention=self.use_flash_attention)
        outputs = rearrange(outputs, "b h n d -> b n (h d)") # [B, N_query_neurons, emb_dim]
        # outputs = outputs + q_res
        if output_attn_weights:
            logits = torch.matmul(q, k.transpose(-1, -2)) * (k.size(-1) ** -0.5)
            attention_weights = logits.softmax(dim=-1)
            return outputs, attention_weights
        return outputs

    def forward(self, q: torch.Tensor, inputs: torch.Tensor, output_attn_weights: bool = False):
        if self.grad_checkpointing and not output_attn_weights:
            outputs = checkpoint(self.mha, q, inputs, preserve_rng_state=True, use_reentrant=False)
            return outputs
        if output_attn_weights:
            outputs, attention_weights = self.mha(q, inputs, output_attn_weights=output_attn_weights)
            return outputs, attention_weights
        
        outputs = self.mha(q, inputs, output_attn_weights=output_attn_weights)
        return outputs


@register("attention")
class AttentionReadout(Readout):
    def __init__(
        self,
        args,
        # input_shape: tuple, # output of core (num_tokens, num_channels)
        output_shape: tuple,
        ds: DataLoader,
        num_heads: int = 2,
        dropout: float = 0.0,
        use_lsa: bool = False,
        use_bias: bool = True,
        key_embedding: bool = True,
        value_embedding: bool = True,
        grad_checkpointing: bool = False,
        use_pos_embedding: bool = False,
        temperature: tuple = (False, 1.0),
        name: str = "AttentionReadout",
    ):
        super(AttentionReadout, self).__init__(
            args, output_shape=output_shape, ds=ds, name=name
        )

        self.use_bias = use_bias
        self.bias_mode = args.bias_mode

        self.cross_attention = CrossAttention(
            num_neurons=self.num_neurons,
            emb_dim=args.emb_dim_readout,
            emb_dim_core=args.emb_dim_core,
            num_heads=num_heads,
            dropout=dropout,
            use_lsa=use_lsa,
            use_bias=use_bias,
            grad_checkpointing=grad_checkpointing,
            key_embedding=key_embedding,
            value_embedding=value_embedding,
            subselect_image_tokens=args.subselect_image_tokens,
            use_pos_embedding=use_pos_embedding,
            use_flash_attention=args.amp,
        )

        self.dropout = nn.Dropout(p=dropout)
        # self.neuron_tokenizer = NeuronTokenizer(num_neurons=self.num_neurons, emb_dim=emb_dim)
        self.project_query_neurons = args.emb_dim_readout != args.emb_dim_neuron_id
        if self.project_query_neurons:
            self.id_query_projection = nn.Linear(in_features=args.emb_dim_neuron_id, out_features=args.emb_dim_readout, bias=True)

        self.initialize_bias(stats=ds.dataset.response_stats)
        self.neuron_projection = nn.Linear(in_features=args.emb_dim_readout, out_features=1, bias=True)

        # self.features = self.embedding = nn.Embedding(self.num_neurons, args.emb_dim_readout)
        # nn.init.constant_(self.embedding.weight, 1.0 / args.emb_dim_readout)

        # r = 20
        # self.U = nn.Linear(args.emb_dim_readout, r)
        # self.V = nn.Linear(args.emb_dim_readout, r)

    def initialize_bias(self, stats: t.Dict[str, np.ndarray]):
        if self.use_bias:
            if self.bias_mode == 0:
                bias = torch.zeros(size=(len(stats["mean"]),))
            elif self.bias_mode == 1:
                bias = torch.from_numpy(stats["mean"])
            elif self.bias_mode == 2:
                bias = stats["mean"] / stats["std"]
                bias = torch.from_numpy(bias)
            else:
                raise NotImplementedError(
                    f"AttentionReadout: bias mode {self.bias_mode} has not been implemented."
                )
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def feature_l1(self, reduction: str = "sum"):
        l1 = self.features.weight.abs()
        l1 = l1.sum() if reduction == "sum" else l1.mean()        
        return l1

    def regularizer(self, reduction: str = "sum"):
        return self.reg_scale * self.feature_l1(reduction=reduction)


    def forward(
        self, 
        inputs: torch.Tensor, 
        query_neurons: t.Optional[torch.Tensor] = None, 
        query_neuron_ids: t.Optional[torch.Tensor] = None, 
        output_attn_weights: bool = False, 
        shifts: t.Optional[torch.Tensor] = None
    ): 
        b, t, c = inputs.size()
        # print("readout inputs shape: ", inputs.shape) # (B, num_tokens, num_channels)
        # print("readout query_neurons shape: ", query_neurons.shape)
        # t_in, c_in = self.input_shape
        # print("readout self.input_shape: ", inputs.shape) # (B, num_tokens, num_channels)

        # if (c_in, t_in) != (c, t):
        #     warnings.warn("Mismatch between expected and actual input shape.")

        # neuron_ids = torch.arange(self.num_neurons, device=inputs.device).unsqueeze(0).expand(b, -1)
        # if neuron_ids is None:
        #     neuron_ids = torch.arange(self.num_neurons, device=inputs.device).unsqueeze(0).expand(b, -1)
        # else:
        #     neuron_ids = neuron_ids.clone().to(inputs.device).unsqueeze(0).expand(b, -1)

        if self.project_query_neurons:
            query_neurons = self.id_query_projection(query_neurons)
        # query_neurons = self.dropout(query_neurons) # [B, N_neurons, emb_dim]
        
        if output_attn_weights:
            outputs, attn_weights = self.cross_attention(q=query_neurons, inputs=inputs, output_attn_weights=output_attn_weights)
        else:
            outputs = self.cross_attention(q=query_neurons, inputs=inputs)
        # outputs = self.dropout(outputs) # [B, N_neurons, emb_dim]
        outputs = self.neuron_projection(outputs).squeeze(-1)  # [B, N_neurons]

        # outputs = outputs * self.features(query_neuron_ids)
        # outputs = torch.sum(outputs, dim=-1) # [B, N_neurons]

        # outputs = outputs * query_neurons
        # outputs = torch.sum(outputs, dim=-1) # [B, N_neurons]

        # outputs= (self.U(outputs) * self.V(query_neurons)).sum(-1)

        bias = self.bias
        if bias is not None:
            outputs += bias[query_neuron_ids]
        
        if output_attn_weights:
            return outputs, attn_weights
        return outputs

    

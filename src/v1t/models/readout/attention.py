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
        use_pos_embedding: bool = True
    ):
        super(CrossAttention, self).__init__()
        self.grad_checkpointing = grad_checkpointing
        self.use_pos_embedding = use_pos_embedding

        inner_dim = emb_dim * num_heads

        self.layer_norm_q = nn.LayerNorm((num_neurons, emb_dim))  
        self.layer_norm_kv = nn.LayerNorm((input_shape[1]*input_shape[2], emb_dim)) 

        self.to_q = nn.Linear(
            in_features=emb_dim, out_features=inner_dim, bias=False
        )  
        self.to_kv = nn.Linear(
                in_features=emb_dim, out_features=inner_dim * 2, bias=False
            ) 
        
        self.positional_embedding = nn.Parameter(
            torch.randn(1, input_shape[1]*input_shape[2], emb_dim)
        ) if use_pos_embedding else None
        
        self.rearrange = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        self.projection = nn.Sequential(
            nn.Linear(in_features=inner_dim, out_features=emb_dim, bias=use_bias),
            nn.Dropout(p=dropout),
        )

        self.scale = (emb_dim**-0.5) if scale else 1.0
        if scale:
            self.register_buffer("scale_readout", torch.tensor(self.scale))

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        outputs = einsum(attn, v, "b h n i, b h i d -> b h n d")
        return outputs

    def mha(self, q: torch.Tensor, kv: torch.Tensor):
        q = self.layer_norm_q(q)
        kv = self.layer_norm_kv(kv)

        if self.use_pos_embedding:
            kv += self.positional_embedding

        q = self.to_q(q)  # [B, N_q, D]
        k, v = torch.chunk(self.to_kv(kv), chunks=2, dim=-1)  # [B, P, D], [B, P, D]

        q = self.rearrange(q)  # [B, N_q, D] -> [B, H, N_q, D_h]
        k = self.rearrange(k)  # [B, P, D] -> [B, H, P, D_h]
        v = self.rearrange(v)  # [B, P, D] -> [B, H, P, D_h]

        outputs = self.scaled_dot_product_attention(q=q, k=k, v=v)  # [B, H, N_q, D_h]
        outputs = rearrange(outputs, "b h n d -> b n (h d)")  # [B, N_q, D]
        return self.projection(outputs)  # project back to original embedding space

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        if self.grad_checkpointing:
            outputs = checkpoint(
                self.mha, q, kv, preserve_rng_state=True, use_reentrant=False
            )
        else:
            outputs = self.mha(q, kv)
        return outputs


class NeuronTokenizer(nn.Module):
    def __init__(
        self, 
        num_neurons: int, 
        emb_dim: int
        ):
        super(NeuronTokenizer, self).__init__()
        self.embedding = nn.Embedding(num_neurons, emb_dim)

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
        num_heads: int = 8,
        dropout: float = 0.0,
        use_lsa: bool = False,
        use_bias: bool = True,
        scale: bool = False,
        grad_checkpointing: bool = False,
        use_pos_embedding: bool = True,
        name: str = "AttentionReadout",
    ):
        super(AttentionReadout, self).__init__(
            args,
            input_shape=input_shape,
            output_shape=output_shape,
            ds=ds,
            name=name,
        )

        emb_dim = input_shape[0] #[C, H, W]
        num_patches = input_shape[-1]*input_shape[-2]  

        self.cross_attention = CrossAttention(
            input_shape=input_shape,
            num_neurons=self.num_neurons,
            emb_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_lsa=use_lsa,
            use_bias=use_bias,
            grad_checkpointing=grad_checkpointing,
            scale=scale,
            use_pos_embedding=use_pos_embedding
        )

        self.dropout = nn.Dropout(p=dropout)
        # print("emb_dim  ", emb_dim)
        self.neuron_tokenizer = NeuronTokenizer(num_neurons=self.num_neurons, emb_dim=emb_dim)
        self.neuron_projection = nn.Linear(in_features=emb_dim, out_features=1, bias=True)

    def forward(self, inputs: torch.Tensor, neuron_ids: torch.Tensor = None, query_neuron_subset: bool = False, shifts: torch.Tensor = None): 
        batch_size = inputs.size(0)
        if not query_neuron_subset:
            neuron_ids = torch.arange(self.num_neurons, device=inputs.device).unsqueeze(0)
            neuron_ids = neuron_ids.expand(batch_size, -1)  # [B, N_neurons]
        else:
            neuron_ids = torch.tensor(neuron_ids, device=inputs.device)
            neuron_ids = neuron_ids.unsqueeze(0).expand(batch_size, -1)
            assert neuron_ids is not None, "Neuron IDs must be provided if querying a subset of neurons."

        neuron_queries = self.neuron_tokenizer(neuron_ids)  # [B, N_neurons, D]
        neuron_queries = self.dropout(neuron_queries)
        # print("q  ", neuron_queries.shape)
        kv = rearrange(inputs, 'b c h w -> b (h w) c') 
        # print("kv  ", kv.shape)
        outputs = self.cross_attention(q=neuron_queries, kv=kv)  # [B, N_neurons, D]
        # print("outputs CA  ", outputs.shape)
        outputs = self.dropout(outputs)
        outputs = self.neuron_projection(outputs).squeeze(-1)  # [B, N_neurons]
        # print("outputs  ", outputs.shape)

        # outputs = []

        # # Attend to each neuron individually
        # for i in range(self.num_neurons):
        #     # Select embedding for the i-th neuron
        #     single_neuron_query = neuron_queries[:, i, :].unsqueeze(1)  # [B, 1, D]

        #     # Compute attention for the single neuron
        #     single_neuron_output = self.cross_attention(q=single_neuron_query, kv=inputs)  # [B, 1, D]

        #     # Apply projection to map to a scalar value for the neuron
        #     single_neuron_output = self.neuron_projection(single_neuron_output).squeeze(1)  # [B, 1] -> [B]

        #     # Append to the results
        #     outputs.append(single_neuron_output)

        # # Concatenate results for all neurons
        # outputs = torch.stack(outputs, dim=1) 

        return outputs


    def feature_l1(self, reduction: str = "sum"):
        l1 = self.neuron_tokenizer.embedding.weight.abs()
        if reduction == "sum":
            l1 = l1.sum()
        elif reduction == "mean":
            l1 = l1.mean()
        return l1

    def regularizer(self, reduction: str = "sum"):
        reg_term = self.reg_scale * self.feature_l1(reduction=reduction)
        
        # l1 regularization for neuron projection weights
        l1 = self.neuron_projection.weight.abs()
        if reduction == "sum":
            l1 = l1.sum()
        elif reduction == "mean":
            l1 = l1.mean()
        if self.neuron_projection.bias is not None:
            if reduction == "sum":
                l1 += self.neuron_projection.bias.abs().sum()
            elif reduction == "mean":
                l1 += self.neuron_projection.bias.abs().mean()
        reg_term += self.reg_scale * l1  

        return reg_term


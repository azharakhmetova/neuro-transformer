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

        emb_dim = 80  # embedding dimension for readout

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


# class CrossAttention(nn.Module):
#     def __init__(
#         self,
#         input_shape: tuple,
#         num_neurons: int, # outdims
#         emb_dim: int,
#         num_heads: int = 8,
#         dropout: float = 0.0,
#         use_lsa: bool = False,
#         use_bias: bool = True,
#         grad_checkpointing: bool = True,
#         scale: bool = False,
#         key_embedding: bool = False,
#         value_embedding: bool = False,
#         use_layer_norm: bool = False,
#         use_pos_embedding: bool = True,
#         temperature: tuple = (False, 1.0) # (learnable-per-neuron, value)
#     ):
#         super(CrossAttention, self).__init__()
#         self.grad_checkpointing = grad_checkpointing
#         self.use_pos_embedding = use_pos_embedding
#         self.key_embedding = key_embedding
#         self.value_embedding = value_embedding
#         print("key, value emb ", key_embedding, value_embedding)
#         self.use_bias = use_bias

#         self.input_shape = input_shape
#         self.num_neurons = num_neurons
#         print("num_neurons  ", num_neurons)
#         self.emb_dim = emb_dim
#         self.heads = num_heads
#         self.use_pos_embedding = use_pos_embedding

#         if scale:
#             dim_head = self.emb_dim // self.heads
#             self.scale = dim_head ** -0.5  # prevent softmax gradients from vanishing (for large dim_head)
#         else:
#             self.scale = 1.0
#         if temperature[0]:
#             self.T = temperature[1]
#         else:
#             self.T = nn.Parameter(torch.ones(self.num_neurons) * temperature[1])
#         # if use_layer_norm:
#         self.layer_norm = nn.LayerNorm(self.emb_dim)  
#         self.layer_norm_inputs = nn.LayerNorm(self.input_shape[0])

#         print("here")
#         # else:
#         #     self.layer_norm = None

#         # inner_dim = emb_dim * num_heads

#         # self.layer_norm_kv = nn.LayerNorm((input_shape[1]*input_shape[2], emb_dim)) 

#         # self.to_q = nn.Linear(
#         #     in_features=self.emb_dim, out_features=dim_head, bias=False
#         # )  

#         if self.key_embedding and self.value_embedding:
#             self.to_kv = nn.Linear(in_features=self.input_shape[0], out_features=self.emb_dim * 2, bias=False) 
#             print("here 1")
#         elif self.key_embedding:
#             self.to_key = nn.Linear(in_features=self.input_shape[0], out_features=self.emb_dim, bias=False)
#         # self.to_kv = nn.Linear(in_features=self.input_shape[3], out_features=self.emb_dim * 2, bias=False) 
        
#         self.positional_embedding = nn.Parameter(
#             torch.randn(1, input_shape[1]*input_shape[2], input_shape[0])
#         ) if use_pos_embedding else None
        
#         self.heads_rearrange = Rearrange("b n (h d) -> b h n d", h=num_heads)
#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(p=dropout)

#         self.projection = nn.Sequential(
#             nn.Linear(in_features=dim_head, out_features=emb_dim, bias=use_bias),
#             nn.Dropout(p=dropout),
#         )

#         if scale:
#             self.register_buffer("scale_readout", torch.tensor(self.scale))

#     def scaled_dot_product_attention(
#         self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
#     ):
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale #/ self.T #.view(1, 1, 1, -1)
#         print("dots  ", dots.size())
#         attn = self.attend(dots)
#         print("attn  ", attn.size())
#         attn = self.dropout(attn)
#         outputs = einsum(attn, v, "b h n i, b h i d -> b h n d")
#         print("outputs", outputs.size())
#         return outputs

#     def mha(self, q: torch.Tensor, inputs: torch.Tensor):
#         print("q in mha", q.size())
#         # if self.layer_norm:
#         q = self.layer_norm(q)
#         print("did layernowm")
#         inputs = self.layer_norm_inputs(inputs)
#         print("did layernowm input")
#         if self.use_pos_embedding:
#             print("use_pos_embedding", self.use_pos_embedding)
#             inputs_pos = inputs + self.positional_embedding
#             print("inputs_pos  ", inputs_pos.size())
#         else:
#             inputs_pos = inputs 

#         b = inputs.size(0)
#         print("b", b)

#         # q = self.to_q(q)  # [B, N_q, D]
#         # k, v = torch.chunk(self.to_kv(kv), chunks=2, dim=-1)  # [B, P, D], [B, P, D]
#         # if self.key_embedding and self.value_embedding:
#         #     key, value = self.to_kv(rearrange(inputs, "b s c-> (b s) c")).chunk(2, dim=-1) # → [(b*s), 2c] → [(b*s), c], [(b*s), c]
#         #     print("key  ", key.size())
#         #     k = rearrange(key, "(b s) (h d) -> b h s d", h=self.heads, b=b)
#         #     print("k", k.size())
#         #     v = rearrange(value, "(b s) (h d) -> b h s d", h=self.heads, b=b)
#         #     print("v", v.size())

#         # elif self.key_embedding:
#         #     key = self.to_key(rearrange(inputs_pos, "b s c -> (b s) c"))
#         #     k = rearrange(key, "(b s) (h d) -> b h s d", h=self.heads, b=b)
#         #     v = rearrange(inputs, "b s (h d) -> b h s d", h=self.heads)

#         # else:
#         #     k = rearrange(inputs_pos, "b s (h d) -> b h s d", h=self.heads)
#         #     v = rearrange(inputs, "b s (h d) -> b h s d", h=self.heads)

#         # q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
#         # # q = self.heads_rearrange(q)  # [B, N_q, D] -> [B, H, N_q, D_h]
#         # print("q  ", q.size())
#         # # # k = self.heads_rearrange(k)  # [B, P, D] -> [B, H, P, D_h]
#         # # # v = self.heads_rearrange(v)  # [B, P, D] -> [B, H, P, D_h]

#         # outputs = self.scaled_dot_product_attention(q=q, k=k, v=v)  # [B, H, N_q, D_h]
#         # print("outputs of dot product attention  ", outputs.size())
#         # outputs = rearrange(outputs, "b h n d -> b n (h d)")  # [B, N_q, D]
#         # return outputs #self.projection(outputs)  # project back to original embedding space

# # #version 2
#         if self.key_embedding and self.value_embedding:
#             key, value = self.to_kv(rearrange(inputs, "b s c-> (b s) c")).chunk(2, dim=-1) # → [(b*s), 2c] → [(b*s), c], [(b*s), c]
#             print("key  ", key.size())
#             k = rearrange(key, "(b s) (h d) -> b h s d", h=self.heads, b=b)
#             print("k", k.size())
#             v = rearrange(value, "(b s) (h d) -> b h s d", h=self.heads, b=b)
#             print("v", v.size())

#         elif self.key_embedding:
#             key = self.to_key(rearrange(inputs_pos, "b s c -> (b s) c"))
#             k = rearrange(key, "(b s) (h d) -> b h s d", h=self.heads, b=b)
#             v = rearrange(inputs, "b s (h d) -> b h s d", h=self.heads)

#         else:
#             k = rearrange(inputs_pos, "b s (h d) -> b h s d", h=self.heads)
#             v = rearrange(inputs, "b s (h d) -> b h s d", h=self.heads)
        
#         q = rearrange(q, "b n (h d) -> b h n", h=self.heads)
#         print("q  ", q.size())
#         # compare neuron query with each spatial position (dot-product)
#         dot = torch.einsum("ihds,ohdn->ihsn", k, q)  # -> [Images, Heads, w*h, Neurons]
#         print("dot  ", dot.size())
#         dot = dot * self.scale / self.T
#         print("dot  ", dot.size())
#         # compute attention weights
#         attention_weights = torch.nn.functional.softmax(dot, dim=2)  # -> [Images, Heads, w*h, Neurons]
#         print("attention_weights  ", attention_weights.size())
#         # compute average weighted with attention weights
#         # y = torch.einsum("ihds,ihsn->ihdn", value, attention_weights)  # -> [Images, Heads, Head_Dim, Neurons]
#         # y = torch.einsum("ihsn,ihsd->ihnd", attention_weights, value)
#         # y = rearrange(y, "i h n d -> i n (h d)")  # -> [Images, Neurons, Channels]
#         y = einsum(attention_weights, v, "b h n i, b h i d -> b h n d")
#         print("y  ", y.size())

#         # y = rearrange(y, "i h d n -> i n (h d)")  # -> [Images, Neurons, Channels]

#         # feat = self.features.view(1, c, self.outdims)
#         # y = torch.einsum("icn,ocn->in", y, feat)  # -> [Images, Neurons]

#         # if self.use_bias is not None:
#         #     y = y + self.bias

#         # if output_attn_weights:
#         #     return y, attention_weights
#         return y





#     def forward(self, q: torch.Tensor, inputs: torch.Tensor):
#         if self.grad_checkpointing:
#             print("in grad_checkpointing")
#             outputs = checkpoint(
#                 self.mha, q, inputs, preserve_rng_state=True, use_reentrant=False
#             )
#             print("outputs forward ", outputs.size())
#         else:
#             print("here forward")
#             outputs = self.mha(q, inputs)
#         return outputs


# class NeuronTokenizer(nn.Module):
#     def __init__(
#         self, 
#         num_neurons: int, 
#         emb_dim: int
#         ):
#         super(NeuronTokenizer, self).__init__()
#         self.embedding = nn.Embedding(num_neurons, emb_dim)

#     def forward(self, neuron_ids: torch.Tensor):
#         return self.embedding(neuron_ids)


# @register("attention")
# class AttentionReadout(Readout):
#     def __init__(
#         self,
#         args,
#         input_shape: tuple,
#         output_shape: tuple,
#         ds: DataLoader,
#         num_heads: int = 2,
#         dropout: float = 0.0,
#         use_lsa: bool = False,
#         use_bias: bool = True,
#         scale: bool = True,
#         key_embedding: bool = True,
#         value_embedding: bool = True,
#         grad_checkpointing: bool = False,
#         use_layer_norm: bool = True,
#         use_pos_embedding: bool = True,
#         temperature: tuple = (False, 1.0),
#         name: str = "AttentionReadout",
#     ):
#         super(AttentionReadout, self).__init__(
#             args,
#             input_shape=input_shape,
#             output_shape=output_shape,
#             ds=ds,
#             name=name,
#         )

#         print("input_shape  ", input_shape)
#         emb_dim = 160 # for readout != C 
#         # num_patches = input_shape[-1]*input_shape[-2]  

#         self.cross_attention = CrossAttention(
#             input_shape=input_shape,
#             num_neurons=self.num_neurons,
#             emb_dim=emb_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             use_lsa=use_lsa,
#             use_bias=use_bias,
#             grad_checkpointing=grad_checkpointing,
#             key_embedding=key_embedding,
#             value_embedding=value_embedding,
#             scale=scale,
#             temperature=temperature,
#             use_pos_embedding=use_pos_embedding
#         )

#         self.dropout = nn.Dropout(p=dropout)
#         # print("emb_dim  ", emb_dim)
#         self.neuron_tokenizer = NeuronTokenizer(num_neurons=self.num_neurons, emb_dim=emb_dim)
#         self.neuron_projection = nn.Linear(in_features=emb_dim, out_features=1, bias=True)

#     def forward(self, inputs: torch.Tensor, neuron_ids: torch.Tensor = None, query_neuron_subset: bool = False, shifts: torch.Tensor = None): 
#         b, c, w, h = inputs.size()
#         print("inputs  ", inputs.size())
#         c_in, w_in, h_in = self.input_shape
#         if (c_in, w_in, h_in) != (c, w, h):
#             warnings.warn("the specified feature map dimension is not the readout's expected input dimension")
        
    
#         if not query_neuron_subset:
#             neuron_ids = torch.arange(self.num_neurons, device=inputs.device).unsqueeze(0)
#             neuron_ids = neuron_ids.expand(b, -1)  # [B, N_neurons]
#         else:
#             neuron_ids = torch.tensor(neuron_ids, device=inputs.device)
#             neuron_ids = neuron_ids.unsqueeze(0).expand(b, -1)
#             assert neuron_ids is not None, "Neuron IDs must be provided if querying a subset of neurons."

#         neuron_queries = self.neuron_tokenizer(neuron_ids)  # [B, N_neurons, D]
#         neuron_queries = self.dropout(neuron_queries)
#         # print("q  ", neuron_queries.shape)
#         inputs = rearrange(inputs, 'b c h w -> b (h w) c') 
        
#         # print("kv  ", kv.shape)
#         outputs = self.cross_attention(q=neuron_queries, inputs=inputs)  # [B, N_neurons, D]
#         # print("outputs CA  ", outputs.shape)
#         outputs = self.dropout(outputs)
#         outputs = self.neuron_projection(outputs).squeeze(-1)  # [B, N_neurons]
#         # print("outputs  ", outputs.shape)

#         # outputs = []

#         # # Attend to each neuron individually
#         # for i in range(self.num_neurons):
#         #     # Select embedding for the i-th neuron
#         #     single_neuron_query = neuron_queries[:, i, :].unsqueeze(1)  # [B, 1, D]

#         #     # Compute attention for the single neuron
#         #     single_neuron_output = self.cross_attention(q=single_neuron_query, kv=inputs)  # [B, 1, D]

#         #     # Apply projection to map to a scalar value for the neuron
#         #     single_neuron_output = self.neuron_projection(single_neuron_output).squeeze(1)  # [B, 1] -> [B]

#         #     # Append to the results
#         #     outputs.append(single_neuron_output)

#         # # Concatenate results for all neurons
#         # outputs = torch.stack(outputs, dim=1) 

#         return outputs


#     def feature_l1(self, reduction: str = "sum"):
#         l1 = self.neuron_tokenizer.embedding.weight.abs()
#         if reduction == "sum":
#             l1 = l1.sum()
#         elif reduction == "mean":
#             l1 = l1.mean()
#         return l1

#     def regularizer(self, reduction: str = "sum"):
#         reg_term = self.reg_scale * self.feature_l1(reduction=reduction)
        
#         # l1 regularization for neuron projection weights
#         l1 = self.neuron_projection.weight.abs()
#         if reduction == "sum":
#             l1 = l1.sum()
#         elif reduction == "mean":
#             l1 = l1.mean()
#         if self.neuron_projection.bias is not None:
#             if reduction == "sum":
#                 l1 += self.neuron_projection.bias.abs().sum()
#             elif reduction == "mean":
#                 l1 += self.neuron_projection.bias.abs().mean()
#         reg_term += self.reg_scale * l1  

#         return reg_term



#-------------------------------------------------------------------------------------

# class CrossAttention(nn.Module):
#     def __init__(
#         self,
#         input_shape: tuple,
#         num_neurons: int,
#         emb_dim: int,
#         num_heads: int = 8,
#         dropout: float = 0.0,
#         use_lsa: bool = False,
#         use_bias: bool = True,
#         grad_checkpointing: bool = True,
#         scale: bool = False,
#         use_pos_embedding: bool = True
#     ):
#         super(CrossAttention, self).__init__()
#         self.grad_checkpointing = grad_checkpointing
#         self.use_pos_embedding = use_pos_embedding

#         inner_dim = emb_dim * num_heads

#         self.layer_norm_q = nn.LayerNorm((num_neurons, emb_dim))  
#         self.layer_norm_kv = nn.LayerNorm((input_shape[1]*input_shape[2], emb_dim)) 

#         self.to_q = nn.Linear(
#             in_features=emb_dim, out_features=inner_dim, bias=False
#         )  
#         self.to_kv = nn.Linear(
#                 in_features=emb_dim, out_features=inner_dim * 2, bias=False
#             ) 
        
#         self.positional_embedding = nn.Parameter(
#             torch.randn(1, input_shape[1]*input_shape[2], emb_dim)
#         ) if use_pos_embedding else None
        
#         self.rearrange = Rearrange("b n (h d) -> b h n d", h=num_heads)
#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(p=dropout)

#         self.projection = nn.Sequential(
#             nn.Linear(in_features=inner_dim, out_features=emb_dim, bias=use_bias),
#             nn.Dropout(p=dropout),
#         )

#         self.scale = (emb_dim**-0.5) if scale else 1.0
#         if scale:
#             self.register_buffer("scale_readout", torch.tensor(self.scale))

#     def scaled_dot_product_attention(
#         self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
#     ):
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         attn = self.attend(dots)
#         attn = self.dropout(attn)
#         print("attention_weights  ", attn.size())
#         outputs = einsum(attn, v, "b h n i, b h i d -> b h n d")
#         return outputs

#     def mha(self, q: torch.Tensor, kv: torch.Tensor):
#         q = self.layer_norm_q(q)
#         kv = self.layer_norm_kv(kv)

#         if self.use_pos_embedding:
#             kv += self.positional_embedding

#         q = self.to_q(q)  # [B, N_q, D]
#         k, v = torch.chunk(self.to_kv(kv), chunks=2, dim=-1)  # [B, P, D], [B, P, D]

#         q = self.rearrange(q)  # [B, N_q, D] -> [B, H, N_q, D_h]
#         print("q", q.size())
#         k = self.rearrange(k)  # [B, P, D] -> [B, H, P, D_h]
#         print("k", k.size())
#         v = self.rearrange(v)  # [B, P, D] -> [B, H, P, D_h]
#         print("v", v.size())

#         outputs = self.scaled_dot_product_attention(q=q, k=k, v=v)  # [B, H, N_q, D_h]
#         print("outputs", outputs.size())
#         outputs = rearrange(outputs, "b h n d -> b n (h d)")  # [B, N_q, D]
#         print("outputs", outputs.size())
#         return self.projection(outputs)  # project back to original embedding space

#     def forward(self, q: torch.Tensor, kv: torch.Tensor):
#         if self.grad_checkpointing:
#             outputs = checkpoint(
#                 self.mha, q, kv, preserve_rng_state=True, use_reentrant=False
#             )
#         else:
#             outputs = self.mha(q, kv)
#         return outputs


# class NeuronTokenizer(nn.Module):
#     def __init__(
#         self, 
#         num_neurons: int, 
#         emb_dim: int
#         ):
#         super(NeuronTokenizer, self).__init__()
#         self.embedding = nn.Embedding(num_neurons, emb_dim)

#     def forward(self, neuron_ids: torch.Tensor):
#         return self.embedding(neuron_ids)


# @register("attention")
# class AttentionReadout(Readout):
#     def __init__(
#         self,
#         args,
#         input_shape: tuple,
#         output_shape: tuple,
#         ds: DataLoader,
#         num_heads: int = 8,
#         dropout: float = 0.0,
#         use_lsa: bool = False,
#         use_bias: bool = True,
#         scale: bool = False,
#         grad_checkpointing: bool = False,
#         use_pos_embedding: bool = True,
#         name: str = "AttentionReadout",
#     ):
#         super(AttentionReadout, self).__init__(
#             args,
#             input_shape=input_shape,
#             output_shape=output_shape,
#             ds=ds,
#             name=name,
#         )

#         emb_dim = input_shape[0] #[C, H, W]
#         num_patches = input_shape[-1]*input_shape[-2]  

#         self.cross_attention = CrossAttention(
#             input_shape=input_shape,
#             num_neurons=self.num_neurons,
#             emb_dim=emb_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             use_lsa=use_lsa,
#             use_bias=use_bias,
#             grad_checkpointing=grad_checkpointing,
#             scale=scale,
#             use_pos_embedding=use_pos_embedding
#         )

#         self.dropout = nn.Dropout(p=dropout)
#         # print("emb_dim  ", emb_dim)
#         self.neuron_tokenizer = NeuronTokenizer(num_neurons=self.num_neurons, emb_dim=emb_dim)
#         self.neuron_projection = nn.Linear(in_features=emb_dim, out_features=1, bias=True)

#     def forward(self, inputs: torch.Tensor, neuron_ids: torch.Tensor = None, query_neuron_subset: bool = False, shifts: torch.Tensor = None): 
#         batch_size = inputs.size(0)
#         if not query_neuron_subset:
#             neuron_ids = torch.arange(self.num_neurons, device=inputs.device).unsqueeze(0)
#             neuron_ids = neuron_ids.expand(batch_size, -1)  # [B, N_neurons]
#         else:
#             neuron_ids = torch.tensor(neuron_ids, device=inputs.device)
#             neuron_ids = neuron_ids.unsqueeze(0).expand(batch_size, -1)
#             assert neuron_ids is not None, "Neuron IDs must be provided if querying a subset of neurons."

#         neuron_queries = self.neuron_tokenizer(neuron_ids)  # [B, N_neurons, D]
#         neuron_queries = self.dropout(neuron_queries)
#         # print("q  ", neuron_queries.shape)
#         kv = rearrange(inputs, 'b c h w -> b (h w) c') 
#         # print("kv  ", kv.shape)
#         outputs = self.cross_attention(q=neuron_queries, kv=kv)  # [B, N_neurons, D]
#         # print("outputs CA  ", outputs.shape)
#         outputs = self.dropout(outputs)
#         outputs = self.neuron_projection(outputs).squeeze(-1)  # [B, N_neurons]
#         # print("outputs  ", outputs.shape)

#         # outputs = []

#         # # Attend to each neuron individually
#         # for i in range(self.num_neurons):
#         #     # Select embedding for the i-th neuron
#         #     single_neuron_query = neuron_queries[:, i, :].unsqueeze(1)  # [B, 1, D]

#         #     # Compute attention for the single neuron
#         #     single_neuron_output = self.cross_attention(q=single_neuron_query, kv=inputs)  # [B, 1, D]

#         #     # Apply projection to map to a scalar value for the neuron
#         #     single_neuron_output = self.neuron_projection(single_neuron_output).squeeze(1)  # [B, 1] -> [B]

#         #     # Append to the results
#         #     outputs.append(single_neuron_output)

#         # # Concatenate results for all neurons
#         # outputs = torch.stack(outputs, dim=1) 

#         return outputs


#     def feature_l1(self, reduction: str = "sum"):
#         l1 = self.neuron_tokenizer.embedding.weight.abs()
#         if reduction == "sum":
#             l1 = l1.sum()
#         elif reduction == "mean":
#             l1 = l1.mean()
#         return l1

#     def regularizer(self, reduction: str = "sum"):
#         reg_term = self.reg_scale * self.feature_l1(reduction=reduction)
        
#         # l1 regularization for neuron projection weights
#         l1 = self.neuron_projection.weight.abs()
#         if reduction == "sum":
#             l1 = l1.sum()
#         elif reduction == "mean":
#             l1 = l1.mean()
#         if self.neuron_projection.bias is not None:
#             if reduction == "sum":
#                 l1 += self.neuron_projection.bias.abs().sum()
#             elif reduction == "mean":
#                 l1 += self.neuron_projection.bias.abs().mean()
#         reg_term += self.reg_scale * l1  

#         return reg_term


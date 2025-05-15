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


class NeuronIDTokenizer(nn.Module):
    # todo
    # add dictionary module for different latent states
    # self.neuron_id_tokenizer = nn.ModuleDict({})
    # for key, n_neurons in num_neurons.items():
    #     self.neuron_id_tokenizer[key] = nn.Embedding(n_neurons, token_dim, device=device)
    def __init__(
        self, 
        num_neurons: int,
        emb_dim: int
        ):
        super(NeuronIDTokenizer, self).__init__()
        self.embedding = nn.Embedding(num_neurons, emb_dim)
        nn.init.constant_(self.embedding.weight, 1.0 / emb_dim)

    def forward(self, neuron_ids: torch.Tensor):
        return self.embedding(neuron_ids)
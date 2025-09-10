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
    

class SimpleResponsesTokenizer(nn.Module):
    # todo 
    # - do we want neuron id tokens independent of the responses?
    # - masking?
    # - add back embedding per latent state:
    # self.session_tokenizer = nn.Embedding(len(num_neurons), token_dim, device=device)
    # self.sessions_enc = {}
    # for i, s in enumerate(num_neurons.keys()):
    #     self.sessions_enc[s] = torch.Tensor([i]).long().to(device)

    def __init__(
        self, 
        args,
        num_output_neurons: t.Dict[str, tuple],
        num_samples_per_neuron: int,
        num_samples_per_token: int, 
        frac_input_neurons: float,
        emb_dim: int, 
        device: torch.device,
        use_masking: bool = False,
    ):
        super().__init__()
        self.use_masking = use_masking
        self.device = device
        self.samples_per_token = num_samples_per_token
        self.tokenizer = nn.Linear(num_samples_per_token, emb_dim)
        self.T = self.num_tokens_per_neuron(num_samples_per_neuron)

        self.output_shapes = {}
        self.num_input_neurons = {}
        for session, n_neurons in num_output_neurons.items():
            self.num_input_neurons[session] = int(frac_input_neurons * n_neurons[0])
            self.num_input_tokens = self.num_input_neurons[session] * self.T
            self.output_shapes[session] = (self.num_input_tokens, emb_dim) 

        self.project_neuron_id = emb_dim != args.emb_dim_n_id
        if self.project_neuron_id:
            self.neuron_id_token_projection = nn.Linear(args.emb_dim_n_id, emb_dim)

    def num_tokens_per_neuron(self, num_samples_per_neuron):
        return num_samples_per_neuron // self.samples_per_token

    def forward(
            self, 
            responses: torch.Tensor, 
            mouse_id: str=None,
            neuron_id_tokens: t.Dict[str, torch.Tensor] = None,
            mask: torch.Tensor = None,
            mask_token = None
    ):
        # todo - this needs validation
        (B, N, S) = responses.shape
        # print("S", S)
        # print("Input device:", responses.device)
        # responses = responses.to(input_neuron_ids.device)
        # print('input neuron id device', input_neuron_ids.device)
        # print("T", self.T)
        # print("responses shape (B, N, S)", responses.shape)
        # responses_subset = responses[:, input_neuron_ids, :] #.float().to(input_neuron_ids.device) # select only the neurons we are interested in
        # print("responses subset shape (B, N, S)", responses_subset.shape)
        # print("Tokenizer weight device:", self.tokenizer.weight.device)

        tok = self.tokenizer(
                responses.view(B, self.num_input_neurons[mouse_id], self.T, self.samples_per_token)
            ) # (B, N, T, samples_per_T) -> (B, N, T, emb_dim_input_neurons)
        if self.use_masking:
            tok = torch.where(mask.view(B, self.num_input_neurons[mouse_id], self.T, 1), tok, mask_token)

        if self.project_neuron_id:
            neuron_id_tok = self.neuron_id_token_projection(neuron_id_tokens) # (N, emb_dim_input_neurons)
            # print("neuron id tokens shape (B, N, emb)", neuron_id_tok.shape)
        else:
            neuron_id_tok = neuron_id_tokens 
        
        # (N, emb_dim_input_neurons) -> (B, N, T, emb_dim_input_neurons)
        tok = tok + neuron_id_tok.unsqueeze(0).unsqueeze(2).repeat(B, 1, self.T, 1)

        tok = tok.view(tok.shape[0], -1, tok.shape[-1]) # (B, N, T, emb_dim_input_neurons) -> (B, N*T, emb_dim_input_neurons)
        # print("final token shape (B, N, emb)", tok.shape)
        return tok
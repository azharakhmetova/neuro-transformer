import os
import sys
import math
import torch
import numpy as np
import typing as t
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

from v1t.models import Model
from v1t.utils.decoding_latents_pytorch import LinearRegression

import torch

def compute_batch_jacobian(z_hat, responses_t, input_neuron_ids) -> torch.Tensor:
    """
    z_hat: [B, K]
    r_in:  [B, Kin, T] or [B, Kin]
    returns A: [B, K, Kin, T] or [B, K, Kin]
    """
    B, K = z_hat.shape
    grads = []
    for k in range(K):
        g_full = torch.autograd.grad(
            z_hat[:, k].sum(),
            responses_t,                # <-- full tensor, not slice
            retain_graph=True,
            allow_unused=False,
        )[0]                             # [B,N,1]
        grads.append(g_full[:, input_neuron_ids, :])  # [B,Kin,1]
    return torch.stack(grads, dim=1).squeeze(-1)                # [B,K,Kin]


def sample_z_and_A_comp(
    ds: DataLoader,
    model: Model,
    probe: nn.Module,
    sel_frac: float,
    input_indices: torch.Tensor,
    query_indices: torch.Tensor,
    device="cuda",
    num_samples=None,
    store_A=False,
):
    model.to(device).eval()
    probe.to(device).eval()
    # for p in model.parameters():
    #     p.requires_grad_(False)
    # for p in probe.parameters():
    #     p.requires_grad_(False)
    mouse_id = ds.dataset.mouse_id
    P = model.core.num_image_patches

    z_all = []
    A_all = [] if store_A else None
    
    remaining = num_samples

    for batch in tqdm(ds, desc=f"Mouse {mouse_id}"): #iterate(ds, desc=f"Mouse {mouse_id}"):
        images = batch["image"].to(device)
        responses_t = batch["response"].to(device).detach().requires_grad_(True) 
        neuron_coords = batch["neuron_coordinates"].to(device)
        behaviors = batch["behavior"].to(device)
        pupil_centers = batch["pupil_center"].to(device)

        N = responses_t.size(1)
        input_neuron_ids = input_indices.to(device)
        query_neuron_ids = query_indices.to(device)

        if sel_frac == 0:
            input_neuron_ids = None
            query_neuron_ids = torch.arange(N, device=device)

        # ---- run your forward until cross-attn outputs ----
        images, _ = model.image_cropper(
            inputs=images, mouse_id=mouse_id,
            behaviors=behaviors, pupil_centers=pupil_centers
        )
        # --- image tokenization (+ optional image self-attn) ---
        image_tokens = model.patch_embedding(images)
        if getattr(model, "self_attend_image_tokens", False):
            image_tokens = model.image_tokens_attention(
                image_tokens, output_attn_weights=False
            )  # [B, Layers, Heads, P, P]

        # --- input neuron tokenization (+ optional input self-attn) ---
        input_neuron_tokens = None
        if getattr(model, "tokenize_neurons", False) and getattr(model, "frac_input_neurons", 0) > 0:
            # ensure responses has time dim
            if responses_t is not None and responses_t.dim() != 3:
                responses = responses_t.unsqueeze(-1)
            else:
                responses = responses_t

            id_tok = model.neuron_id_tokenizer[mouse_id](input_neuron_ids.long())
            id_tok = id_tok + model.session_tokenizer(model.sessions_enc[mouse_id].to(model.device))

            input_neuron_tokens = model.input_neuron_embedding(
                responses=responses[:, input_neuron_ids, :],
                mouse_id=mouse_id,
                neuron_id_tokens=id_tok,
            )

            # add neuron PEs if enabled 
            if getattr(model, "use_input_neuron_pe", False):
                print("adding neuron PEs")
                if model.neuron_pe_mode == "1d":
                    input_neuron_tokens += model.neuron_pe(responses)[:, input_neuron_ids, :]
                elif model.neuron_pe_mode == "coord":
                    input_coords = neuron_coords[:, input_neuron_ids, :]
                    input_neuron_tokens += model.neuron_coord_pe(input_coords)
                elif model.neuron_pe_mode == "both":
                    input_coords = neuron_coords[:, input_neuron_ids, :]
                    input_neuron_tokens += (model.neuron_pe(responses)[:, input_neuron_ids, :]
                                            + model.neuron_coord_pe(input_coords))

            if getattr(model, "self_attend_input_neurons", False):
                # print("self-attending input neuron tokens")
                input_neuron_tokens = model.input_neurons_attention(
                    input_neuron_tokens, output_attn_weights=False
                )  # [B, Layers, Heads, K, K]

        # --- core ---
        outputs = model.core(
            image_tokens=image_tokens,
            neuron_tokens=input_neuron_tokens,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
            output_attn_weights=False,
        )  # outputs: [B, S, C]; core_attn: [B, Layers, Heads, S, S] (S = P (+K))
        

        if model.subselect_image_tokens:
            # CHANGE LEARNABLE CASE and remove it from attention readout
            # add positional encoding after the core 
            if model.use_pe_after_core:
                if model.pe_after_core == "2d":
                    temp = outputs.reshape(outputs.shape[0], model.patch_embedding.height, model.patch_embedding.width, outputs.shape[-1])  # (B, h, w, num_channels)
                    temp += model.patch_embedding.pos_embedding(temp)
                    outputs = temp.reshape(outputs.shape[0], -1, outputs.shape[-1])  # (B, num_tokens, num_channels)
                elif model.pe_after_core == "1d":
                    outputs += model.patch_embedding.pos_embedding(outputs)

        if model.readout_type == "attention":
            query_neurons = model.neuron_id_tokenizer[mouse_id](neuron_ids=query_neuron_ids.to(torch.long).unsqueeze(0).expand(image_tokens.shape[0], -1)) # (B, N-K, emb_dim_n_id)
            query_neurons += model.session_tokenizer(model.sessions_enc[mouse_id].to(model.device))
            if model.use_query_neuron_pe:   
                if model.neuron_pe_mode == "1d":
                    if model.project_query_pe:
                        query_neurons += model.projection_query_pe(model.neuron_pe(responses)[:, query_neuron_ids, :])
                    else:
                        query_neurons += model.neuron_pe(responses)[:, query_neuron_ids, :]
                elif model.neuron_pe_mode == "coord":
                    query_coords = neuron_coords[:, query_neuron_ids, :]
                    if model.project_query_pe:
                        query_neurons += model.projection_query_pe(model.neuron_coord_pe(query_coords))
                    else:
                        query_neurons += model.neuron_coord_pe(query_coords)
                elif model.neuron_pe_mode == "both":
                    query_coords = neuron_coords[:, query_neuron_ids, :]
                    if model.project_query_pe:
                        query_neurons += (model.projection_query_pe(model.neuron_coord_pe(query_coords)) + model.projection_query_pe(model.neuron_pe(responses)[:, query_neuron_ids, :]))
                    else:
                        query_neurons += (model.neuron_coord_pe(query_coords) + model.neuron_pe(responses)[:, query_neuron_ids, :])
        readout = model.readouts[mouse_id]
        if readout.project_query_neurons: 
            query_neurons = readout.id_query_projection(query_neurons)
        outputs = readout.cross_attention(q=query_neurons, inputs=outputs)

# ------------- attach the linear head ------------------
        probe.to(device)
        z_hat = probe(outputs[:, P:, :])
        z_all.append(z_hat.detach().cpu())

        if store_A:
            # define r_in as the actual inputs you want derivative wrt:
            # if responses.dim() != 3:
            #     responses = responses.unsqueeze(-1)
            # else:
            #     responses = responses
            # r_in = responses[:, input_neuron_ids, :]  # [B, Kin, T]

            A = compute_batch_jacobian(z_hat, responses, input_neuron_ids)     # [B, K, Kin, T]
            A_all.append(A.detach().cpu())

        if remaining is not None:
            remaining -= images.size(0)
            if remaining <= 0:
                break

    z_all = torch.cat(z_all, dim=0)
    if store_A:
        A_all = torch.cat(A_all, dim=0)
        return z_all.numpy(), A_all.numpy() 
    return z_all.numpy() 
# # @torch.no_grad()
# def sample_z_post_comp(
#     ds: DataLoader,
#     model: Model,
#     probe: LinearRegression,
#     sel_frac: float,
#     num_samples: int = None,
#     device: torch.device = "cpu",
#     verbose: int = 1,
#     out_dir: str = None,
#     input_indices: t.Optional[torch.Tensor] = None,
#     query_indices: t.Optional[torch.Tensor] = None,
# ):
#     model.to(device).eval()
#     mouse_id = ds.dataset.mouse_id
#     batch_idx = 0

#     P = model.core.num_image_patches
#     i_T_img = ds.dataset.i_transform_image

#     remaining = num_samples

#     def iterate(ds, desc):
#         if sys.stdout.isatty():  # interactive
#             return tqdm(ds, desc=desc, dynamic_ncols=True, mininterval=2.0)
#         else:  # log file / non-TTY: manual periodic prints
#             total = len(ds)
#             every = max(1, total // 100)  # ~1% steps
#             for i, batch in enumerate(ds, 1):
#                 if i % every == 0 or i == 1 or i == total:
#                     print(f"{desc}: {i}/{total} [{i/total:6.2%}]", flush=True)
#                 yield batch
#     for batch in tqdm(ds, desc=f"Mouse {mouse_id}", disable=not verbose): #iterate(ds, desc=f"Mouse {mouse_id}"):
#         images = batch["image"].to(device)
#         # print("input images:", images.shape)
#         responses = batch["response"].to(device).detach().requires_grad_(True)#batch["response"].to(device)
#         N = responses.size(1)
#         neuron_coords = batch["neuron_coordinates"].to(device)
#         behaviors = batch["behavior"].to(device)
#         pupil_centers = batch["pupil_center"].to(device)
#         # ids are 1D tensors as they the same for the whole batch        
#         input_neuron_ids = input_indices.to(device)  
#         query_neuron_ids = query_indices.to(device) 

#         if sel_frac == 0:
#             input_neuron_ids = None
#             query_neuron_ids = torch.arange(N).to(device)

#         images, _ = model.image_cropper(
#             inputs=images, mouse_id=mouse_id,
#             behaviors=behaviors, pupil_centers=pupil_centers
#         )
#         # --- image tokenization (+ optional image self-attn) ---
#         image_tokens = model.patch_embedding(images)
#         if getattr(model, "self_attend_image_tokens", False):
#             image_tokens = model.image_tokens_attention(
#                 image_tokens, output_attn_weights=False
#             )  # [B, Layers, Heads, P, P]

#         # --- input neuron tokenization (+ optional input self-attn) ---
#         input_neuron_tokens = None
#         if getattr(model, "tokenize_neurons", False) and getattr(model, "frac_input_neurons", 0) > 0:
#             # ensure responses has time dim
#             if responses is not None and responses.dim() != 3:
#                 responses = responses.unsqueeze(-1)

#             id_tok = model.neuron_id_tokenizer[mouse_id](input_neuron_ids.long())
#             id_tok = id_tok + model.session_tokenizer(model.sessions_enc[mouse_id].to(model.device))

#             input_neuron_tokens = model.input_neuron_embedding(
#                 responses=responses[:, input_neuron_ids, :],
#                 mouse_id=mouse_id,
#                 neuron_id_tokens=id_tok,
#             )

#             # add neuron PEs if enabled 
#             if getattr(model, "use_input_neuron_pe", False):
#                 print("adding neuron PEs")
#                 if model.neuron_pe_mode == "1d":
#                     input_neuron_tokens += model.neuron_pe(responses)[:, input_neuron_ids, :]
#                 elif model.neuron_pe_mode == "coord":
#                     input_coords = neuron_coords[:, input_neuron_ids, :]
#                     input_neuron_tokens += model.neuron_coord_pe(input_coords)
#                 elif model.neuron_pe_mode == "both":
#                     input_coords = neuron_coords[:, input_neuron_ids, :]
#                     input_neuron_tokens += (model.neuron_pe(responses)[:, input_neuron_ids, :]
#                                             + model.neuron_coord_pe(input_coords))

#             if getattr(model, "self_attend_input_neurons", False):
#                 # print("self-attending input neuron tokens")
#                 input_neuron_tokens = model.input_neurons_attention(
#                     input_neuron_tokens, output_attn_weights=False
#                 )  # [B, Layers, Heads, K, K]

#         # --- core ---
#         outputs = model.core(
#             image_tokens=image_tokens,
#             neuron_tokens=input_neuron_tokens,
#             mouse_id=mouse_id,
#             behaviors=behaviors,
#             pupil_centers=pupil_centers,
#             output_attn_weights=False,
#         )  # outputs: [B, S, C]; core_attn: [B, Layers, Heads, S, S] (S = P (+K))


#         if model.subselect_image_tokens:
#             # CHANGE LEARNABLE CASE and remove it from attention readout
#             # add positional encoding after the core 
#             if model.use_pe_after_core:
#                 if model.pe_after_core == "2d":
#                     temp = outputs.reshape(outputs.shape[0], model.patch_embedding.height, model.patch_embedding.width, outputs.shape[-1])  # (B, h, w, num_channels)
#                     temp += model.patch_embedding.pos_embedding(temp)
#                     outputs = temp.reshape(outputs.shape[0], -1, outputs.shape[-1])  # (B, num_tokens, num_channels)
#                 elif model.pe_after_core == "1d":
#                     outputs += model.patch_embedding.pos_embedding(outputs)

#         if model.readout_type == "attention":
#             query_neurons = model.neuron_id_tokenizer[mouse_id](neuron_ids=query_neuron_ids.to(torch.long).unsqueeze(0).expand(image_tokens.shape[0], -1)) # (B, N-K, emb_dim_n_id)
#             query_neurons += model.session_tokenizer(model.sessions_enc[mouse_id].to(model.device))
#             if model.use_query_neuron_pe:   
#                 if model.neuron_pe_mode == "1d":
#                     if model.project_query_pe:
#                         query_neurons += model.projection_query_pe(model.neuron_pe(responses)[:, query_neuron_ids, :])
#                     else:
#                         query_neurons += model.neuron_pe(responses)[:, query_neuron_ids, :]
#                 elif model.neuron_pe_mode == "coord":
#                     query_coords = neuron_coords[:, query_neuron_ids, :]
#                     if model.project_query_pe:
#                         query_neurons += model.projection_query_pe(model.neuron_coord_pe(query_coords))
#                     else:
#                         query_neurons += model.neuron_coord_pe(query_coords)
#                 elif model.neuron_pe_mode == "both":
#                     query_coords = neuron_coords[:, query_neuron_ids, :]
#                     if model.project_query_pe:
#                         query_neurons += (model.projection_query_pe(model.neuron_coord_pe(query_coords)) + model.projection_query_pe(model.neuron_pe(responses)[:, query_neuron_ids, :]))
#                     else:
#                         query_neurons += (model.neuron_coord_pe(query_coords) + model.neuron_pe(responses)[:, query_neuron_ids, :])
#         readout = model.readouts[mouse_id]
#         if readout.project_query_neurons: 
#             query_neurons = readout.id_query_projection(query_neurons)
#         outputs = readout.cross_attention(q=query_neurons, inputs=outputs)

# # ------------- attach the linear head ------------------
#         probe.to(device)
#         z_hat = probe(outputs)
        
#         # --- early stopping ---
#         if num_samples is not None:
#             # print(remaining)
#             remaining -= images.size(0)
#             if remaining <= 0:
#                 break

#     print("complete")
#     return z_hat

        
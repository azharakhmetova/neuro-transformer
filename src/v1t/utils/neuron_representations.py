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

@torch.no_grad()
def sample_neuron_tokens(
    ds: DataLoader,
    model: Model,
    num_samples: int = None,
    device: torch.device = "cpu",
    verbose: int = 1,
    out_dir: str = None,
    input_indices: t.Optional[torch.Tensor] = None,
    query_indices: t.Optional[torch.Tensor] = None,
    repr_type: str = "initial",
):
    model.to(device).eval()
    mouse_id = ds.dataset.mouse_id
    batch_idx = 0

    P = model.core.num_image_patches
    i_T_img = ds.dataset.i_transform_image

    # out = {
    #     "images": [],
    #     # "input_neuron_ids": [],
    #     # "query_neuron_ids": [],
    #     "input_neuron_tokens_initial": [],
    #     # "input_neuron_tokens_after_sa": [],
    #     # "input_neuron_tokens_after_core": [],
    # }

    remaining = num_samples

    def iterate(ds, desc):
        if sys.stdout.isatty():  # interactive
            return tqdm(ds, desc=desc, dynamic_ncols=True, mininterval=2.0)
        else:  # log file / non-TTY: manual periodic prints
            total = len(ds)
            every = max(1, total // 100)  # ~1% steps
            for i, batch in enumerate(ds, 1):
                if i % every == 0 or i == 1 or i == total:
                    print(f"{desc}: {i}/{total} [{i/total:6.2%}]", flush=True)
                yield batch

    for batch in tqdm(ds, desc=f"Mouse {mouse_id}", disable=not verbose): #iterate(ds, desc=f"Mouse {mouse_id}"):
        images = batch["image"].to(device)
        # print("input images:", images.shape)
        responses = batch["response"].to(device)
        N = responses.size(1)
        neuron_coords = batch["neuron_coordinates"].to(device)
        behaviors = batch["behavior"].to(device)
        pupil_centers = batch["pupil_center"].to(device)
        # ids are 1D tensors as they the same for the whole batch        
        input_neuron_ids = input_indices.to(device) if input_indices is not None else None #batch.get("input_neuron_ids", None)
        if repr_type == "initial" and input_neuron_ids is None:
            input_neuron_ids = torch.arange(N).to(device)
        if repr_type == "after_sa" and input_neuron_ids is None:
            input_neuron_ids = torch.arange(N).to(device)

        # if input_neuron_ids is not None:
        #     input_neuron_ids = input_neuron_ids.to(device)
            # print("input neuron ids:", input_neuron_ids)
            # sort_indices = torch.argsort(input_neuron_ids)
            # print("sort_indices:", sort_indices)
            # print("sorted input neuron ids:", input_neuron_ids[sort_indices])

        query_neuron_ids = query_indices.to(device) if query_indices is not None else batch.get("query_neuron_ids", None)
        if query_neuron_ids is not None:
            query_neuron_ids = query_neuron_ids.to(device)

        B = images.size(0)
        Nq = 0 if query_neuron_ids is None else query_neuron_ids.numel()
        K = 0 if input_neuron_ids is None else input_neuron_ids.numel()

        images, _ = model.image_cropper(
            inputs=images, mouse_id=mouse_id,
            behaviors=behaviors, pupil_centers=pupil_centers
        )
        image_shape = images.shape[2:]  # (H,W) for resizing heatmaps later

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
            if responses is not None and responses.dim() != 3:
                responses = responses.unsqueeze(-1)

            id_tok = model.neuron_id_tokenizer[mouse_id](input_neuron_ids.long())
            id_tok = id_tok + model.session_tokenizer(model.sessions_enc[mouse_id].to(model.device))

            input_neuron_tokens = model.input_neuron_embedding(
                responses=responses[:, input_neuron_ids, :],
                mouse_id=mouse_id,
                neuron_id_tokens=id_tok,
            )
            input_neuron_tokens_initial = input_neuron_tokens #[:, sort_indices, :]
            # print("input n tokens before SA:", input_neuron_tokens.shape)
            if repr_type == "initial":
                # print("saving initial input neuron tokens")
                np.save(os.path.join(out_dir, f"{batch_idx}.npy"), input_neuron_tokens_initial.detach().cpu().numpy().astype(np.float32))
                batch_idx += 1
                continue
            # print("input n tokens initial:", input_neuron_tokens_initial.shape)

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
            if repr_type == "after_sa":
                # print("saving initial input neuron tokens")
                np.save(os.path.join(out_dir, f"{batch_idx}.npy"), input_neuron_tokens.detach().cpu().numpy().astype(np.float32))
                batch_idx += 1
                # print("input neuron tokens after SA:", input_neuron_tokens.shape)
                continue

        #         # re-add neuron PEs 
        #         if getattr(model, "use_input_neuron_pe", False):
        #             if model.neuron_pe_mode == "1d":
        #                 input_neuron_tokens += model.neuron_pe(responses)[:, input_neuron_ids, :]
        #             elif model.neuron_pe_mode == "coord":
        #                 input_coords = neuron_coords[:, input_neuron_ids, :]
        #                 input_neuron_tokens += model.neuron_coord_pe(input_coords)
        #             elif model.neuron_pe_mode == "both":
        #                 input_coords = neuron_coords[:, input_neuron_ids, :]
        #                 input_neuron_tokens += (model.neuron_pe(responses)[:, input_neuron_ids, :]
        #                                         + model.neuron_coord_pe(input_coords))
                
        #         # print("input n tokens after sa:", input_neuron_tokens.shape)
        # print("saving input neuron tokens after SA")
        
        # --- core ---
        outputs = model.core(
            image_tokens=image_tokens,
            neuron_tokens=input_neuron_tokens,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
            output_attn_weights=False,
        )  # outputs: [B, S, C]; core_attn: [B, Layers, Heads, S, S] (S = P (+K))
        if repr_type == "after_core":
            print("core output tokens:", outputs[:, P:].shape)
            np.save(os.path.join(out_dir, f"{batch_idx}.npy"), outputs[:, P:].detach().cpu().numpy().astype(np.float32))
            batch_idx += 1
            continue

        if repr_type == "queries":
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
            print("query neuron tokens:", outputs.shape)
            np.save(os.path.join(out_dir, f"{batch_idx}.npy"), outputs.detach().cpu().numpy().astype(np.float32))
            batch_idx += 1
            
        # out["images"].append(i_T_img(images.cpu()))
        # # out["input_neuron_ids"].append(input_neuron_ids.cpu().unsqueeze(0) if input_neuron_ids is not None else torch.tensor([]))
        # # out["query_neuron_ids"].append(query_neuron_ids.cpu().unsqueeze(0) if query_neuron_ids is not None else torch.tensor([]))
        # if K > 0:
        #     out["input_neuron_tokens_initial"].append(input_neuron_tokens_initial.cpu())  
        # else:
        #     out["input_neuron_tokens_initial"].append(torch.tensor([]))

        # if getattr(model, "self_attend_input_neurons", False):    
        #     if K > 0:
        #         out["input_neuron_tokens_after_sa"].append(input_neuron_tokens_after_sa.cpu())  
        #     else:
        #         out["input_neuron_tokens_after_sa"].append(torch.tensor([]))
                
        # if K > 0:
        #     out["input_neuron_tokens_after_core"].append(outputs[:, P:][:, sort_indices, :].cpu())  
        # else:
        #     out["input_neuron_tokens_after_core"].append(torch.tensor([]))

                # --- early stopping ---
        if num_samples is not None:
            # print(remaining)
            remaining -= images.size(0)
            if remaining <= 0:
                break

    print("complete")
    # for key in out.keys():
    #     if key == 'input_neuron_ids' or key == 'query_neuron_ids':
    #         continue
    #     out[key] = torch.cat(out[key], dim=0)
    #     print(f"{key} shape:", out[key].shape)
    # return out

        
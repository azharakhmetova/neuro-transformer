import math
from tabnanny import verbose
import torch
import numpy as np
import typing as t
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

from v1t.models import Model

def find_shape(num_patches: int):
    dim1 = math.ceil(math.sqrt(num_patches))
    while num_patches % dim1 != 0 and dim1 > 0:
        dim1 -= 1
    dim2 = num_patches // dim1
    return dim1, dim2

def normalize_along(x, reduce_dims, keep_dims_shape, eps=1e-12):
    minv = x.amin(dim=reduce_dims, keepdim=True)
    maxv = x.amax(dim=reduce_dims, keepdim=True)
    return (x - minv) / (maxv - minv + eps)

def extract_legrad_attention_maps(
    ds: DataLoader,
    model: Model,
    device: torch.device,
    head_reduce: str = "mean",
    neuron_idx: int = None,   # specific query neuron to compute LeGrad for
    num_samples: int = None,  # number of trials to process
    verbose: bool = True,
):
    model.to(device).eval()
    mouse_id = ds.dataset.mouse_id
    i_T_img = ds.dataset.i_transform_image
    i_T_beh = ds.dataset.i_transform_behavior
    i_T_pup = ds.dataset.i_transform_pupil_center

    # fraction of neurons the model was trained with
    frac_input_neurons = getattr(model, "frac_input_neurons", 0.0)
    # dictionary to store outputs
    out = {
        "images": [],
        "input_neuron_ids": [],
        "query_neuron_ids": [],
        "behaviors": [],
        "pupil_centers": [],
        # core 
        # "core_image_legrad_maps": [],              # [B, L, H_img, W_img] rollout or diag-like
        # "core_neuron_to_image_legrad_maps": [],        # [B, L, K, H_img, W_img] 
        # "core_neuron_to_neuron_legrad_maps": [],       # [B, L, K, K]

        "core_query_to_image_legrad_maps": [],     # [B, Nq, H_img, W_img]
        "core_query_to_neuron_legrad_maps": [],     # [B, Nq, K]
        # readout
        "readout_query_to_image_legrad_maps": [],     # [B, Nq, H_img, W_img]
        "readout_query_to_neuron_legrad_maps": [],     # [B, Nq, K]
        # self-attn (optional)
        "image_self_attn_legrad_maps": [],             # [B, H_img, W_img] 
        "input_neuron_self_attn_legrad_maps": [],      # [B, K, K] or per-neuron heatmaps
    }
    remaining = num_samples

    for batch in tqdm(ds, desc=f"Mouse {mouse_id}", disable=not verbose):
        images = batch["image"].to(device)
        responses = batch["response"].to(device)
        neuron_coords = batch["neuron_coordinates"].to(device)
        behaviors = batch["behavior"].to(device)
        pupil_centers = batch["pupil_center"].to(device)

        N = responses.size(1)
        K = int(frac_input_neurons * N)
        # ids are 1D tensors as they the same for the whole batch
        input_neuron_ids = batch.get("input_neuron_ids", None)
        if input_neuron_ids is not None:
            input_neuron_ids = input_neuron_ids.to(device)

        query_neuron_ids = batch.get("query_neuron_ids", None)
        if query_neuron_ids is not None:
            query_neuron_ids = query_neuron_ids.to(device)

        B = images.size(0)
        K = 0 if input_neuron_ids is None else input_neuron_ids.numel()
        Nq = 0 if query_neuron_ids is None else query_neuron_ids.numel()

        # forward pass
        # --- image cropping ---
        images, _ = model.image_cropper(
            inputs=images, mouse_id=mouse_id,
            behaviors=behaviors, pupil_centers=pupil_centers,
        )
        image_shape = images.shape[2:]

        # --- image tokenization (+ optional image self-attn) ---
        image_tokens = model.patch_embedding(images)
        images_self_attn = None
        if getattr(model, "self_attend_image_tokens", False):
            image_tokens, images_self_attn = model.image_tokens_attention(
                image_tokens, output_attn_weights=True
            )  # [B, Layers, Heads, P, P]

        # --- input neuron tokenization (+ optional input self-attn) ---
        input_neuron_tokens = None
        if getattr(model, "tokenize_neurons", False) and getattr(model, "frac_input_neurons", 0) > 0:
            if responses.dim() != 3:
                responses = responses.unsqueeze(-1)
            # build ID tokens 
            id_tok = model.neuron_id_tokenizer[mouse_id](input_neuron_ids.long())
            id_tok = id_tok + model.session_tokenizer(model.sessions_enc[mouse_id].to(model.device))
            
            input_neuron_tokens = model.input_neuron_embedding(
                responses=responses[:, input_neuron_ids, :],
                mouse_id=mouse_id,
                neuron_id_tokens=id_tok,
            )
            
            if getattr(model, "self_attend_input_neurons", False):
                input_neuron_tokens, input_neurons_attn = model.input_neurons_attention(
                    input_neuron_tokens, output_attn_weights=True
                )  # [B, Layers, Heads, K, K]

        # --- core with attn ---
        outputs, core_attn, layer_tokens = model.core(
            image_tokens=image_tokens,
            neuron_tokens=input_neuron_tokens,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
            output_attn_weights=True,
            return_layer_tokens=True,
        )  # outputs: [B, P, C]; core_attn: [B, Layers, Heads, S, S] (S = P (+K))

        # optional: add PE after core if subselecting image tokens
        if getattr(model, "subselect_image_tokens", False) and getattr(model, "use_pe_after_core", False):
            if model.pe_after_core == "2d":
                tmp = outputs.view(outputs.size(0), model.patch_embedding.height, model.patch_embedding.width, outputs.size(-1))
                tmp = tmp + model.patch_embedding.pos_embedding(tmp)
                outputs = tmp.view(outputs.size(0), -1, outputs.size(-1))
            elif model.pe_after_core == "1d":
                outputs = outputs + model.patch_embedding.pos_embedding(outputs)

        # --- readout with attn ---
        readout_attn = None
        y_pred = None
        if getattr(model, "readout_type", None) == "attention":
            query_neurons = model.neuron_id_tokenizer[mouse_id](
                query_neuron_ids.long().unsqueeze(0).expand(images.size(0), -1)
            ) + model.session_tokenizer(model.sessions_enc[mouse_id].to(model.device))
            # + query PE if needed...
            y_pred, readout_attn = model.readouts(
                outputs, mouse_id=mouse_id,
                query_neurons=query_neurons,
                query_neuron_ids=query_neuron_ids,
                shifts=None,
                output_attn_weights=True,
            )  # y_pred: [B, Nq], readout_attn: [B, H, Nq, S] (S = P (+K) depending on the core output)

        # --- choose scalar score ---
        if neuron_idx is None:
            score = y_pred.sum()
        else:
            score = y_pred[:, neuron_idx].sum()

        # --- gradients wrt readout attn (cross-attn) ---
        grad_readout, = torch.autograd.grad(
            score, [readout_attn],
            retain_graph=True,
            create_graph=False,
        )  # [B, H, Nq, S]
        grad_readout = grad_readout.clamp(min=0.)

        # head-reduce: [B, Nq, S]
        if head_reduce == "mean":
            G_r = grad_readout.mean(dim=1)
        elif head_reduce == "max":
            G_r = grad_readout.max(dim=1).values
        else:
            raise ValueError

        # --- slice image tokens & reshape to maps ---
        P = model.core.num_image_patches
        H_P, W_P = find_shape(P)
        image_cols = slice(0, P)

        G_r_img = G_r[:, :, image_cols]                   # [B, Nq, P]
        G_r_img = resize(G_r_img.reshape(B, Nq, H_P, W_P), size=image_shape, antialias=False)
        G_r_img = normalize_along(G_r_img, reduce_dims=(-2, -1), keep_dims_shape=True)
        # G_r_img: [B, Nq, H_img, W_img] -> LeGrad query→image maps

        # --- slice input neuron tokens & reshape to maps ---
        if K > 0:
            inp_cols = slice(P, P+K)
            G_r_n = G_r[:, :, inp_cols]                   # [B, Nq, K]
            # G_r_n = normalize_along(G_r_n, reduce_dims=(-2, -1), keep_dims_shape=True)
            # G_r_n: [B, Nq, K] -> LeGrad query→neuron maps
        else:
            G_r_n = None

        # --- gradients wrt core attn layers  ---
        for i, layer_t in enumerate(layer_tokens):
            # apply readout head on intermediate layer tokens (not optimised for these layers)
            y_pred, ca_attn_core_l = model.readouts(
                    layer_t, mouse_id=mouse_id,
                    query_neurons=query_neurons,
                    query_neuron_ids=query_neuron_ids,
                    shifts=None,
                    output_attn_weights=True,
                )
            if neuron_idx is None:
                score = y_pred.sum()
            else:
                score = y_pred[:, neuron_idx].sum()
            grad_core, = torch.autograd.grad(
                score, [ca_attn_core_l],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )  # [B, L, H, S, S]
            grad_core = grad_core.clamp(min=0.)

            # head-reduce: [B, Nq, S]
            if head_reduce == "mean":
                G_core = grad_core.mean(dim=1)
            elif head_reduce == "max":
                G_core = grad_core.max(dim=1).values
            else:
                raise ValueError
            G_core_img = G_core[:, :, image_cols]                   # [B, Nq, P]
            G_core_img = resize(G_core_img.reshape(B, Nq, H_P, W_P), size=image_shape, antialias=False)
            # G_core_img = normalize_along(G_core_img, reduce_dims=(-2, -1), keep_dims_shape=True)
            # G_core_img: [B, Nq, H_img, W_img] -> LeGrad query→image maps

            # --- slice input neuron tokens & reshape to maps ---
            if K > 0:
                inp_cols = slice(P, P+K)
                G_core_n = G_core[:, :, inp_cols]                   # [B, Nq, K]
                # G_core_n = normalize_along(G_core_n, reduce_dims=(-2, -1), keep_dims_shape=True)
                # G_core_n: [B, Nq, K] -> LeGrad query→neuron maps
            else:
                G_core_n = None
            out["core_query_to_image_legrad_maps"].append(G_core_img.detach().cpu())       # [B, Nq, H_img, W_img]
            out["core_query_to_neuron_legrad_maps"].append(G_core_n.detach().cpu() if K > 0 else None)     # [B, Nq, K]
            

        # --- optional: self-attn maps ---     MODIFY to work with Nq !!!!!!
        if getattr(model, "self_attend_image_tokens", False): 
            y_pred, ca_attn_sa_images = model.readouts(
                    image_tokens, mouse_id=mouse_id,
                    query_neurons=query_neurons,
                    query_neuron_ids=query_neuron_ids,
                    shifts=None,
                    output_attn_weights=True,
                )
            if neuron_idx is None:
                score = y_pred.sum()
            else:
                score = y_pred[:, neuron_idx].sum()  
            grad_images_self_attn, = torch.autograd.grad(
                score, ca_attn_sa_images,
                retain_graph=False,
                create_graph=False,
            )  # [B, H, P, P]
            grad_images_self_attn = grad_images_self_attn.clamp(min=0.)

            if head_reduce == "mean":
                G_img_self = grad_images_self_attn.mean(dim=1)   # [B, P, P]
            else:
                G_img_self = grad_images_self_attn.max(dim=1).values

            G_img_self = G_img_self.mean(dim=1)  # [B, P]
            G_img_self = resize(G_img_self.reshape(B, H_P, W_P), size=image_shape, antialias=False)
            G_img_self = normalize_along(G_img_self, reduce_dims=(-1,), keep_dims_shape=True)
            out["image_self_attn_legrad_maps"].append(G_img_self.detach().cpu())  # [B, H_img, W_img]
        
        if getattr(model, "self_attend_input_neurons", False) and K > 0:
            y_pred, ca_attn_sa_neurons = model.readouts(
                    input_neuron_tokens, mouse_id=mouse_id,
                    query_neurons=query_neurons,
                    query_neuron_ids=query_neuron_ids,
                    shifts=None,
                    output_attn_weights=True,
                )
            if neuron_idx is None:
                score = y_pred.sum()
            else:
                score = y_pred[:, neuron_idx].sum()  
            grad_input_neurons_self_attn, = torch.autograd.grad(
                score, ca_attn_sa_neurons,
                retain_graph=False,
                create_graph=False,
            )  # [B, H, K, K]
            grad_input_neurons_self_attn = grad_input_neurons_self_attn.clamp(min=0.)

            if head_reduce == "mean":
                G_inp_self = grad_input_neurons_self_attn.mean(dim=1)   # [B, L, K, K]
            else:
                G_inp_self = grad_input_neurons_self_attn.max(dim=1).values

            G_inp_self = G_inp_self  # [B, K, K]
            # G_inp_self = normalize_along(G_inp_self, reduce_dims=(-1,-2), keep_dims_shape=True)
            out["input_neuron_self_attn_legrad_maps"].append(G_inp_self.detach().cpu())  # [B, K, K]


        out["images"].append(i_T_img(images.cpu()))
        out["input_neuron_ids"].append(input_neuron_ids.cpu().unsqueeze(0) if input_neuron_ids is not None else torch.tensor([]))
        out["query_neuron_ids"].append(query_neuron_ids.cpu().unsqueeze(0) if query_neuron_ids is not None else torch.tensor([]))
        out["behaviors"].append(i_T_beh(behaviors.cpu()).unsqueeze(0) if behaviors is not None else torch.tensor([]))
        out["pupil_centers"].append(i_T_pup(pupil_centers.cpu()).unsqueeze(0) if pupil_centers is not None else torch.tensor([]))
        # # core
        # out["core_image_legrad_maps"].append(G_core_img.detach().cpu())              # [B, L, H_img, W_img] rollout or diag-like
        # out["core_neuron_to_image_legrad_maps"].append(G_core_n2i.detach().cpu() if K > 0 else None)        # [B, L, K, H_img, W_img] 
        # out["core_neuron_to_neuron_legrad_maps"].append(G_core_n2n.detach().cpu() if K > 0 else None)       # [B, L, K, K]
        # readout
        out["readout_query_to_image_legrad_maps"].append(G_r_img.detach().cpu())       # [B, Nq, H_img, W_img]
        out["readout_query_to_neuron_legrad_maps"].append(G_r_n.detach().cpu() if K > 0 else None)     # [B, Nq, K]
        # --- early stopping ---
        if num_samples is not None:
            remaining -= images.size(0)
            if remaining <= 0:
                break

    # concatenate results
    for k, vlist in out.items():
        if len(vlist) == 0:
            continue
        if isinstance(vlist, list) and isinstance(vlist[0], torch.Tensor):
            out[k] = torch.cat(vlist, dim=0).numpy()   # concat along batch
            print("k:", k, "shape:", out[k].shape)
   
    print(out["images"].shape)
    # for k in out.keys():
    #     if len(out[k]) > 0:
    #         out[k] = torch.vstack(out[k]).numpy()

    if num_samples is not None and isinstance(out.get("images"), np.ndarray):
        for k in out.keys():
            out[k] = out[k][:num_samples]
    out["neuron_coordinates"] = neuron_coords.cpu().squeeze(0) if neuron_coords is not None else None
    return out

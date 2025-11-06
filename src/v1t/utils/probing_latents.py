import math
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

@torch.no_grad()
def sample_neuron_tokens(
    ds: DataLoader,
    model: Model,
    num_samples: int = None,
    device: torch.device = "cpu",
    verbose: int = 1,
    use_rollout: bool = True,
    head_reduce: str = "mean",
):
    model.to(device).eval()
    mouse_id = ds.dataset.mouse_id

    P = model.core.num_image_patches
    H_P, W_P = find_shape(P) # height and width in image tokens grid

    out = {
        "input_neuron_ids": [],
        "query_neuron_ids": [],
        "input_neuron_tokens": [],
    }

    remaining = num_samples

    for batch in tqdm(ds, desc=f"Mouse {mouse_id}", disable=not verbose):
        images = batch["image"].to(device)
        responses = batch["response"].to(device)
        neuron_coords = batch["neuron_coordinates"].to(device)
        behaviors = batch["behavior"].to(device)
        pupil_centers = batch["pupil_center"].to(device)
        # ids are 1D tensors as they the same for the whole batch
        input_neuron_ids = batch.get("input_neuron_ids", None)
        if input_neuron_ids is not None:
            input_neuron_ids = input_neuron_ids.to(device)

        query_neuron_ids = batch.get("query_neuron_ids", None)
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

            # build ID tokens 
            id_tok = model.neuron_id_tokenizer[mouse_id](input_neuron_ids.long())
            id_tok = id_tok + model.session_tokenizer(model.sessions_enc[mouse_id].to(model.device))

            input_neuron_tokens = model.input_neuron_embedding(
                responses=responses[:, input_neuron_ids, :],
                mouse_id=mouse_id,
                neuron_id_tokens=id_tok,
            )
            # add neuron PEs if enabled 
            if getattr(model, "use_input_neuron_pe", False):
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
                input_neuron_tokens = model.input_neurons_attention(
                    input_neuron_tokens, output_attn_weights=False
                )  # [B, Layers, Heads, K, K]

                # re-add neuron PEs 
                if getattr(model, "use_input_neuron_pe", False):
                    if model.neuron_pe_mode == "1d":
                        input_neuron_tokens += model.neuron_pe(responses)[:, input_neuron_ids, :]
                    elif model.neuron_pe_mode == "coord":
                        input_coords = neuron_coords[:, input_neuron_ids, :]
                        input_neuron_tokens += model.neuron_coord_pe(input_coords)
                    elif model.neuron_pe_mode == "both":
                        input_coords = neuron_coords[:, input_neuron_ids, :]
                        input_neuron_tokens += (model.neuron_pe(responses)[:, input_neuron_ids, :]
                                                + model.neuron_coord_pe(input_coords))

        # --- core ---
        outputs = model.core(
            image_tokens=image_tokens,
            neuron_tokens=input_neuron_tokens,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
            output_attn_weights=False,
        )  # outputs: [B, P, C]; core_attn: [B, Layers, Heads, S, S] (S = P (+K))
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

def normalize(x: t.Union[np.ndarray, torch.Tensor]):
    return (x - x.min()) / (x.max() - x.min())

def normalize_along(x, reduce_dims, keep_dims_shape, eps=1e-12):
    # reduce_dims: tuple of ints to reduce over (e.g., (-1, -2) for H,W only)
    minv = x.amin(dim=reduce_dims, keepdim=True)
    maxv = x.amax(dim=reduce_dims, keepdim=True)
    return (x - minv) / (maxv - minv + eps)

# Examples:
# per-map (reduce over H,W):     normalize_along(x, reduce_dims=(-1,-2), keep_dims_shape=True)
# per-sample (reduce K,H,W):     normalize_along(x, reduce_dims=(1,2,3), keep_dims_shape=True)
# global (reduce B,K,H,W):       normalize_along(x, reduce_dims=(0,1,2,3), keep_dims_shape=True)
# Per-map: best for visualizing individual maps (maximizes their contrast). Not comparable across maps.
# Per-sample: compare different neurons within the same sample fairly.
# Per-channel: compare the same neuron across different samples fairly.
# Global: compare everything on one absolute scale; good for quantitative comparison, but can wash out weaker maps.

def head_reduce_last(attn_BLHNN: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    """
    attn_BLHNN: [B, L, H, N, N] -> returns [B, N, N] for the last layer, head-reduced.
    """
    A_BHNN = attn_BLHNN[:, -1]           # [B, H, N, N]
    if reduce == "mean":
        return A_BHNN.mean(dim=1)        # [B, N, N]
    elif reduce == "max":
        return A_BHNN.max(dim=1).values
    else:
        raise ValueError("reduce ∈ {mean,max}")

@torch.no_grad()
def extract_attention_maps(
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
    i_T_img = ds.dataset.i_transform_image
    i_T_beh = ds.dataset.i_transform_behavior
    i_T_pup = ds.dataset.i_transform_pupil_center

    P = model.core.num_image_patches
    H_P, W_P = find_shape(P) # height and width in image tokens grid

    out = {
        "images": [],
        "input_neuron_ids": [],
        "query_neuron_ids": [],
        "behaviors": [],
        "pupil_centers": [],
        # core (rollout)
        "core_image_rollout_maps": [],              # [B, L, H_img, W_img] rollout or diag-like
        "core_neuron_to_image_rollout_maps": [],        # [B, L, K, H_img, W_img] 
        "core_neuron_to_neuron_rollout_maps": [],       # [B, L, K, K]
        # core (last layer)
        "core_neuron_to_image_maps": [],        # [B, K, H_img, W_img]
        "core_image_to_neuron_maps": [],        # [B, P, K]
        "core_neuron_to_neuron_maps": [],       # [B, K, K]
        # readout
        "readout_query_to_image_maps": [],     # [B, Nq, H_img, W_img]
        "readout_query_to_neuron_maps": [],     # [B, Nq, K]
        # self-attn (optional)
        "image_self_attn_maps": [],             # [B, H_img, W_img] 
        "input_neuron_self_attn_maps": [],      # [B, K, K] or per-neuron heatmaps
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
        images_self_attn = None
        if getattr(model, "self_attend_image_tokens", False):
            image_tokens, images_self_attn = model.image_tokens_attention(
                image_tokens, output_attn_weights=True
            )  # [B, Layers, Heads, P, P]

        # --- input neuron tokenization (+ optional input self-attn) ---
        input_neuron_tokens = None
        input_neurons_attn = None
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
                input_neuron_tokens, input_neurons_attn = model.input_neurons_attention(
                    input_neuron_tokens, output_attn_weights=True
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
        outputs, core_attn = model.core(
            image_tokens=image_tokens,
            neuron_tokens=input_neuron_tokens,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
            output_attn_weights=True,
        )  # outputs: [B, P, C]; core_attn: [B, Layers, Heads, S, S] (S = P (+K))

        # optional: add PE after core if subselecting image tokens
        if getattr(model, "subselect_image_tokens", False) and getattr(model, "use_pe_after_core", False):
            if model.pe_after_core == "2d":
                tmp = outputs.view(outputs.size(0), model.patch_embedding.height, model.patch_embedding.width, outputs.size(-1))
                tmp = tmp + model.patch_embedding.pos_embedding(tmp)
                outputs = tmp.view(outputs.size(0), -1, outputs.size(-1))
            elif model.pe_after_core == "1d":
                outputs = outputs + model.patch_embedding.pos_embedding(outputs)

        # --- readout  ---
        readout_attn = None
        query_neurons = None
        if getattr(model, "readout_type", None) == "attention":
            query_neurons = model.neuron_id_tokenizer[mouse_id](
                query_neuron_ids.long().unsqueeze(0).expand(images.size(0), -1)
            ) + model.session_tokenizer(model.sessions_enc[mouse_id].to(model.device))

            if getattr(model, "use_query_neuron_pe", False):
                if model.neuron_pe_mode == "1d":
                    add = model.neuron_pe(responses)[:, query_neuron_ids, :]
                elif model.neuron_pe_mode == "coord":
                    add = model.neuron_coord_pe(neuron_coords[:, query_neuron_ids, :])
                else:  # both
                    add = (model.neuron_pe(responses)[:, query_neuron_ids, :]
                           + model.neuron_coord_pe(neuron_coords[:, query_neuron_ids, :]))
                if getattr(model, "project_query_pe", False):
                    add = model.projection_query_pe(add)
                query_neurons = query_neurons + add

            outputs, readout_attn = model.readouts(
                outputs, mouse_id=mouse_id,
                query_neurons=query_neurons,
                query_neuron_ids=query_neuron_ids,
                shifts=None,
                output_attn_weights=True
            )  # readout_attn: [B, H, Nq, S] (S = P (+K) depending on the core output)

        # --- indices to slice attention maps ---
        image_cols = slice(0, P)
        inp_cols = slice(P, P + K)

        image_token_indices = torch.arange(0, P, device=core_attn.device)
        neuron_token_indices = None
        if K > 0:
            neuron_token_indices = torch.arange(P, P + K, device=core_attn.device)
            print("P:", P, "K:", K)

        # =========================
        #  CORE HEATMAPS
        # =========================

        # a) Rollout-based global image map (CLS-free)
        if use_rollout:
            batch_image_maps = []
            batch_neuron_to_image = []
            batch_neuron_to_neuron = []
            for b in range(images.size(0)):
                s_im = rollout(core_attn[b], slice_idx=image_cols, head_reduce=head_reduce)
                print(s_im.shape)
                # s_im = rollout_single_sample(core_attn[b], slice_idx=image_cols,
                                            #  seed="uniform_image", head_reduce=head_reduce)
                # batch_image_maps.append(s_im.view(H_P, W_P))
                s_im = [normalize(s_im[i]) for i in range(s_im.size(0))] 
                s_im = torch.stack(s_im, 0)  # [L, P]
                print("s_im", s_im.shape)
                s_im = resize(s_im.reshape(s_im.shape[0], H_P, W_P), size=image_shape, antialias=False)
                batch_image_maps.append(s_im)  # [H_img, W_img] resized

                if K > 0:
                    print(image_cols)
                    s_im_neu = rollout_multi_seed(core_attn[b], image_cols, neuron_token_indices, head_reduce)
                    print("s_im_neu", s_im_neu.shape)
                    s_im_neu = normalize(s_im_neu)
                    s_im_neu = resize(s_im_neu.transpose(0,1).contiguous().view(K, H_P, W_P), size=image_shape, antialias=False)
                    # s_im_neu: [P, K]  (rows: image tokens, cols: seeds = neurons)
                    # transpose to [K, P] then reshape each row to H×W
                    batch_neuron_to_image.append(s_im_neu)  # [K, H_img, W_img]

                    s_neu = rollout_multi_seed(core_attn[b], inp_cols, neuron_token_indices, head_reduce)
                    # s_neu: [K, K]  (rows: neuron tokens, cols: seeds = neurons)
                    batch_neuron_to_neuron.append(s_neu.T)  # [K, K] (seed per row, target per col)

            core_image_rollout_heatmaps = torch.stack(batch_image_maps, 0)  # [B, H_img, W_img]
            core_neuron_to_image_heatmaps = torch.stack(batch_neuron_to_image, 0)  # [B, K, H_img, W_img]
            core_neuron_to_neuron_heatmaps = torch.stack(batch_neuron_to_neuron, 0)  # [B, K, K]
        # else:
        #     # b) Non-rollout variants from last layer (head-mean)
        #     A_last = head_reduce_last(core_attn, reduce=head_reduce)  # [B, N, N]
        #     # image→image diagonal as a simple per-patch self-focus example (often meh, but matches your placeholder)
        #     core_image_heatmaps = A_last[:, image_cols, image_cols].diagonal(dim1=-2, dim2=-1) \
        #                           .view(images.size(0), H_img, W_img)  # [B, H_img, W_img]

        # c) Neuron→Image (last layer, head-mean), per-input-neuron
        core_neuron_to_image_maps_lastlayer = None
        # d) Image→Neuron and Neuron→Neuron (keep as matrices)
        core_image_to_neuron_maps_lastlayer = None
        core_neuron_to_neuron_maps_lastlayer = None
        if K > 0:
            A_last = head_reduce_last(core_attn, reduce=head_reduce)  # [B, N, N]
            A_n2i = A_last[:, inp_cols, image_cols]                   # [B, K, P]
            A_n2i = normalize(A_n2i)
            A_n2i = resize(A_n2i.reshape(B, K, H_P, W_P), size=image_shape, antialias=False)
            core_neuron_to_image_maps_lastlayer = A_n2i
            core_image_to_neuron_maps_lastlayer = A_last[:, image_cols, inp_cols]    # [B, P, K]
            core_neuron_to_neuron_maps_lastlayer = A_last[:, inp_cols, inp_cols]     # [B, K, K]

        # =========================
        #  READOUT HEATMAPS
        # =========================
        readout_query_to_image_maps = None
        if readout_attn is not None:
            # readout_attn: [B, H, Nq, S]; reduce heads
            A_r = readout_attn.mean(dim=1)                     # [B, Nq, S]
            A_r_img = A_r[:, :, image_cols]                    # [B, Nq, P]
            A_r_img = normalize(A_r_img)
            A_r_img = resize(A_r_img.reshape(B, Nq, H_P, W_P), size=image_shape, antialias=False)
            readout_query_to_image_maps = A_r_img  # [B, Nq, H, W]
            print("Readout query to image maps:", readout_query_to_image_maps.shape)
            if not getattr(model, "subselect_image_tokens", False) and K > 0:
                readout_query_to_neuron_maps = A_r[:, :, inp_cols] if K > 0 else None  # [B, Nq, K] or None
                
        # =========================
        #  SELF-ATTN HEATMAPS (optional)
        # =========================
        # REVIEW
        image_self_maps = None
        if images_self_attn is not None:
            A_img_last = images_self_attn[:, -1].mean(dim=1)   # [B, P, P]
            # A_img_last = normalize(A_img_last)
            # A_img_last = resize(A_img_last.view(B, H_img, W_img, ), size=image_shape, antialias=False)
            # Example: per-patch self-focus diag as a quick map
            image_self_maps = A_img_last.diagonal(dim1=-2, dim2=-1).view(images.size(0), image_shape[0], image_shape[1])
            # A_img_last = images_self_attn[:, -1].mean(dim=1)   # [B, P, P]
            # Example: per-patch self-focus diag as a quick map
            # image_self_maps = A_img_last.diagonal(dim1=-2, dim2=-1).view(images.size(0), H_img, W_img)

        input_neuron_self_maps = None
        if input_neurons_attn is not None:
            input_neuron_self_maps = input_neurons_attn[:, -1].mean(dim=1)  # [B, K, K]
        print("input neuron id shape before collect:", input_neuron_ids.shape if input_neuron_ids is not None else None)
        # --- collect outputs ---
        out["images"].append(i_T_img(images.cpu()))
        out["input_neuron_ids"].append(input_neuron_ids.cpu().unsqueeze(0) if input_neuron_ids is not None else torch.tensor([]))
        out["query_neuron_ids"].append(query_neuron_ids.cpu().unsqueeze(0) if query_neuron_ids is not None else torch.tensor([]))
        out["behaviors"].append(i_T_beh(behaviors.cpu()).unsqueeze(0) if behaviors is not None else torch.tensor([]))
        out["pupil_centers"].append(i_T_pup(pupil_centers.cpu()).unsqueeze(0) if pupil_centers is not None else torch.tensor([]))

        out["core_image_rollout_maps"].append(core_image_rollout_heatmaps.cpu())          # [B, H, W]
        out["core_neuron_to_image_rollout_maps"].append(core_neuron_to_image_heatmaps.cpu())  # [B, K, H, W]
        out["core_neuron_to_neuron_rollout_maps"].append(core_neuron_to_neuron_heatmaps.cpu())    # [B, K, K]

        if core_neuron_to_image_maps_lastlayer is not None:
            out["core_neuron_to_image_maps"].append(core_neuron_to_image_maps_lastlayer)  # list of [K, H, W]
            out["core_image_to_neuron_maps"].append(core_image_to_neuron_maps_lastlayer)  # list of [P, K]
            out["core_neuron_to_neuron_maps"].append(core_neuron_to_neuron_maps_lastlayer)# list of [K, K]

        if readout_query_to_image_maps is not None:
            out["readout_query_to_image_maps"].append(readout_query_to_image_maps)  # list of [Nq, H, W]
            if not getattr(model, "subselect_image_tokens", False) and K > 0:
                out["readout_query_to_neuron_maps"].append(readout_query_to_neuron_maps)  # list of [Nq, K]
        
        if image_self_maps is not None:
            out["image_self_attn_maps"].append(image_self_maps.cpu())          # [B, H, W]

        if input_neuron_self_maps is not None:
            out["input_neuron_self_attn_maps"].append(input_neuron_self_maps)  # list of [K, K]

        # --- early stopping ---
        if num_samples is not None:
            remaining -= images.size(0)
            if remaining <= 0:
                break
    print("out[behaviors] shape before pack:", len(out["behaviors"]) if len(out["behaviors"]) > 0 else None)
    print("out[input ids] shape before pack:", len(out["input_neuron_ids"]) if len(out["input_neuron_ids"]) > 0 else None)
    # ======= pack results =======
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

def _prune_per_row(A: torch.Tensor, discard_ratio: float) -> torch.Tensor:
    """
    A: [L,N,N] nonnegative attention (fused across heads)
    Zero out the lowest 'discard_ratio' fraction per row, per layer.
    """
    if discard_ratio <= 0.0:
        return A
    L, N, _ = A.shape
    M = A.clone()
    k_drop = int(N * discard_ratio)
    if k_drop <= 0:
        return M

    for l in range(L):
        row = M[l]                # [N,N]
        # find per-row lowest k indices
        _, idx_low = row.topk(k_drop, dim=-1, largest=False)  # [N,k_drop]
        keep = torch.ones_like(row, dtype=torch.bool)         # [N,N]
        keep.scatter_(dim=-1, index=idx_low, value=False)
        M[l] = row * keep
    return M

def rollout(attn_LHNN: torch.Tensor,
            slice_idx: slice,
            head_reduce: str = "mean", 
            discard_ratio: float = 0.3,
            ) -> torch.Tensor:
    """
    attn_LHNN: [L, H, N, N] for one sample
    slice_idx:  slice for columns of tokens of interest (e.g., slice(0, P) image patches)
    head_reduce: "mean" or "max"
    returns: 1D heat vector over sliced tokens (length = |slice_idx|)
    """
    L, H, N, _ = attn_LHNN.shape
    # head reduction
    if head_reduce == "mean":
        A = attn_LHNN.mean(dim=1)            # [L,N,N]
    elif head_reduce == "max":
        A, _ = attn_LHNN.max(dim=1)
    else:
        raise ValueError("head_reduce ∈ {mean,max}")

    # optional pruning (per row)
    if discard_ratio > 0.0:
        A = _prune_per_row(A, discard_ratio)

    # add identity & renormalize each layer s.t. rows sum to 1
    I = torch.eye(N, device=A.device, dtype=A.dtype)
    A = A + I
    A = A / A.sum(dim=-1, keepdim=True)

    print("A", A.shape)

    # rollout (left-multiply)
    J = torch.zeros_like(A)
    J[0] = A[0]                                    # [N, N]
    for l in range(1, A.size(0)):
        J[l] = A[l] @ J[l-1]

    J_mean = [J[i].mean(dim = 0) for i in range(J.size(0))]
    J_mean = torch.stack(J_mean)

    # Return only desired columns (slice)
    return J_mean[:, slice_idx]

def rollout_single_sample(attn_LHNN: torch.Tensor,
                          slice_idx: slice,
                          seed: t.Union[str,int,torch.Tensor] = "uniform_image",
                          head_reduce: str = "mean", 
                          discard_ratio: float = 0.3,
                          ) -> torch.Tensor:
    """
    attn_LHNN: [L, H, N, N] for one sample
    slice_idx:  slice for columns of tokens of interest (e.g., slice(0, P) image patches)
    seed:
      - "uniform_image": start mass uniformly over image tokens (size N) to replicate CLS token
      - int i: start from a single source token i (e.g., a neuron token index)
      - tensor s: distribution summing to 1
    head_reduce: "mean" or "max"
    returns: 1D heat vector over sliced tokens (length = |slice_idx|)
    """
    L, H, N, _ = attn_LHNN.shape
    # head reduction
    if head_reduce == "mean":
        A = attn_LHNN.mean(dim=1)            # [L,N,N]
    elif head_reduce == "max":
        A, _ = attn_LHNN.max(dim=1)
    else:
        raise ValueError("head_reduce ∈ {mean,max}")
    
    # 2) optional pruning (per row)
    if discard_ratio > 0.0:
        A = _prune_per_row(A, discard_ratio)

    # add identity & renormalize each layer s.t. rows sum to 1
    I = torch.eye(N, device=A.device, dtype=A.dtype)
    A = A + I
    A = A / A.sum(dim=-1, keepdim=True)

    # choose seed distribution s0
    if isinstance(seed, str):
        if seed == "uniform_image":
            s = torch.zeros(N, device=A.device, dtype=A.dtype)
            s[slice_idx] = 1.0
            s = s / s.sum()
        else:
            raise ValueError("unknown seed")
    elif isinstance(seed, int):
        s = torch.zeros(N, device=A.device, dtype=A.dtype); s[seed] = 1.0
    else:
        s = seed.to(A)  # assume length N, sums to 1

    # propagate s through layers: s_{l+1} = A_l^T s_l
    for l in range(L):
        s = A[l].transpose(-1, -2) @ s  

    # return only the slice
    return s[slice_idx]

def rollout_multi_seed(attn_LHNN: torch.Tensor,
                       slice_idx: slice,            # columns to return (e.g., image slice)
                       seed_indices: torch.Tensor,  # [M] indices in [0..N-1]
                       head_reduce: str = "mean",
                       discard_ratio: float = 0.3) -> torch.Tensor:
    """
    attn_LHNN: [L, H, N, N] for one sample (self-attn)
    seed_indices: length-M tensor of token indices used as seeds
    returns: [|slice_idx|, M] matrix with rollout mass over the slice for each seed
    """
    L, H, N, _ = attn_LHNN.shape
    # Head reduce
    if head_reduce == "mean":
        A = attn_LHNN.mean(dim=1)               # [L, N, N]
    elif head_reduce == "max":
        A, _ = attn_LHNN.max(dim=1)
    else:
        raise ValueError("head_reduce ∈ {mean,max}")

    # optional pruning (per row)
    if discard_ratio > 0.0:
        A = _prune_per_row(A, discard_ratio)

    # Residual correction + renorm
    I = torch.eye(N, device=A.device, dtype=A.dtype)
    A = A + I
    A = A / A.sum(dim=-1, keepdim=True)

    # rollout (left-multiply)
    J = torch.zeros_like(A)
    J[0] = A[0]                                    # [N, N]
    for l in range(1, A.size(0)):
        J[l] = A[l] @ J[l-1]
    print("J", J.shape)
    J_last = J[-1][seed_indices]
    print("slice_idx", slice_idx)
    print("J_last", J_last[:, slice_idx].shape)

    # Return only desired columns (slice)
    return J[-1][seed_indices][:, slice_idx]               # [|slice|, M]


def rollout_multi_seed_col(attn_LHNN: torch.Tensor,
                       slice_idx: slice,            # columns to return (e.g., image slice)
                       seed_indices: torch.Tensor,  # [M] indices in [0..N-1]
                       head_reduce: str = "mean",
                       discard_ratio: float = 0.3) -> torch.Tensor:
    """
    attn_LHNN: [L, H, N, N] for one sample (self-attn)
    seed_indices: length-M tensor of token indices used as seeds
    returns: [|slice_idx|, M] matrix with rollout mass over the slice for each seed
    """
    L, H, N, _ = attn_LHNN.shape
    # Head reduce
    if head_reduce == "mean":
        A = attn_LHNN.mean(dim=1)               # [L, N, N]
    elif head_reduce == "max":
        A, _ = attn_LHNN.max(dim=1)
    else:
        raise ValueError("head_reduce ∈ {mean,max}")

    # optional pruning (per row)
    if discard_ratio > 0.0:
        A = _prune_per_row(A, discard_ratio)

    # Residual correction + renorm
    I = torch.eye(N, device=A.device, dtype=A.dtype)
    A = A + I
    A = A / A.sum(dim=-1, keepdim=True)

    # Build seed matrix S0: [N, M]
    M = seed_indices.numel()
    S = torch.zeros(N, M, device=A.device, dtype=A.dtype)
    S[seed_indices, torch.arange(M, device=A.device)] = 1.0

    # Propagate S through layers: S_{l+1} = A_l^T S_l
    for l in range(L):
        S = A[l].transpose(-1, -2) @ S   # [N, M]

    # Return only desired columns (slice)
    return S[slice_idx, :]               # [|slice|, M]


# def attention_rollout(attention: torch.Tensor, image_shape: t.List[int]):
#     """
#     Apply Attention rollout from https://arxiv.org/abs/2005.00928 to a single
#     sample of softmax attention
#     Code examples
#     - https://keras.io/examples/vision/probing_vits/#method-ii-attention-rollout
#     - https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
#     """
#     assert attention.dim() == 4
#     with torch.device(attention.device):
#         # take max values of attention heads
#         attention, _ = torch.max(attention, dim=1)

#         # to account for residual connections, we add an identity matrix to the
#         # attention matrix and re-normalize the weights.
#         residual_att = torch.eye(attention.size(1))
#         aug_att_mat = attention + residual_att
#         aug_att_mat = aug_att_mat / torch.sum(aug_att_mat, dim=-1, keepdim=True)

#         # recursively multiply the weight matrices
#         joint_attentions = torch.zeros_like(aug_att_mat)
#         joint_attentions[0] = aug_att_mat[0]

#         for n in range(1, aug_att_mat.size(0)):
#             joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

#         heatmap = joint_attentions[-1, 0, 1:]
#         heatmap = torch.reshape(heatmap, shape=find_shape(len(heatmap)))
#         heatmap = normalize(heatmap)
#         heatmap = resize(heatmap[None, ...], size=image_shape, antialias=False)
#     return heatmap[0]


# def attention_rollouts(attentions: torch.Tensor, image_shape: t.List[int]):
#     """Apply attention rollout to a batch of softmax attentions"""
#     assert attentions.dim() == 5
#     batch_size = attentions.size(0)
#     with torch.device(attentions.device):
#         heatmaps = torch.zeros((batch_size, *image_shape))
#         for i in range(batch_size):
#             heatmaps[i] = attention_rollout(attentions[i], image_shape=image_shape)
#     return heatmaps


# @torch.no_grad()
# def extract_attention_maps(
#     ds: DataLoader,
#     model: Model,
#     num_samples: int = None,
#     device: torch.device = "cpu",
#     verbose: int = 1,
# ) -> t.Dict[str, np.ndarray]:
#     """Extract attention rollout maps for a given DataLoader ds.

#     Args:
#         ds: DataLoader, DataLoader for a MouseDataset
#         model: Model, model with ViT/V1T core
#         num_samples: int, number of samples to extract, extract all samples if None.
#         device: torch.device, device to run on.
#     """
#     model.to(device)
#     model.train(False)

#     mouse_id = ds.dataset.mouse_id
#     i_transform_image = ds.dataset.i_transform_image
#     i_transform_behavior = ds.dataset.i_transform_behavior
#     i_transform_pupil_center = ds.dataset.i_transform_pupil_center

#     recorder = Recorder(model.core)
#     results = {"images": [], "heatmaps": [], "pupil_centers": [], "behaviors": []}

#     count = num_samples
#     for batch in tqdm(ds, desc=f"Mouse {mouse_id}", disable=not verbose):
#         images = batch["image"].to(device)
#         behaviors = batch["behavior"].to(device)
#         pupil_centers = batch["pupil_center"].to(device)
#         images, _ = model.image_cropper(
#             inputs=images,
#             mouse_id=mouse_id,
#             behaviors=behaviors,
#             pupil_centers=pupil_centers,
#         )
#         _, attentions = recorder(
#             images=images,
#             behaviors=behaviors,
#             pupil_centers=pupil_centers,
#             mouse_id=mouse_id,
#         )
#         recorder.clear()

#         # extract attention rollout maps within the loop in order to avoid OOM
#         heatmaps = attention_rollouts(
#             attentions=attentions, image_shape=images.shape[2:]
#         )

#         results["images"].append(i_transform_image(images.cpu()))
#         results["heatmaps"].append(heatmaps.cpu())
#         results["behaviors"].append(i_transform_behavior(behaviors.cpu()))
#         results["pupil_centers"].append(i_transform_pupil_center(pupil_centers.cpu()))

#         if num_samples is not None and (count := count - len(images)) <= 0:
#             break

#     recorder.eject()
#     del recorder

#     results = {k: torch.vstack(v).numpy() for k, v in results.items()}
#     if num_samples is not None:
#         results = {k: v[:num_samples] for k, v in results.items()}
#     return results

import sys
import os
import pickle
import torch
import numpy as np
import typing as t
import argparse
import numpy as np
import wandb
import json, yaml, csv, time
from bisect import bisect_right

from tqdm import trange
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from v1t.utils import utils, tensorboard
from v1t import data
from v1t.models.model import Model
from v1t.utils.scheduler import Scheduler
from v1t.utils.neuron_representations import sample_neuron_tokens

utils.set_random_seed(1234)

from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

tensorboard.set_font()

# def get_neuron_indices(rep_dir, neuron_fraction=1.0, neuron_seed=42):
#     """Get neuron indices for nested subsampling"""
#     # Get N from first training file
#     train_dir = os.path.join(rep_dir, 'train')
#     first_file = os.listdir(train_dir)[0]
#     first_arr = np.load(os.path.join(train_dir, first_file), mmap_mode='r')
#     N = first_arr.shape[1]  # number of neurons
    
#     # Create nested neuron indices (reproducible)
#     np.random.seed(neuron_seed)
#     all_indices = np.random.permutation(N)
#     n_neurons = int(N * neuron_fraction)
#     selected_indices = np.sort(all_indices[:n_neurons])  # Sort to maintain order
    
#     print(f"Selected {n_neurons}/{N} neurons ({neuron_fraction*100:.1f}%) with seed {neuron_seed}")
#     return selected_indices, N

def get_neuron_indices(rep_dir, neuron_fraction=1.0, neuron_seed=42, idx_folder=None, save_folder=None):
    """Get neuron indices for nested subsampling"""
    # Try to load existing indices
    if idx_folder is not None:
        file_path = os.path.join(save_folder, f"input_frac_{neuron_fraction:.3f}_seed_{neuron_seed}.pkl")
        try:
            with open(file_path, 'rb') as f:
                neuron_data = pickle.load(f)
            input_indices = neuron_data['input_neuron_indices']
            query_indices = neuron_data['query_neuron_indices']
            print(f"Loaded input/query neuron indices from {file_path}")
            
            # Get N from first training file (for verification)
            train_dir = os.path.join(rep_dir, 'train')
            fnames = sorted([f for f in os.listdir(train_dir) if f.endswith('.npy')], 
                           key=lambda x: int(x.split('.')[0]))  # Consistent sorting
            first_file = fnames[0]
            first_arr = np.load(os.path.join(train_dir, first_file), mmap_mode='r')
            N = first_arr.shape[1]  # number of neurons
            
            # Verify loaded data is consistent
            if neuron_data['total_neurons'] != N:
                print(f"Warning: Loaded N={neuron_data['total_neurons']} but current data has N={N}")
            
            return input_indices, query_indices, N
            
        except (FileNotFoundError, KeyError, pickle.UnpicklingError) as e:
            print(f"Could not load existing indices ({e}), generating new ones...")
    else:
        print("No idx_folder provided, generating new neuron indices...")

        train_dir = os.path.join(rep_dir, 'train')
        first_file = os.listdir(train_dir)[0]
        first_arr = np.load(os.path.join(train_dir, first_file), mmap_mode='r')
        N = first_arr.shape[1]  # number of neurons
        
        # create nested neuron indices (reproducible)
        np.random.seed(neuron_seed)
        all_indices = np.random.permutation(N)
        n_neurons = int(N * neuron_fraction)

        # input neurons
        input_indices = np.sort(all_indices[:n_neurons])
        # complement = query neurons
        query_indices = np.sort(all_indices[n_neurons:])
        # mask = np.ones(N, dtype=bool)
        # mask[input_indices] = False
        # query_indices = np.nonzero(mask)[0]

        print(f"Selected {n_neurons}/{N} neurons ({neuron_fraction*100:.1f}%) "
            f"with seed {neuron_seed}")
        if save_folder is not None:
            neuron_data = {
                'input_neuron_indices': input_indices,
                'query_neuron_indices': query_indices,
                'neuron_fraction': neuron_fraction,
                'neuron_seed': neuron_seed,
                'total_neurons': N,
                'n_selected': n_neurons
            }
            
            # Create filename
            filename = f"input_frac_{neuron_fraction:.3f}_seed_{neuron_seed}.pkl"
            filepath = os.path.join(save_folder, filename)
            os.makedirs(save_folder, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(neuron_data, f)
    return input_indices, query_indices, N


def _nat_sorted_npy(dirpath):
    fnames = [f for f in os.listdir(dirpath) if f.endswith('.npy')]
    fnames = sorted(fnames, key=lambda x: int(x.split('.')[0]))
    print(fnames)
    return fnames

def compute_normalization_stats(
    rep_dir: str,
    tier: str = "train",
    pooling: str = "mean",         # one of {"mean","max","min"} or any other string → no pooling (global per-feature)
    neuron_indices: np.ndarray | None = None,
):
    """
    Compute feature-wise mean/std for normalization from training representations.

    Modes
    -----
    1) pooling in {"mean","max","min"}:
       - Pool each trial array [N,152] → a single [152] vector (per trial),
         then compute mean/std across trials for those pooled vectors.
       - Result: mean,std shapes [152].

    2) any other pooling string (interpreted as "no pooling"):
       - Compute *global per-feature* stats over BOTH trials and neurons
         on arrays shaped [B,N,152], i.e., across axes (0,1).
       - Result: mean,std shapes [152].

    Parameters
    ----------
    rep_dir : str
        Directory with subfolders 'train'/'validation'/'test' containing .npy files (each [B,N,152]).
    tier : str
        Which split to read from.
    pooling : str
        "mean"/"max"/"min" for pooled stats; anything else → no pooling (global stats over trials×neurons).
    neuron_indices : np.ndarray | None
        Optional subset of neuron indices to include (nested subsampling).

    Returns
    -------
    mean_x : np.ndarray, shape [152]
    std_x  : np.ndarray, shape [152]
    """
    tier_dir = os.path.join(rep_dir, tier)
    fnames = sorted([f for f in os.listdir(tier_dir) if f.endswith(".npy")],
                    key=lambda x: int(x.split(".")[0]))
    if not fnames:
        raise FileNotFoundError(f"No .npy files found in {tier_dir}")

    print(f"Found {len(fnames)} {tier} files...")
    if neuron_indices is not None:
        print(f"Using {len(neuron_indices)} selected neurons for stats")
    else:
        print("Using all neurons for stats")

    pooled_mode = pooling in {"mean", "max", "min"}

    # -----------------------
    # first pass: compute mean
    # -----------------------
    sum_x = None
    count = 0  # number of pooled samples (pooled mode) OR total (trials×neurons) count (no-pool mode)

    for fname in fnames:
        arr = np.load(os.path.join(tier_dir, fname), mmap_mode="r")  # [B,N,152]
        if neuron_indices is not None:
            arr = arr[:, neuron_indices, :]  # [B,n_sel,152]

        if pooled_mode:
            # pool per trial: [B,N,152] → [B,152]
            if pooling == "mean":
                pooled = arr.mean(axis=1)
            elif pooling == "max":
                pooled = arr.max(axis=1)
            else:  # pooling == "min"
                pooled = arr.min(axis=1)
            if sum_x is None:
                sum_x = pooled.sum(axis=0).astype(np.float64)  # [152]
            else:
                sum_x += pooled.sum(axis=0)
            count += pooled.shape[0]                            # B
        else:
            # no pooling: global per-feature stats over trials×neurons
            # sum over axes (0,1): [B,N,152] → [152]
            s = arr.sum(axis=(0, 1)).astype(np.float64)
            if sum_x is None:
                sum_x = s
            else:
                sum_x += s
            count += arr.shape[0] * arr.shape[1]               # B * N

    if count <= 1:
        raise ValueError("Not enough samples to compute statistics.")

    mean_x = (sum_x / count).astype(np.float32)                # [152]

    # ------------------------
    # second pass: compute std
    # ------------------------
    sum_sq = np.zeros_like(mean_x, dtype=np.float64)

    for fname in fnames:
        arr = np.load(os.path.join(tier_dir, fname), mmap_mode="r")  # [B,N,152]
        if neuron_indices is not None:
            arr = arr[:, neuron_indices, :]

        if pooled_mode:
            if pooling == "mean":
                pooled = arr.mean(axis=1)       # [B,152]
            elif pooling == "max":
                pooled = arr.max(axis=1)
            else:
                pooled = arr.min(axis=1)
            diff = pooled - mean_x[None, :]     # [B,152]
            sum_sq += (diff ** 2).sum(axis=0)   # [152]
        else:
            diff = arr - mean_x[None, None, :]  # [B,N,152]
            sum_sq += (diff ** 2).sum(axis=(0, 1))  # [152]

    std_x = np.sqrt(sum_sq / (count - 1)).astype(np.float32)   # [152]

    # logs
    print(f"Computed stats from count={count} elements"
          f" ({'pooled trials' if pooled_mode else 'trials×neurons'})")
    print(f"Mean range: [{mean_x.min():.3f}, {mean_x.max():.3f}]")
    print(f"Std  range: [{std_x.min():.3f}, {std_x.max():.3f}]")

    return mean_x, std_x


class ReprLatentDataset(Dataset):
    def __init__(self, 
                 latents_path: str, 
                 rep_dir: str, 
                 tier: str,
                 pooling: str = 'mean',
                 normalize: bool = False,
                 norm_mean: np.ndarray | None = None,
                 norm_std:  np.ndarray | None = None,
                 neuron_indices: np.ndarray | None = None,
                 N: int | None = None):
        assert tier in ('train','validation','test')
        self.pooling = pooling
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std  = norm_std
        self.neuron_indices = neuron_indices
        self.N = N

        # latents_path should be a single .npy of shape [T_all, k]; we split by tiers.npy
        self.z = self._load_latents(latents_path, tier)            # [T,k]

        self.tier_repr_dir = os.path.join(rep_dir, tier)
        self.fnames = _nat_sorted_npy(self.tier_repr_dir)
        if not self.fnames:
            raise FileNotFoundError(f"No .npy files in {self.tier_repr_dir}")

        first = np.load(os.path.join(self.tier_repr_dir, self.fnames[0]), mmap_mode='r')  # [B, N, 152]
        self.B = int(first.shape[0])   # nominal rows per file (last file may be smaller)
        
        # Light consistency check (not fatal):
        expected = (len(self.z) + self.B - 1) // self.B
        if expected != len(self.fnames):
            pass

        # Calculate ACTUAL total samples across all files
        self.total_samples = 0
        self.file_offsets = [0]  # Cumulative sample counts
        
        for fname in self.fnames:
            arr = np.load(os.path.join(self.tier_repr_dir, fname), mmap_mode='r')
            samples_in_file = arr.shape[0]
            self.total_samples += samples_in_file
            self.file_offsets.append(self.total_samples)
        
        print(f"Total samples: {self.total_samples}, Expected: {len(self.z)}")
        assert self.total_samples == len(self.z), f"Mismatch: {self.total_samples} vs {len(self.z)}"

        self._current_file_idx = None
        self._current_array = None

    def _get_array_for_file(self, k: int):
        if self._current_file_idx != k:
            path = os.path.join(self.tier_repr_dir, self.fnames[k])
            self._current_array = np.load(path, mmap_mode='r')
            self._current_file_idx = k
        return self._current_array

    def __len__(self):
        return self.total_samples  

    def _load_latents(self, path_to_latents: str, tier: str) -> np.ndarray:
        z = np.load(path_to_latents, mmap_mode='r')   # [T_all, k]
        merged_root = '/user/azhar.akhmetova/merged_sensorium'
        tiers = np.load(os.path.join(merged_root, 'meta', 'trials', 'tiers.npy'))
        if tier == 'train':
            idx = np.nonzero(tiers == 'train')[0]
        elif tier == 'validation':
            idx = np.nonzero(tiers == 'validation')[0]
        else:
            idx = np.nonzero(tiers == 'test')[0]
        return z[idx]  # [T,k]

    def _pool(self, x: np.ndarray) -> np.ndarray:
        # x shape: [N, 152] - subsample neurons first if specified
        if self.neuron_indices is not None:
            x = x[self.neuron_indices]  # [n_selected, 152]

        if self.pooling == 'mean':
            v = x.mean(axis=0)
        elif self.pooling == 'max':
            v = x.max(axis=0)
        elif self.pooling == 'min':
            v = x.min(axis=0)
        elif self.pooling == 'none':
            return x
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")
        return v

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        if self.norm_mean is not None and self.norm_std is not None:
            std = np.where(self.norm_std == 0, 1.0, self.norm_std)
            v = (v - self.norm_mean) / std
        return v.astype(np.float32)  

    def __getitem__(self, i: int):
        # b = i // self.B
        # r = i %  self.B
        # arr = np.load(os.path.join(self.tier_repr_dir, self.fnames[b]), mmap_mode='r')  # [B_i, N, 152]
        # # guard if last file shorter:
        # if r >= arr.shape[0]:
        #     r = arr.shape[0] - 1
        # Find which file contains sample i
        # for file_idx, offset in enumerate(self.file_offsets[1:]):
        #     if i < offset:
        #         break
        # # Calculate position within that file
        # start_offset = self.file_offsets[file_idx]
        # r = i - start_offset
        # # Load the file and get the sample
        # arr = np.load(os.path.join(self.tier_repr_dir, self.fnames[file_idx]), mmap_mode='r')
        # x = self._pool(arr[r])                                      # [152] or [N,152]
        # if self.normalize:
        #     x = self._normalize(x)
        # y = self.z[i].astype(np.float32)            # [k]
        if i < 0 or i >= self.total_samples:
            raise IndexError(f"i={i} out of range [0,{self.total_samples})")

        # self.file_offsets = [0, len0, len0+len1, ..., total]
        k = bisect_right(self.file_offsets, i) - 1           # file index
        start = self.file_offsets[k]
        r = i - start                                        # row within file

        arr = self._get_array_for_file(k)        
        if r >= arr.shape[0]:
            raise IndexError(f"Row r={r} out of bounds for file {self.fnames[k]} (B={arr.shape[0]})")

        x = self._pool(arr[r])                               # [152] because ds_pooling='max'
        if self.normalize:
            x = self._normalize(x)
        y = self.z[i].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

class AttnPool(torch.nn.Module):
    def __init__(self, d, n_queries=1):
        super().__init__()
        self.n_queries = n_queries
        self.q = torch.nn.Parameter(torch.randn(n_queries, d))
        # self.key = torch.nn.Linear(d, d)
        # self.val = torch.nn.Linear(d, d)

    def forward(self, x):  # x: [B, N, d]
        # K = self.key(x)                        # [B, N, d]
        # V = self.val(x)                        # [B, N, d]
        # scores per trial: [B, n_queries, N]
        scores = torch.einsum('qd,bnd->bqn', self.q, x) / (x.shape[-1] ** 0.5)
        w = scores.softmax(dim=-1)             # [B, n_queries, N]
        pooled = torch.einsum('bqn,bnd->bqd', w, x)  # [B, n_queries, d]
        return pooled.squeeze(1) if self.n_queries == 1 else pooled  # [B,d] or [B,q,d]

class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, pooling=None, N=None, n_queries=1):
        super(LinearRegression, self).__init__()
        self.pooling = pooling
        self.project_queries = None
        if pooling == "linear":
            self.N = N
            self.pool = torch.nn.Linear(N, 1)
        elif pooling == "attention":
            self.pool = AttnPool(d=input_dim, n_queries=n_queries)
            if n_queries > 1:
                self.project_queries = torch.nn.Linear(n_queries, 1)
        self.linear = torch.nn.Linear(input_dim, output_dim)
                
    def forward(self, x):
        if self.pooling == "linear":
            x = self.pool(x.transpose(1,2)).squeeze(-1)  # [B, input_dim]
        elif self.pooling == "attention":
            x = self.pool(x)
            if x.dim() > 2:
                x = self.project_queries(x.transpose(1,2)).squeeze(-1)  # [B, input_dim]
        outputs = self.linear(x)
        return outputs

def l1_penalty(module: torch.nn.Module) -> torch.Tensor:
    l1 = 0.0
    for p in module.parameters():
        if p.requires_grad:
            l1 = l1 + p.abs().sum()
    return l1

def train(dataloader, optimizer, model, loss_fn, device, l1_lambda: float=0.0):
    """ method to train the model """
    epoch_loss = []
    epoch_correct, epoch_total = 0, 0

    model.train()
    # for x, y in fastprogress.progress_bar(dataloader, parent=master_bar):
    for x, y in dataloader:
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x.to(device))
        epoch_total += len(y)

        # Compute loss
        loss_mse = loss_fn(y_pred, y.to(device))

        if l1_lambda > 0:
            loss_total = loss_mse + l1_lambda * l1_penalty(model)
        else:
            loss_total = loss_mse
        # Backward pass
        loss_total.backward()
        optimizer.step()

        # For plotting the train loss, save it for each sample
        epoch_loss.append(loss_mse.item())

    # Return the mean loss and the accuracy of this epoch
    return np.mean(epoch_loss)    

# def validate(dataloader, model, loss_fn, device, master_bar):
#     """ method to compute the metrics on the validation set """
#     epoch_loss = []

#     model.eval()

#     with torch.no_grad():
#         # for x, y in fastprogress.progress_bar(dataloader, parent=master_bar):
#         for x, y in dataloader:
#             # make a prediction on validation set
#             y_pred = model(x.to(device))

#             # Compute loss
#             loss = loss_fn(y_pred, y.to(device))

#             # For plotting the train loss, save it
#             epoch_loss.append(loss.item())

#     # Return the mean loss and the accuracy
#     return np.mean(epoch_loss)

def validate(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            # loss_fn assumed to return mean over batch
            batch_loss = loss_fn(y_pred, y).item()
            bsz = y.shape[0]
            total_loss += batch_loss * bsz   # de-mean to sum
            total_count += bsz
    return total_loss / max(total_count, 1)

def report(dataloader, model, device):
    model.eval()

    sum_sq, count = 0.0, 0
    ys, yhats = [], []

    with torch.no_grad():
        for x, y in dataloader:
            y = y.to(device)
            yhat = model(x.to(device))

            # True global MSE (independent of batch sizes)
            sum_sq += F.mse_loss(yhat, y, reduction="sum").item()
            count  += y.numel()

            # stash for R² / Pearson
            ys.append(y.cpu().numpy())
            yhats.append(yhat.cpu().numpy())

    mse = sum_sq / count
    y     = np.concatenate(ys, axis=0)
    yhat  = np.concatenate(yhats, axis=0)

    r2_avg  = r2_score(y, yhat, multioutput="uniform_average")
    r2_each = r2_score(y, yhat, multioutput="raw_values")
    pearson_each = np.array([pearsonr(y[:, j], yhat[:, j])[0] for j in range(y.shape[1])])

    return {
        "mse": float(mse),
        "r2_avg": float(r2_avg),
        "r2_each": r2_each,          # np.ndarray
        "pearson_each": pearson_each # np.ndarray
    }


def run_training_with_early_stopper(args, model, optimizer, scheduler, loss_function, device, num_epochs,
                 train_dataloader, val_dataloader, tol=5e-4, epoch_tol=4, min_epochs=7, ckpt_dir=None):
    print("Starting training...")
    # master_bar = fastprogress.master_bar(range(num_epochs))
    master_bar = trange(num_epochs)
    train_losses, val_losses = [],[]
    best_path = os.path.join(ckpt_dir, "best.pt")
    best_epoch_val_loss = float('inf')
    counter = epoch_tol


    for epoch in master_bar:
        # Train the model. Use the functions defined above.
        epoch_train_loss = train(train_dataloader, optimizer, model, loss_function, device, l1_lambda=args.l1_reg)
        # Compute metrics on validation set. Use functions defined above.
        epoch_val_loss = validate(val_dataloader, model, loss_function, device)
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"epoch": epoch, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss, "lr": current_lr})
        torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": epoch_val_loss,
        "config": dict(wandb.config),
        }, os.path.join(ckpt_dir, "last.pt"))

        if epoch >= min_epochs:
            if epoch_val_loss + tol < best_epoch_val_loss:
                best_epoch_val_loss = epoch_val_loss
                best_epoch = epoch
                counter = epoch_tol
                if best_path is not None:
                    torch.save({
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_loss": best_epoch_val_loss,
                        "config": dict(wandb.config),
                    }, best_path)
                wandb.run.summary["best_val_loss"] = best_epoch_val_loss
                wandb.run.summary["best_epoch"] = best_epoch
                wandb.run.summary["best_ckpt_path"] = best_path
            else:
                counter -= 1
                if counter == 0:
                    print(f"Early stopping at epoch {epoch}. Best val loss: {best_epoch_val_loss:.4f} at epoch {best_epoch}.")
                    break
        wandb.run.summary["best_val_loss"] = best_epoch_val_loss

        # Save losses and accuracies for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # master_bar.write(f'Train loss: {epoch_train_loss:.2f}, val loss: {epoch_val_loss:.2f}')

    return train_losses, val_losses

def main(args):
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")

    utils.get_device(args)
    utils.set_random_seed(1234)

    utils.load_args(args)

    print("no_shuffle:", args.no_shuffle)
    train_ds, val_ds, test_ds = data.get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = Model(args, ds=val_ds)
    model.eval()

    scheduler = Scheduler(args, model=model, save_optimizer=False)
    scheduler.restore(force=True)

    if args.representation_type == "initial":
        rep_dir = args.output_dir  + "/neuron_token_representations/k_4_a_03_xyz_b1e7_poisson/" + args.representation_type
    elif args.representation_type == "after_sa":
        rep_dir = args.output_dir  + "/neuron_token_representations/k_4_a_03_xyz_b1e7_poisson/" + args.representation_type
    else:
        rep_dir = args.output_dir  + "/neuron_token_representations/k_4_a_03_xyz_b1e7_poisson/" + args.representation_type + f"/sel_frac_{args.select_frac_neurons}"
    latents_path = args.latents_path
    if args.representation_type == "initial":
        neuron_indices, query_indices, N = get_neuron_indices(rep_dir, neuron_fraction=args.select_frac_neurons, neuron_seed=42, idx_folder=args.idx_folder)
    else:
        neuron_indices = None

    ds_pooling = args.ds_pooling
    learned_pooling = None if ds_pooling != "none" else args.learned_pooling

    normalize = True
    if normalize:
        mean_x, std_x = compute_normalization_stats(rep_dir, tier="train", pooling=ds_pooling, neuron_indices=neuron_indices)

    train_ds, val_ds, test_ds = {}, {}, {}
    train_loader, val_loader, test_loader = {}, {}, {}
    mouse_id = args.mouse_ids[0]  # just one mouse for now for mouse_id in args.mouse_ids:
    
    train_ds[mouse_id] = ReprLatentDataset(
                latents_path=latents_path,
                rep_dir=rep_dir,
                tier="train",
                pooling=ds_pooling,
                normalize=normalize,
                norm_mean=mean_x if normalize else None,
                norm_std=std_x if normalize else None,
                neuron_indices=neuron_indices
                )
    val_ds[mouse_id] = ReprLatentDataset(
                latents_path=latents_path,
                rep_dir=rep_dir,
                tier="validation",
                pooling=ds_pooling,
                normalize=normalize,
                norm_mean=mean_x if normalize else None,
                norm_std=std_x if normalize else None,
                neuron_indices=neuron_indices
                )
    test_ds[mouse_id] = ReprLatentDataset(
                latents_path=latents_path,
                rep_dir=rep_dir,
                tier="test",
                pooling=ds_pooling,
                normalize=normalize,
                norm_mean=mean_x if normalize else None,
                norm_std=std_x if normalize else None,
                neuron_indices=neuron_indices
                )
    N = train_ds[mouse_id].__getitem__(0)[0].shape[0] if ds_pooling == "none" else train_ds[mouse_id].N
    print(f"Total neurons N: {N}")
    train_loader[mouse_id] = DataLoader(train_ds[mouse_id], batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_loader[mouse_id] = DataLoader(val_ds[mouse_id], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    test_loader[mouse_id] = DataLoader(test_ds[mouse_id], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    
    input_dim = train_ds[mouse_id].__getitem__(0)[0].shape[-1]
    print(f"Input dim: {input_dim}")

    output_dim = train_ds[mouse_id].__getitem__(0)[1].shape[0]
    print(f"Output dim: {output_dim}")

    num_epochs = 100
    linear_probe = LinearRegression(input_dim, output_dim, pooling=learned_pooling, N=N)
    linear_probe.to(args.device)
    loss_function = torch.nn.MSELoss()

    if args.reg_type == "lasso":
        args.weight_decay = 0.0  # disable L2 if using L1
    if args.reg_type == "ridge":
        args.l1_reg = 0.0        # disable L1 if using L2
    decay, no_decay = [], []
    for name, p in linear_probe.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    run = wandb.init(
    project="linear-probe-v1t",
    entity="azharakhmetova-master-thesis",
    name=f"{args.frac_input_neurons}_frac_n_{args.select_frac_neurons}_initT_reg_{args.reg_type}_ds_pool_{ds_pooling}_lpool_{learned_pooling}_lr_{args.lr:.5g}_wd_{args.weight_decay:.5g}_l1_{args.l1_reg:.5g}",
    group=f"frac_n_{args.select_frac_neurons}_k_4_a_03_xyz_b1e7_poisson",
    tags=[f"frac={args.select_frac_neurons}", f"ds_pool={ds_pooling}", f"reg={args.reg_type}"],
    config=dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        l1_reg=args.l1_reg,
        ds_pooling=ds_pooling,
        learned_pooling=learned_pooling,
        normalize=normalize,
        neuron_fraction=args.select_frac_neurons,
        input_dim=input_dim,
        output_dim=output_dim,
        batch_size=args.batch_size,
        optimizer="AdamW",
        scheduler="ReduceLROnPlateau",
        loss="MSE",
        ),
    )
    # --- Per-sweep/run directories ---
    sweep_id = os.environ.get("WANDB_SWEEP_ID")  # set by wandb agent
    if ds_pooling != "none":
        base_dir  = os.path.join(args.output_dir, "analysis", args.representation_type, f"sel_frac_{args.select_frac_neurons}", f"{args.ds_pooling}_p", f"sweep-{sweep_id or 'nosweep'}")
    else:
        base_dir  = os.path.join(args.output_dir, "analysis", args.representation_type, f"sel_frac_{args.select_frac_neurons}", f"{args.learned_pooling}_p", f"sweep-{sweep_id or 'nosweep'}")
    os.makedirs(base_dir, exist_ok=True)

    run_dir  = os.path.join(base_dir, f"run-{wandb.run.id}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save the *run* config
    with open(os.path.join(run_dir, "run_config.yaml"), "w") as f:
        yaml.safe_dump(dict(wandb.config), f)

    # Save the *sweep* config if available
    if sweep_id:
        try:
            api = wandb.Api()
            sweep = api.sweep(f"{run.entity}/{run.project}/{sweep_id}")
            with open(os.path.join(base_dir, "sweep_config.yaml"), "w") as f:
                yaml.safe_dump(sweep.config, f)
            with open(os.path.join(base_dir, "sweep_id.txt"), "w") as f:
                f.write(f"{run.entity}/{run.project}/{sweep_id}\n")
        except Exception as e:
            print("Warning: could not fetch sweep config:", e)


    train_losses, val_losses = run_training_with_early_stopper(args, linear_probe, optimizer, sched, loss_function, args.device, num_epochs,
                                                              train_loader[mouse_id], val_loader[mouse_id], ckpt_dir=ckpt_dir)
    
    # --- Load best and eval on val & test (so each run is self-contained) ---
    best_path = wandb.run.summary.get("best_ckpt_path", os.path.join(ckpt_dir,"best.pt"))
    state = None
    if os.path.exists(best_path):
        try:
            state = torch.load(best_path, map_location=args.device, weights_only=True)
        except Exception as e:
            print(f"[warn] failed to load best: {e}")

    if state is None:
        last_path = os.path.join(ckpt_dir, "last.pt")
        if os.path.exists(last_path):
            try:
                state = torch.load(last_path, map_location=args.device, weights_only=True)
                wandb.summary["eval_used_last_ckpt"] = True
            except Exception as e:
                print(f"[warn] failed to load last: {e}")

    if state is None:
        wandb.summary.update({
            "eval_skipped": True,
            "eval_reason": "no_valid_checkpoint_found"
        })
        # exit cleanly so “stopped/killed” doesn’t look like a crash
        raise SystemExit(0)
    
    linear_probe.load_state_dict(state["model"])

    val_report  = report(val_loader[mouse_id],  linear_probe, args.device)
    test_report = report(test_loader[mouse_id], linear_probe, args.device)
    wandb.run.summary.update({"final_val_mse": float(val_report["mse"]), "final_test_mse": float(test_report["mse"])})


    # --- Append to a sweep-level CSV for quick plotting later ---
    # csv_path = os.path.join(base_dir, "metrics.csv")
    csv_path = os.path.join(args.output_dir, "analysis", args.representation_type,
                        f"sel_frac_{args.select_frac_neurons}", "metrics.csv")
    # --- CSV append with lock ---
    from filelock import FileLock
    csv_path = os.path.join(base_dir, "metrics.csv")
    lock = FileLock(csv_path + ".lock")
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sweep_id": sweep_id, "project": run.project, "run_id": run.id, "run_name": run.name,
        "frac": args.select_frac_neurons, "ds_pool": ds_pooling, "reg": args.reg_type,
        "lr": args.lr, "wd": args.weight_decay, "l1": args.l1_reg,
        # "mean_x": mean_x.tolist() if normalize else None,
        # "std_x": std_x.tolist() if normalize else None,
        # "neuron_indices": neuron_indices.tolist(),
        "best_val_loss": float(wandb.run.summary["best_val_loss"]),
        "final_val_mse": float(val_report["mse"]), 
        "final_test_mse": float(test_report["mse"]),
        "final_val_psnr": val_report["pearson_each"].tolist(),
        "final_test_psnr": test_report["pearson_each"].tolist(),
        "final_val_r2avg": float(val_report["r2_avg"]),
        "final_test_r2avg": float(test_report["r2_avg"]),
        "final_val_r2_each": val_report["r2_each"].tolist(),
        "final_test_r2_each": test_report["r2_each"].tolist(),
    }
    header = list(row.keys())
    with lock:
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerow(row)
    
    
    artifact_name = f"{run.project}_{run.id}_frac_{args.select_frac_neurons}".replace('.', '_')
    artifact = wandb.Artifact(artifact_name, type="model",
                              metadata={"base_model": wandb.run.group,
                                    "frac": args.select_frac_neurons,
                                    "ds_pooling": ds_pooling,
                                    "reg_type": args.reg_type})
    artifact.add_file(os.path.join(ckpt_dir, "best.pt"))
    wandb.log_artifact(artifact, aliases=["best", f"frac_{str(args.select_frac_neurons).replace('.','_')}"])

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/user/azhar.akhmetova/corrViT/refactor/data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--idx_folder", type=str, default="/user/azhar.akhmetova/corrViT/analysis/misc/analysis/neuron_indices")

    parser.add_argument("--representation_type", type=str, default="initial", choices=["initial", "after_sa", "after_core", "queries"])
    parser.add_argument("--select_frac_neurons", type=float, default=1.0, help="Fraction of neurons to select for decoding")
    parser.add_argument("--latents_path", type=str, required=True, help="Path to latents .npy file")
    parser.add_argument("--reg_type", type=str, default="ridge", choices=["ridge", "lasso", "elasticnet"], help="Type of regularization for linear model")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01, help="L2 via AdamW")
    parser.add_argument("--l1_reg", type=float, default=0.0, help="L1 regularization strength")
    parser.add_argument("--learned_pooling", type=str, default="none", choices=["linear", "attention", "none"], help="Pooling mode: linear/attention")
    parser.add_argument("--ds_pooling", type=str, default="mean", choices=["mean", "max", "min", "none"], help="Pooling mode: mean/max/min")
    main(parser.parse_args())

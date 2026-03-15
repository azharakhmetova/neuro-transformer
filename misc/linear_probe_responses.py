import sys
import os
import pickle
import torch
import numpy as np
import typing as t
import argparse
import numpy as np
import wandb
import yaml, csv, time

from tqdm import trange
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from v1t.utils import utils, tensorboard

utils.set_random_seed(1234)

from scipy.stats import pearsonr
from sklearn.metrics import r2_score

tensorboard.set_font()

def get_neuron_indices(resp_dir, neuron_fraction, idx_folder, neuron_seed=42):
    """Get neuron indices for nested subsampling"""
    # Try to load existing indices

    file_path = os.path.join(idx_folder, f"input_frac_{neuron_fraction:.2f}_seed_{neuron_seed}.pkl")
    try:
        with open(file_path, 'rb') as f:
            neuron_data = pickle.load(f)
            
        # Get N from first training file (for verification)
        first_arr = np.load(os.path.join(resp_dir, '0.npy'), mmap_mode='r')
        N = first_arr.shape[0]  # number of neurons
        print(f"Loaded N={N} neurons")
        
        # Verify loaded data is consistent
        if neuron_data['total_neurons'] != N:
            print(f"Warning: Loaded N={neuron_data['total_neurons']} but current data has N={N}")
        
        if neuron_fraction !=0.0:
            input_indices = neuron_data['input_neuron_indices']
            query_indices = neuron_data['query_neuron_indices']
            print(f"Loaded input/query neuron indices from {file_path}")
        else:
            input_indices = None
            query_indices = np.arange(N, dtype=np.int64)

        
        return input_indices, query_indices
        
    except (FileNotFoundError, KeyError, pickle.UnpicklingError) as e:
        print("No idx_folder provided, generating new neuron indices...")
        print(f"Could not load existing indices ({e}), generating new ones...")
        first_arr = np.load(os.path.join(resp_dir, '0.npy'), mmap_mode='r')
        N = first_arr.shape[0]  # number of neurons
        
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
        if idx_folder is not None:
            neuron_data = {
                'input_neuron_indices': input_indices,
                'query_neuron_indices': query_indices,
                'neuron_fraction': neuron_fraction,
                'neuron_seed': neuron_seed,
                'total_neurons': N,
                'n_selected': n_neurons
            }
            
            # Create filename
            filename = f"input_frac_{neuron_fraction:.2f}_seed_{neuron_seed}.pkl"
            filepath = os.path.join(idx_folder, filename)
            os.makedirs(idx_folder, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(neuron_data, f)
    return input_indices, query_indices

def compute_input_norm_stats(
    resp_dir: str,
    stats_dir: str,
    tiers: np.ndarray,     # indices in [0..T-1]
    tier: str,
    input_indices: np.ndarray,       # neuron ids (Kin,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns mean/std for input neurons only:
      mean_in, std_in shape [Kin]
    computed over train trials only.
    """
    input_indices = input_indices.astype(np.int64)
    Kin = input_indices.size

    sum_x = np.zeros((Kin,), dtype=np.float64)
    sum_x2 = np.zeros((Kin,), dtype=np.float64)
    count = 0
    train_trial_ids   = np.where(tiers == tier)[0]

    for tid in train_trial_ids:
        r = np.load(os.path.join(resp_dir, f"{int(tid)}.npy"), mmap_mode="r")  # [N]
        x = np.asarray(r[input_indices], dtype=np.float64)                    # [Kin]
        sum_x += x
        sum_x2 += x * x
        count += 1

    if count < 2:
        raise ValueError(f"Not enough train trials to compute stats (count={count})")

    mean_in = (sum_x / count).astype(np.float32)
    var_in = (sum_x2 / count - mean_in.astype(np.float64) ** 2).astype(np.float32)
    std_in = np.sqrt(np.maximum(var_in, 1e-12)).astype(np.float32)

    return mean_in, std_in
    

class ResponseDataset(Dataset):
    def __init__(
        self,
        resp_dir: str,
        tiers: np.ndarray,
        tier: str,
        input_indices: np.ndarray,
        query_indices: np.ndarray,
        normalize: bool = True,
        input_mean: np.ndarray | None = None,   # [Kin]
        input_std: np.ndarray | None = None,    # [Kin]
        latents_path: str | None = None,
    ):
        self.resp_dir = resp_dir
        self.trial_ids = np.where(tiers == tier)[0].astype(np.int64)
        self.input_indices = input_indices.astype(np.int64) 
        self.query_indices = query_indices.astype(np.int64)

        self.normalize = normalize
        self.input_mean = input_mean
        self.input_std = input_std

        if self.normalize:
            assert input_mean is not None and input_std is not None
            assert input_mean.shape[0] == self.input_indices.shape[0]
            assert input_std.shape[0] == self.input_indices.shape[0]

        self.latents_path = latents_path
        if latents_path is not None:
            self.z = self._load_latents(latents_path, tier)            # [T,k]
    
    def _load_latents(self, path_to_latents: str, tier: str) -> np.ndarray:
        z = np.load(path_to_latents, mmap_mode='r')   # [T_all, k]
        merged_root = '/user/azhar.akhmetova/u12008/neuro-transformer/misc/analysis'
        tiers = np.load(os.path.join(merged_root, 'merged_meta', 'trials', 'tiers.npy'))
        if tier == 'train':
            idx = np.nonzero(tiers == 'train')[0]
        elif tier == 'validation':
            idx = np.nonzero(tiers == 'validation')[0]
        else:
            idx = np.nonzero(tiers == 'test')[0]
        return z[idx]  # [T,k]

    def __len__(self):
        return self.trial_ids.shape[0]

    def __getitem__(self, j: int):
        tid = int(self.trial_ids[j])
        r = np.load(os.path.join(self.resp_dir, f"{tid}.npy"), mmap_mode="r")  # [N]
        r = np.asarray(r, dtype=np.float32)

        x = r[self.input_indices]  # [Kin]
        y = r[self.query_indices]  if self.latents_path is None else self.z[j] #[Kout] or [K]

        if self.normalize:
            std = np.where(self.input_std == 0, 1.0, self.input_std)
            x = (x - self.input_mean) / std

        return torch.from_numpy(x), torch.from_numpy(y)

class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
                
    def forward(self, x):
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
    pearson_mean = float(np.nanmean(pearson_each)) 

    return {
        "mse": float(mse),
        "r2_avg": float(r2_avg),
        "r2_each": r2_each,          # np.ndarray
        "pearson_each": pearson_each, # np.ndarray
        "pearson_mean": pearson_mean
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

    best_epoch = -1
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
    if not (0.0 < args.select_frac_neurons < 1.0):
        raise ValueError("select_frac_neurons must be in (0,1) so there are held-out neurons.")
    resp_dir = f'/user/azhar.akhmetova/u12008/neuro-transformer/data/sensorium/k_{args.K}_a_{args.ALPHA}_beta1e7_poisson/data/responses'
    stat_dir = f'/user/azhar.akhmetova/u12008/neuro-transformer/data/sensorium/k_{args.K}_a_{args.ALPHA}_beta1e7_poisson/meta/statistics/responses/all'
    tiers = np.load(f'/user/azhar.akhmetova/u12008/neuro-transformer/data/sensorium/k_{args.K}_a_{args.ALPHA}_beta1e7_poisson/meta/trials/tiers.npy')

    input_indices, query_indices = get_neuron_indices(resp_dir, neuron_fraction=args.select_frac_neurons, neuron_seed=42, idx_folder=args.idx_folder)

    normalize = True
    if normalize:
        mean_in, std_in = compute_input_norm_stats(resp_dir, stat_dir, tiers, tier="train", input_indices=input_indices)
    
    if args.predict == "latents":
        latents_path = f"/user/azhar.akhmetova/u12008/neuro-transformer/misc/analysis/latents/k_{args.K}/z_k_{args.K}_no_train_dupl.npy"
    else: 
        latents_path = None

    train_ds, val_ds, test_ds = {}, {}, {}
    train_loader, val_loader, test_loader = {}, {}, {}
    mouse_id = f"k_{args.K}_a_{args.ALPHA}_beta1e7_poisson" # just one mouse for now for mouse_id in args.mouse_ids:
    
    train_ds[mouse_id] = ResponseDataset(
                latents_path=latents_path,
                resp_dir=resp_dir,
                tiers=tiers,
                tier="train",
                input_indices=input_indices,
                query_indices=query_indices,
                normalize=True,
                input_mean=mean_in,   
                input_std=std_in,    
                )
    val_ds[mouse_id] = ResponseDataset(
                latents_path=latents_path,
                resp_dir=resp_dir,
                tiers=tiers,
                tier="validation",
                input_indices=input_indices,
                query_indices=query_indices,
                normalize=True,
                input_mean=mean_in,   
                input_std=std_in,    
                )
    test_ds[mouse_id] = ResponseDataset(
                latents_path=latents_path,
                resp_dir=resp_dir,
                tier="test",
                tiers=tiers,
                input_indices=input_indices,
                query_indices=query_indices,
                normalize=True,
                input_mean=mean_in,   
                input_std=std_in,    
                )
    N = train_ds[mouse_id].__getitem__(0)[0].shape[0]
    print(f"Total neurons N: {N}")
    train_loader[mouse_id] = DataLoader(train_ds[mouse_id], batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)
    val_loader[mouse_id] = DataLoader(val_ds[mouse_id], batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    test_loader[mouse_id] = DataLoader(test_ds[mouse_id], batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    
    input_dim = len(input_indices)#train_ds[mouse_id].__getitem__(0)[0].shape[-1]
    print(f"Input dim: {input_dim}")

    output_dim = len(query_indices) if args.predict == "responses" else args.K #train_ds[mouse_id].__getitem__(0)[1].shape[0]
    print(f"Output dim: {output_dim}")

    num_epochs = 100
    linear_probe = LinearRegression(input_dim, output_dim)
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
    project="responses-regression",
    entity="azharakhmetova-master-thesis",
    name=f"k_{args.K}_a_{args.ALPHA}_frac_n_{args.select_frac_neurons}_initT_reg_{args.reg_type}_lr_{args.lr:.5g}_wd_{args.weight_decay:.5g}_l1_{args.l1_reg:.5g}",
    group=f"k_{args.K}_a_{args.ALPHA}",
    tags=[f"frac={args.select_frac_neurons}", f"reg={args.reg_type}"],
    config=dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        l1_reg=args.l1_reg,
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
    if args.predict == "latents":
        sweep_dir = f"/mnt/lustre-grete/usr/u12008/responses_latents_probe/k_{args.K}_a_{args.ALPHA}_b1e7_poisson"
    if args.predict == "responses":
        sweep_dir = f"/mnt/lustre-grete/usr/u12008/responses_probe/k_{args.K}_a_{args.ALPHA}_b1e7_poisson"
    sweep_id = os.environ.get("WANDB_SWEEP_ID")  # set by wandb agent

    base_dir  = os.path.join(sweep_dir, f"sel_frac_{args.select_frac_neurons}", f"sweep-{sweep_id or 'nosweep'}")
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
    wandb.run.summary.update({
    "final_val_mse": float(val_report["mse"]),
    "final_test_mse": float(test_report["mse"]),
    "final_val_corr_mean": float(val_report["pearson_mean"]),
    "final_test_corr_mean": float(test_report["pearson_mean"]),
    })
    wandb.log({
    "final_val_corr_mean": float(val_report["pearson_mean"]),
    "final_test_corr_mean": float(test_report["pearson_mean"]),
    })
    # --- CSV append with lock ---
    from filelock import FileLock
    csv_path = os.path.join(base_dir, "metrics.csv")
    lock = FileLock(csv_path + ".lock")
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sweep_id": sweep_id, "project": run.project, "run_id": run.id, "run_name": run.name,
        "frac": args.select_frac_neurons, "reg": args.reg_type,
        "lr": args.lr, "wd": args.weight_decay, "l1": args.l1_reg,
        # "mean_x": mean_x.tolist() if normalize else None,
        # "std_x": std_x.tolist() if normalize else None,
        # "neuron_indices": neuron_indices.tolist(),
        "best_val_loss": float(wandb.run.summary["best_val_loss"]),
        "final_val_mse": float(val_report["mse"]), 
        "final_test_mse": float(test_report["mse"]),
        "final_val_corr": float(val_report["pearson_mean"]),
        "final_test_corr": float(test_report["pearson_mean"]),
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
                                    "reg_type": args.reg_type})
    artifact.add_file(os.path.join(ckpt_dir, "best.pt"))
    wandb.log_artifact(artifact, aliases=["best", f"frac_{str(args.select_frac_neurons).replace('.','_')}"])

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--output_dir", type=str, required=True)
    # to build output and latents paths:
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--ALPHA", type=str, default="06")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--idx_folder", type=str, default="/user/azhar.akhmetova/u12008/neuro-transformer/misc/analysis/neuron_indices")

    parser.add_argument("--select_frac_neurons", type=float, default=0.10, help="Fraction of neurons to select for decoding")
    parser.add_argument("--predict", type=str, default="responses", choices=["responses", "latents"], help="What to predict: responses or latents")
    parser.add_argument("--reg_type", type=str, default="ridge", choices=["ridge", "lasso", "elasticnet"], help="Type of regularization for linear model")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01, help="L2 via AdamW")
    parser.add_argument("--l1_reg", type=float, default=0.0, help="L1 regularization strength")
    main(parser.parse_args())

#
# SBATCH --array=1-5
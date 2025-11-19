# eval_decoding_latents.py
import os, argparse, numpy as np, torch, wandb
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

# ---- import your helpers from the training file (or paste them here) ----
# from decoding_latents_pytorch import (
#     ReprLatentDataset, compute_normalization_stats, get_neuron_indices,
#     LinearRegression
# )

def load_ckpt(ckpt_ref: str, entity=None, project=None, alias="best"):
    """
    ckpt_ref:
      - local: /path/to/best.pt
      - artifact: entity/project/artifact_name:alias    (e.g., "azhar/.../k4_frac0_5:best")
    """
    if os.path.exists(ckpt_ref):
        return torch.load(ckpt_ref, map_location="cpu")

    # Treat as W&B artifact
    assert entity and project, "When using a W&B artifact, provide --entity and --project"
    api = wandb.Api()
    art = api.artifact(ckpt_ref if ":" in ckpt_ref else f"{entity}/{project}/{ckpt_ref}:{alias}", type="model")
    dirpath = art.download()
    # prefer best.pt, fall back to last.pt
    best = os.path.join(dirpath, "best.pt")
    last = os.path.join(dirpath, "last.pt")
    path = best if os.path.exists(best) else last
    return torch.load(path, map_location="cpu")

@torch.no_grad()
def evaluate(dataloader, model, device):
    model.eval()
    ys, yhats = [], []
    for x, y in dataloader:
        yhat = model(x.to(device))
        ys.append(y.cpu().numpy())
        yhats.append(yhat.cpu().numpy())
    Y = np.concatenate(ys, axis=0)
    YH = np.concatenate(yhats, axis=0)
    mse_avg  = mean_squared_error(Y, YH)                         # scalar (averaged over dims)
    mse_each = mean_squared_error(Y, YH, multioutput="raw_values")
    r2_avg   = r2_score(Y, YH)
    r2_each  = r2_score(Y, YH, multioutput="raw_values")
    r_each   = np.array([pearsonr(Y[:, i], YH[:, i])[0] for i in range(Y.shape[1])])
    return dict(mse=mse_avg, mse_each=mse_each, r2=r2_avg, r2_each=r2_each, r_each=r_each)

def main():
    parser = argparse.ArgumentParser()
    # Required data bits (same as training)
    parser.add_argument("--dataset", type=str, default="/user/azhar.akhmetova/corrViT/refactor/data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--latents_path", type=str, required=True)
    parser.add_argument("--select_frac_neurons", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    # Checkpoint reference
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to best.pt OR W&B artifact like entity/project/name:best")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (if ckpt is an artifact)")
    parser.add_argument("--project", type=str, default=None, help="W&B project (if ckpt is an artifact)")
    # Optional overrides (rarely needed)
    parser.add_argument("--rep_subdir", type=str,
                        default="neuron_token_representations/k_4_a_03_xyz_b1e7_poisson/initial",
                        help="Subdir under output_dir where reps live")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load checkpoint (local or artifact) ----
    ckpt = load_ckpt(args.ckpt, entity=args.entity, project=args.project)
    state_dict = ckpt["model"]
    cfg = ckpt.get("config", {})  # stored in training: dict(wandb.config)
    # Use ds_pooling from checkpoint to ensure identical preprocessing
    ds_pooling = cfg.get("ds_pooling", "mean")
    learned_pooling = cfg.get("learned_pooling", None)
    input_dim = int(cfg.get("input_dim", 152))
    output_dim = int(cfg.get("output_dim", 4))

    # ---- Rebuild datasets exactly like training ----
    # Rep dir used in training:
    rep_dir = os.path.join(args.output_dir, args.rep_subdir)
    neuron_indices, N = get_neuron_indices(rep_dir, neuron_fraction=args.select_frac_neurons, neuron_seed=42)

    # recompute normalization from train set with the SAME pooling
    mean_x, std_x = compute_normalization_stats(rep_dir, tier="train", pooling=ds_pooling, neuron_indices=neuron_indices)

    # Build test dataset/loader
    test_ds = ReprLatentDataset(
        latents_path=args.latents_path,
        rep_dir=rep_dir,
        tier="test",
        pooling=ds_pooling,
        normalize=True,
        norm_mean=mean_x,
        norm_std=std_x,
        neuron_indices=neuron_indices,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ---- Rebuild model & load weights ----
    # learned_pooling is None for dataset-pooled features
    model = LinearRegression(input_dim=input_dim, output_dim=output_dim,
                             pooling=learned_pooling, N=N).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("State dict load note - Missing:", missing, "Unexpected:", unexpected)

    # ---- Evaluate ----
    metrics = evaluate(test_loader, model, device)
    print(f"TEST | MSE={metrics['mse']:.6f}  R2={metrics['r2']:.4f}")
    print("MSE_each:", np.round(metrics["mse_each"], 6))
    print("R2_each :", np.round(metrics["r2_each"], 4))
    print("r_each  :", np.round(metrics["r_each"], 4))

    # (Optional) log back to W&B if running inside a run
    if wandb.run is not None:
        wandb.log({
            "test/mse": metrics["mse"],
            "test/r2": metrics["r2"],
        })

if __name__ == "__main__":
    main()

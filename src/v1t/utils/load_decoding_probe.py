
import os
from pathlib import Path
import pandas as pd
import torch
import argparse

from torch.utils.data import DataLoader

from v1t.utils.decoding_latents_pytorch import LinearRegression, AttnPool



def find_sel_frac_dir(base_dir: Path, target_frac: float, tolerance: float = 1e-6) -> Path | None:
    """
    Find the sel_frac_* directory that matches the target fraction.
    
    Args:
        base_dir: Directory containing sel_frac_* subdirectories
        target_frac: The fraction we're looking for (e.g., 0.05, 1.0)
        tolerance: Floating point comparison tolerance
    
    Returns:
        Path to the matching sel_frac_* directory, or None if not found
    """
    if not base_dir.exists():
        print(f"Base directory does not exist: {base_dir}")
        return None
    
    print(f"Looking for fraction {target_frac} in {base_dir}")
    
    for sel_dir in base_dir.glob("sel_frac_*"):
        if not sel_dir.is_dir():
            continue
            
        # Extract fraction from directory name
        dir_name = sel_dir.name
        if not dir_name.startswith("sel_frac_"):
            continue
            
        frac_str = dir_name[len("sel_frac_"):]
        try:
            dir_frac = float(frac_str)
            print(f"  Found directory: {dir_name} -> fraction {dir_frac}")
            
            # Handle both exact match and floating point comparison
            if abs(dir_frac - target_frac) < tolerance:
                print(f"  ✓ Matched!")
                return sel_dir
        except ValueError:
            print(f"  Skipping {dir_name} (non-numeric)")
            continue
    
    print(f"  No match found for fraction {target_frac}")
    return None

def find_best_run_row(
    output_dir: str | Path,
    representation_type: str,
    pooling_type: str,
    csv_name: str | None = None,
    base_csv_dir: str | Path | None = None,
) -> pd.Series:
    """
    Load best_by_fraction_<pooling_type>.csv and return the row
    with minimal final_val_mse.
    """
    output_dir = Path(output_dir)
    if base_csv_dir is None:
        base_csv_dir = output_dir / representation_type
    else:
        base_csv_dir = Path(base_csv_dir)

    if csv_name is None:
        csv_name = f"best_by_fraction_{pooling_type}.csv"

    csv_path = base_csv_dir / csv_name
    print(f"Loading best runs from: {csv_path}")
    df = pd.read_csv(csv_path)

    # pick by validation MSE
    best_idx = df["final_val_mse"].astype(float).idxmin()
    return df.loc[best_idx]

def find_frac_run_row(
    output_dir: str | Path,
    representation_type: str,
    pooling_type: str,
    csv_name: str | None = None,
    base_csv_dir: str | Path | None = None,
    load_frac: float = 1.0,
) -> pd.Series:
    """
    Load best_by_fraction_<pooling_type>.csv and return the row
    with the specified fraction and best validation MSE.
    """
    output_dir = Path(output_dir)
    if base_csv_dir is None:
        base_csv_dir = output_dir / representation_type
    else:
        base_csv_dir = Path(base_csv_dir)

    if csv_name is None:
        csv_name = f"best_by_fraction_{pooling_type}.csv"

    csv_path = base_csv_dir / csv_name
    print(f"Loading runs from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Filter by the specified fraction first
    matching_rows = df.loc[df["frac"] == load_frac]
    
    if matching_rows.empty:
        raise ValueError(f"No rows found with frac={load_frac}")
    
    if len(matching_rows) > 1:
        # If multiple rows with same fraction, pick the one with best validation MSE
        print(f"Found {len(matching_rows)} rows with frac={load_frac}, selecting best by val MSE")
        best_idx = matching_rows["final_val_mse"].astype(float).idxmin()
        return df.loc[best_idx]
    else:
        # Return the single matching row as a Series
        return matching_rows.iloc[0]
    
def checkpoint_path_from_row(
    output_dir: str | Path,
    representation_type: str,
    pooling_type: str,
    row: pd.Series,
    base_csv_dir: str | Path | None = None,
) -> Path:
    """
    Reconstruct checkpoints/best.pt path from a row in best_by_fraction_*.csv.
    Expects row to contain: 'frac', 'sweep_id', 'run_id'.
    """
    output_dir = Path(output_dir)
    if base_csv_dir is None:
        base_csv_dir = output_dir / representation_type
    else:
        base_csv_dir = Path(base_csv_dir)
    
    frac = float(row["frac"])
    sweep_id = row["sweep_id"]
    run_id = row["run_id"]

    sel_dir = find_sel_frac_dir(base_csv_dir, frac)
    
    if sel_dir is None:
        # Debug: list what directories actually exist
        if base_csv_dir.exists():
            existing_dirs = [d.name for d in base_csv_dir.iterdir() if d.is_dir() and d.name.startswith("sel_frac_")]
            print(f"Available directories in {base_csv_dir}: {existing_dirs}")
        raise FileNotFoundError(f"No sel_frac_* directory found for fraction {frac} in {base_csv_dir}")
    
    pooling_dir = f"{pooling_type}"             # e.g. 'linear_p' or 'attention_p'

    ckpt_path = (
        sel_dir  # sel_dir is already the full path
        / pooling_dir
        / f"sweep-{sweep_id}"
        / f"run-{run_id}"
        / "checkpoints"
        / "best.pt"
    )
    return ckpt_path, frac


def load_best_linear_probe(
    output_dir: str | Path,
    decoding_dir: str | Path,
    representation_type: str,
    pooling_type: str,
    val_loader,              # DataLoader yielding (x, y)
    load_best_row: bool | pd.Series = False,
    load_frac: float | None = None,
    device: str = "cuda",
    csv_name: str | None = None,
    base_csv_dir: str | Path | None = None,
    ckpt_path: str | Path | None = None,
    n_queries: int = 1,
) -> torch.nn.Module:
    """
    Convenience wrapper:
      - find best row for pooling_type
      - build checkpoint path
      - instantiate LinearRegression with correct shapes
      - load weights and return model in eval() mode
    """

    if load_best_row:
        best_row = find_best_run_row(
            output_dir=decoding_dir,
            representation_type=representation_type,
            pooling_type=pooling_type,
            csv_name=csv_name,
            base_csv_dir=base_csv_dir,
        )
        ckpt_path, frac = checkpoint_path_from_row(decoding_dir, representation_type, pooling_type, best_row, base_csv_dir)
    elif load_frac is not None:
        row = find_frac_run_row(
                    output_dir=decoding_dir,
                    representation_type=representation_type,
                    pooling_type=pooling_type,
                    csv_name=csv_name,
                    base_csv_dir=base_csv_dir,
                    load_frac=load_frac,
                )
        ckpt_path, frac = checkpoint_path_from_row(decoding_dir, representation_type, pooling_type, row, base_csv_dir)
    print(f"Loading checkpoint from: {ckpt_path}")

    # Peek at one batch to infer shapes
    x_sample, y_sample = next(iter(val_loader))
    x_sample = x_sample.to(device)
    y_sample = y_sample.to(device)

    pooling_type = pooling_type.replace("_p", "") if pooling_type.endswith("_p") else pooling_type
    if pooling_type in {"linear", "attention"}:
        # x: [B, N, d]
        N = x_sample.shape[1]
        input_dim = x_sample.shape[-1]
    else:
        # already pooled, x: [B, d]
        N = None
        input_dim = x_sample.shape[-1]

    output_dim = y_sample.shape[-1]

    model = LinearRegression(
        input_dim=input_dim,
        output_dim=output_dim,
        pooling=pooling_type if pooling_type in {"linear", "attention"} else None,
        N=N,
        n_queries=n_queries,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model

def load_best_linear_probe_(
    decoding_dir: str | Path,
    representation_type: str,
    pooling_type: str,
    K: int,
    load_best_row: bool | pd.Series = False,
    load_frac: float | None = None,
    device: str = "cuda",
    csv_name: str | None = None,
    base_csv_dir: str | Path | None = None,
    ckpt_path: str | Path | None = None,
    n_queries: int = 1,
) -> torch.nn.Module:
    """
    Convenience wrapper:
      - find best row for pooling_type
      - build checkpoint path
      - instantiate LinearRegression with correct shapes
      - load weights and return model in eval() mode
    """

    if load_best_row:
        best_row = find_best_run_row(
            output_dir=decoding_dir,
            representation_type=representation_type,
            pooling_type=pooling_type,
            csv_name=csv_name,
            base_csv_dir=base_csv_dir,
        )
        ckpt_path, frac = checkpoint_path_from_row(decoding_dir, representation_type, pooling_type, best_row, base_csv_dir)
    elif load_frac is not None:
        row = find_frac_run_row(
                    output_dir=decoding_dir,
                    representation_type=representation_type,
                    pooling_type=pooling_type,
                    csv_name=csv_name,
                    base_csv_dir=base_csv_dir,
                    load_frac=load_frac,
                )
        ckpt_path, frac = checkpoint_path_from_row(decoding_dir, representation_type, pooling_type, row, base_csv_dir)
    print(f"Loading checkpoint from: {ckpt_path}")

    pooling_type = pooling_type.replace("_p", "") if pooling_type.endswith("_p") else pooling_type
    if pooling_type in {"linear", "attention"}:
        # x: [B, N, d]
        N = 7334*(1-load_frac)
        input_dim = 160
    else:
        # already pooled, x: [B, d]
        N = None
        input_dim = 160

    output_dim = K

    model = LinearRegression(
        input_dim=input_dim,
        output_dim=output_dim,
        pooling=pooling_type if pooling_type in {"linear", "attention"} else None,
        N=N,
        n_queries=n_queries,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model
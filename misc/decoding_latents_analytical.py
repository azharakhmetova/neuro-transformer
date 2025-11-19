import sys
import os
import pickle
import matplotlib
import torch
import numpy as np
import typing as t
import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import center_of_mass

from v1t.utils import utils, tensorboard
from v1t import data
from v1t.models.model import Model
from v1t.utils.scheduler import Scheduler
from v1t.utils.neuron_representations import sample_neuron_tokens
from v1t.misc.decoding_latents_pytorch import get_neuron_indices

from torch.utils.data import DataLoader

utils.set_random_seed(1234)

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

tensorboard.set_font()

# def get_neuron_indices(rep_dir, neuron_fraction=1.0, neuron_seed=42):
#     """Get consistent neuron indices for nested subsampling"""
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


def load_all_data_with_subsampling(rep_dir, 
                                   latents_path, 
                                   pooling="mean", 
                                   neuron_indices=None):
    """Load data with consistent neuron subsampling across splits"""

    print("Loading data with subsampling...")
    # Get neuron indices from first training file
    train_dir = rep_dir + '/train'
    first_file = sorted([f for f in os.listdir(train_dir) if f.endswith('.npy')], 
                       key=lambda x: int(x.split('.')[0]))[0]
    first_arr = np.load(os.path.join(train_dir, first_file), mmap_mode='r')
    N = first_arr.shape[1]
    
    def load_split_data(split_dir, neuron_indices, pooling):
        split_reps = []
        fnames = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
        fnames = sorted(fnames, key=lambda x: int(x.split('.')[0]))
        
        for fname in fnames:
            arr = np.load(os.path.join(split_dir, fname))  # [B, N, 152]
            arr_subset = arr[:, neuron_indices, :]  # [B, n_selected, 152]
            
            if pooling == 'mean':
                pooled = arr_subset.mean(axis=1)  # [B, 152]
            elif pooling == 'max':
                pooled = arr_subset.max(axis=1)
            elif pooling == 'min':
                pooled = arr_subset.min(axis=1)
            
            split_reps.append(pooled)
        
        return np.concatenate(split_reps, axis=0)
    
    # Load all splits
    X_train = load_split_data(train_dir, neuron_indices, pooling)
    X_val = load_split_data(rep_dir + '/validation', neuron_indices, pooling)
    X_test = load_split_data(rep_dir + '/test', neuron_indices, pooling)
    
    # Load latents (unchanged)
    latents = np.load(latents_path)
    merged_root = '/user/azhar.akhmetova/merged_sensorium'
    tiers = np.load(os.path.join(merged_root, 'meta', 'trials', 'tiers.npy'))
    
    train_idx = np.where(tiers == 'train')[0]
    val_idx = np.where(tiers == 'validation')[0]
    test_idx = np.where(tiers == 'test')[0]
    
    y_train = latents[train_idx]
    y_val = latents[val_idx] 
    y_test = latents[test_idx]
    
    print(f"Final shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def run_cv(X_train, y_train, X_val, X_test, alphas=None, regularization="ridge"):
    # Use cross-validated Ridge with wide range of alphas
    if alphas is None:
        alphas = np.logspace(-3, 6, 50)  # Try regularization from 1e-3 to 1e6
    if regularization == "ridge":
        model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=alphas, cv=5)
        )
    elif regularization == "lasso":
        model = make_pipeline(
            StandardScaler(),
            LassoCV(alphas=alphas, cv=5)
        )
    elif regularization == "elasticnet":
        model = make_pipeline(
            StandardScaler(),
            ElasticNetCV(alphas=alphas, cv=5)
        )

    print("Training with cross-validation...")
    model.fit(X_train, y_train)

    # Check what alpha was selected
    print(f"Selected alpha: {model.named_steps['ridgecv'].alpha_}")

    # Evaluate
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    return y_val_pred, y_test_pred

def report(split, y_val, y_val_pred):
    r2_all  = r2_score(y_val, y_val_pred, multioutput='uniform_average')
    r2_each = r2_score(y_val, y_val_pred, multioutput='raw_values')
    mse_all = mean_squared_error(y_val, y_val_pred)                     # average over all dims
    mse_each= mean_squared_error(y_val, y_val_pred, multioutput='raw_values')  # per dim
    r_each  = np.array([pearsonr(y_val[:, i], y_val_pred[:, i])[0] for i in range(y_val.shape[1])])
    print(
        f"{split}: "
        f"R2_avg={r2_all:.3f}  R2_each={np.round(r2_each,3)}  "
        f"MSE_avg={mse_all:.4f}  MSE_each={np.round(mse_each,4)}  "
        f"r_each={np.round(r_each,3)}"
    )
    print(f"R2: {r2_score(y_val, y_val_pred):.3f}")


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

    rep_dir = args.output_dir  + "/neuron_token_representations/k_4_a_03_xyz_b1e7_poisson/initial"
    latents_path = args.latents_path
    neuron_indices, N = get_neuron_indices(rep_dir, neuron_fraction=1.0, neuron_seed=42)
    pooling = 'max'
    X_train, X_val, X_test, y_train, y_val, y_test = load_all_data_with_subsampling(rep_dir, latents_path, neuron_indices=neuron_indices, pooling=pooling)
    y_val_pred, y_test_pred = run_cv(X_train, y_train, X_val, X_test, regularization=args.reg_type)

    report("VAL", y_val, y_val_pred)
    report("TEST", y_test, y_test_pred)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/user/azhar.akhmetova/corrViT/refactor/data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--latents_path", type=str, required=True, help="Path to latents .npy file")
    parser.add_argument("--reg_type", type=str, default="ridge", choices=["ridge", "lasso", "elasticnet"], help="Type of regularization for linear model")
    main(parser.parse_args())

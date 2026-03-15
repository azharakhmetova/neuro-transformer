# #!/usr/bin/env python3
# """
# Neural Transformer Model Evaluation Script
# Run this on the compute node to evaluate model performance across different input neuron fractions
# """

# import sys
# import os
# import pickle
# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend for server
# import numpy as np
# import torch
# from tqdm import tqdm
# import typing as t
# import argparse
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# from scipy.ndimage import center_of_mass
# import socket
# import gc

# from v1t.utils import utils, tensorboard
# from v1t import losses, data
# from v1t.models.model import Model
# from v1t.utils.scheduler import Scheduler

# from torch.utils.data import DataLoader
# from torch.amp import autocast, GradScaler

# utils.set_random_seed(1234)

# def main():
#     print("=== Neural Transformer Evaluation ===")
#     print(f"Hostname: {socket.gethostname()}")
#     print(f"CUDA available: {torch.cuda.is_available()}")
    
#     if torch.cuda.is_available():
#         print(f"GPU count: {torch.cuda.device_count()}")
#         for i in range(torch.cuda.device_count()):
#             print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
#             print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
#     else:
#         print("⚠️  No CUDA GPUs available - running on CPU")
    
#     def load_model(k, alpha_cap, frac):
#         # Setup arguments
#         out_dir = f"/user/azhar.akhmetova/u12008/neuro-transformer/runs_debug/analysis_bs16/k_{k}/a_{alpha_cap}/{k}_4_a_0_xyz_nosubsel_1dLL_lrx04_meanbias_2dpe_bef_core_bs_16_frac0_{frac}_s_1"
        
#         parser = argparse.ArgumentParser()
#         parser.add_argument("--dataset", type=str, default="/user/azhar.akhmetova/corrViT/refactor/data/sensorium")
#         parser.add_argument("--output_dir", type=str, required=True)
#         parser.add_argument("--batch_size", type=int, default=10)
#         parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

#         args = parser.parse_args([
#             "--output_dir", f"{out_dir}",
#         ])
        
#         print(f"Using device: {args.device}")
#         print(f"Output directory: {args.output_dir}")
        
#         if not os.path.isdir(args.output_dir):
#             raise FileNotFoundError(f"Cannot find {args.output_dir}.")

#         utils.get_device(args)
#         utils.set_random_seed(1234)
#         utils.load_args(args)

#         # Load data
#         print("Loading datasets...")
#         train_ds, val_ds, test_ds = data.get_training_ds(
#             args,
#             data_dir=args.dataset,
#             mouse_ids=args.mouse_ids,
#             batch_size=args.batch_size,
#             device=args.device,
#         )

#         # Load model
#         print("Loading model...")
#         model = Model(args, ds=val_ds)
#         model.eval()

#         scheduler = Scheduler(args, model=model, save_optimizer=False)
#         scheduler.restore(force=True)

#         scaler = GradScaler(enabled=args.amp)
#         print("Model loaded successfully!")
#         return model, val_ds
#     # Define evaluation function
#     @torch.no_grad()
#     def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
#         """Metrics to compute as part of training and validation step"""
#         msse = losses.msse(y_true=y_true, y_pred=y_pred)
#         poisson_loss = losses.poisson_loss(y_true=y_true, y_pred=y_pred)
#         correlation = losses.correlation(y1=y_pred, y2=y_true, dim=0)
#         correlation = torch.mean(correlation)
#         bits_per_spike = losses.bits_per_spike(y_true=y_true, y_pred=y_pred)
#         return {
#             "metrics/msse": msse,
#             "metrics/poisson_loss": poisson_loss,
#             "metrics/single_trial_correlation": correlation,
#             "metrics/bits_per_spike": bits_per_spike,
#         }

#     def evaluate_fraction(model, val_ds, device, k, alpha_cap, frac, max_iterations=25):
#         """Evaluate model for a specific fraction of input neurons"""
#         print(f"\n{'='*50}")
#         print(f"EVALUATING FRACTION: {frac}")
#         print(f"{'='*50}")
        
#         # Clear memory
#         torch.cuda.empty_cache()
#         gc.collect()

        
#         model, val_ds = load_model(k, alpha_cap, frac)
#         # Set fraction
#         fraction = model.frac_input_neurons  
#         model.to(device).eval()
        
#         results = {}
        
#         for mouse_id, ds in val_ds.items():
#             print(f"\nProcessing mouse: {mouse_id}")
#             N = ds.dataset.num_neurons
#             print(f"N neurons: {N}")
            
#             if fraction == 0.0:
#                 # Single evaluation for fraction 0
#                 targets, predictions = [], []
                
#                 for batch in tqdm(ds, desc=f"Mouse {mouse_id}"):
#                     with autocast(device_type=device.type, enabled=scaler.is_enabled(), dtype=torch.float16):
#                         y_true = batch["response"].to(device)
#                         y_pred, _, _ = model(
#                             images=batch["image"].to(device),
#                             responses=batch["response"].to(device),
#                             neuron_coords=batch["neuron_coordinates"].to(device),
#                             input_neuron_ids=None,
#                             query_neuron_ids=torch.arange(N).to(device),
#                             mouse_id=mouse_id,
#                             behaviors=batch["behavior"].to(device),
#                             pupil_centers=batch["pupil_center"].to(device),
#                         )
#                         targets.append(y_true.cpu())
#                         predictions.append(y_pred.cpu())
                        
#                         del y_true, y_pred
                        
#                 targets = torch.vstack(targets)
#                 predictions = torch.vstack(predictions)
#                 results[mouse_id] = compute_metrics(y_true=targets, y_pred=predictions)
                
#                 del targets, predictions
                
#             else:
#                 # Multiple iterations for non-zero fractions
#                 all_results = []
                
#                 for i in range(max_iterations):
#                     print(f"Iteration {i+1}/{max_iterations}")
                    
#                     perm = torch.randperm(N, device=device)
#                     K = int(fraction * N)
#                     input_neuron_ids = perm[:K]
#                     query_neuron_ids = perm[K:]
                    
#                     targets, predictions = [], []
                    
#                     for batch in tqdm(ds, desc=f"Iter {i+1}", leave=False):
#                         with autocast(device_type=device.type, enabled=scaler.is_enabled(), dtype=torch.float16):
#                             y_true = batch["response"].to(device)[:, query_neuron_ids]
#                             y_pred, _, _ = model(
#                                 images=batch["image"].to(device),
#                                 responses=batch["response"].to(device),
#                                 neuron_coords=batch["neuron_coordinates"].to(device),
#                                 input_neuron_ids=input_neuron_ids,
#                                 query_neuron_ids=query_neuron_ids,
#                                 mouse_id=mouse_id,
#                                 behaviors=batch["behavior"].to(device),
#                                 pupil_centers=batch["pupil_center"].to(device),
#                             )
#                             targets.append(y_true.cpu())
#                             predictions.append(y_pred.cpu())
                            
#                             del y_true, y_pred
                    
#                     targets = torch.vstack(targets)
#                     predictions = torch.vstack(predictions)
#                     iteration_results = compute_metrics(y_true=targets, y_pred=predictions)
                    
#                     # Store scalar results
#                     scalar_results = {}
#                     for key, value in iteration_results.items():
#                         scalar_results[key] = value.item() if hasattr(value, 'item') else value
#                     all_results.append(scalar_results)
                    
#                     del targets, predictions, iteration_results
#                     del input_neuron_ids, query_neuron_ids, perm
                
#                 # Aggregate results
#                 mouse_results = {}
#                 for key in all_results[0].keys():
#                     values = [result[key] for result in all_results]
#                     mouse_results[f"{key}_mean"] = np.mean(values)
#                     mouse_results[f"{key}_std"] = np.std(values)
#                     mouse_results[f"{key}_all"] = values
            
#                 results[mouse_id] = mouse_results
        
#         torch.cuda.empty_cache()
#         gc.collect()
#         print(f"Memory usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
#         return results

#     # Run evaluations for different fractions
#     fractions_to_test = ['00' 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # Start with fewer fractions
#     all_results = {}
    
#     for fraction in fractions_to_test:
#         try:
#             results = evaluate_fraction(model, val_ds, args.device, fraction)
#             all_results[f"fraction_{fraction}"] = results
#             print(f"✓ Completed fraction {fraction}")
            
#             # Save intermediate results
#             with open(f"results_fraction_{fraction}.pkl", "wb") as f:
#                 pickle.dump(results, f)
                
#         except Exception as e:
#             print(f"✗ Error for fraction {fraction}: {e}")
#             continue
    
#     # Save all results
#     with open("all_evaluation_results.pkl", "wb") as f:
#         pickle.dump(all_results, f)
    
#     print("\n" + "="*50)
#     print("EVALUATION COMPLETE!")
#     print("Results saved to all_evaluation_results.pkl")
#     print("="*50)

# if __name__ == "__main__":
#     main()
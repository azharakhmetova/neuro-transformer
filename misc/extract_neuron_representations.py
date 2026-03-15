import sys
import os
import pickle
import torch
import typing as t
import argparse
from v1t.utils import utils
from v1t import data
from v1t.models.model import Model
from v1t.utils.scheduler import Scheduler
from v1t.utils.neuron_representations import sample_neuron_tokens

from torch.utils.data import DataLoader

utils.set_random_seed(1234)

# Extract the part of path that comes after any 'seed_X/' pattern
def extract_after_seed(path):
    import re
    # Look for pattern like seed_1/, seed_2/, seed_42/, etc.
    match = re.search(r'seed_\d+/(.+)', path)
    if match:
        return match.group(1)
    else:
        return os.path.basename(path)

def extract(
        ds: t.Dict[str, DataLoader], 
        model: Model,
        device: torch.device = "cpu", 
        num_samples: int = None,
        store_dir: str = None,
        run_name: str = None,
        tier: str = None,
        frac: float = None,
        input_indices: t.Optional[torch.Tensor] = None,
        query_indices: t.Optional[torch.Tensor] = None,
        repr_type: str = "initial",
    ):

    for mouse_id, mouse_ds in ds.items():
        # if mouse_id in ("S0", "S1"):
        #     continue
        print(mouse_id)
        # if mouse_id == "k_4_a_03_b1e7_poisson":
        if repr_type == "initial":
            out_dir = os.path.join(store_dir, mouse_id, run_name, repr_type, tier)
        elif repr_type == "after_sa":
            out_dir = os.path.join(store_dir, mouse_id, run_name, repr_type, f"sel_frac_{frac}", tier)
        else:
            out_dir = os.path.join(store_dir, mouse_id, run_name, repr_type, f"sel_frac_{frac}", tier)
        os.makedirs(out_dir, exist_ok=True)
        with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
            sample_neuron_tokens(
                ds=mouse_ds, model=model, device=device, num_samples=num_samples, out_dir=out_dir, input_indices=input_indices, query_indices=query_indices, repr_type=repr_type
            )

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
        print(f"Extracting all initial neuron representations for the model trained with {args.frac_input_neurons} input neurons.")
        args.fract_input_neurons = args.select_frac_neurons
    elif args.representation_type == "after_sa":
        print(f"Extracting all neuron representations after SA for the model trained with {args.frac_input_neurons} input neurons.")
        args.fract_input_neurons = args.select_frac_neurons
    elif args.representation_type == "after_core":
        if args.select_frac_neurons == 0.0:
            model.frac_input_neurons = args.select_frac_neurons
        print(f"Extracting intermediate representations of {args.select_frac_neurons} input neurons for the model trained with {args.frac_input_neurons} input neurons.")
    elif args.representation_type == "queries":
        if args.select_frac_neurons == 0.0:
            model.frac_input_neurons = args.select_frac_neurons
        print(f"Extracting query representations of {1-args.select_frac_neurons} query neurons for the model trained with {args.frac_input_neurons} input neurons.")

    # does not work for many mouse_ids yet
    for mouse_id in args.mouse_ids:
        num_samples = args.num_trials if args.num_trials is not None else None
        print("num_samples:", num_samples)
        if args.select_frac_neurons > 0.0:
            indices_dir = args.indices_dir
            file_path = os.path.join(indices_dir, f"input_frac_{args.select_frac_neurons:.2f}_seed_42.pkl")
            try:
                with open(file_path, 'rb') as f:
                    neuron_data = pickle.load(f)
                input_indices = torch.tensor(neuron_data['input_neuron_indices'], dtype=torch.long)
                query_indices = torch.tensor(neuron_data['query_neuron_indices'], dtype=torch.long)
                print(f"Loaded input/query neuron indices from {file_path}")
                print("input_indices:", input_indices)
                print("query_indices:", query_indices)
            except FileNotFoundError:
                input_indices = None
                query_indices = None
                print(f"No neuron indices file found at {file_path}.")
        else:
            input_indices = None
            query_indices = torch.arange(args.num_output_neurons[mouse_id][0], device=args.device)
        run_name = extract_after_seed(args.output_dir)
        print(f"Model component: {run_name}")
        print(f"Extract neuron token representations from train set.")
        extract(ds=train_ds, model=model, device=args.device, num_samples=num_samples, store_dir=args.store_dir, run_name=run_name, tier = "train", frac=args.select_frac_neurons, input_indices=input_indices, query_indices=query_indices, repr_type=args.representation_type)
        print(f"Extract neuron token representations from validation set.")
        extract(ds=val_ds, model=model, device=args.device, num_samples=num_samples, store_dir=args.store_dir, run_name=run_name, tier = "validation", frac=args.select_frac_neurons, input_indices=input_indices, query_indices=query_indices, repr_type=args.representation_type)
        print(f"\nExtract neuron token representations from test set.")
        extract(ds=test_ds, model=model, device=args.device, num_samples=num_samples, store_dir=args.store_dir, run_name=run_name, tier = "test", frac=args.select_frac_neurons, input_indices=input_indices, query_indices=query_indices, repr_type=args.representation_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/user/azhar.akhmetova/u12008/neuro-transformer/data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--store_dir", type=str, default="/mnt/lustre-grete/usr/u12008/neuron_representations")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    # parser.add_argument("--frac_input_neurons", type=float, default=0.)
    parser.add_argument("--no_shuffle", action="store_true", help="do not shuffle the training data.")
    parser.add_argument("--num_trials", type=int, default=None, help="number of trials to extract representations for.")
    parser.add_argument("--representation_type", type=str, default="initial", help="type of neuron representation to extract: initial, after_sa, after_core, queries")
    parser.add_argument("--select_frac_neurons", type=float, default=1.0, help="fraction of neurons to select for representation extraction.")
    parser.add_argument("--indices_dir", type=str, default="/user/azhar.akhmetova/u12008/neuro-transformer/misc/analysis/neuron_indices")
    # args = parser.parse_args([
    #     "--frac_input_neurons", "1.",
    #     "--output_dir", "/user/azhar.akhmetova/corrViT/runs_debug/stabilized-training/k_4_no_subselect/k_4_a_03_nosubsel_bits_regfeatures_lrx04_noinit_LLbias_meanbias_s_1_frac0_1_2dpe_bef_core",
    # ])

    main(parser.parse_args())

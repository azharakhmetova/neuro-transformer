import os
import torch
import pickle
import argparse
import typing as t
import numpy as np

from v1t import data
from v1t.utils import utils
from v1t.models.model import Model
from v1t.utils.scheduler import Scheduler
from v1t.utils.attention_rollout import extract_attention_maps

from torch.utils.data import DataLoader

from sklearn.manifold import TSNE as skTSNE
from openTSNE import TSNE

def perform_tsne(features):
    print(features.shape)
    assert len(features.shape) == 2, "Incorrect feature shape"
    
    n_neurons = features.shape[0]
    
    perplexity = n_neurons / 100
    learning_rate = n_neurons / 12
    
    tsne = skTSNE(
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=42
    )
    
    tsne_result = tsne.fit_transform(features)
    return tsne_result

def perform_opentsne(features):
    print(features.shape)
    assert len(features.shape) == 2, "Incorrect feature shape"
    
    n_neurons = features.shape[0]
    
    perplexity = n_neurons / 100
    learning_rate = n_neurons / 12
    
    tsne = TSNE(
        perplexity=perplexity,
        learning_rate=learning_rate,
        initialization="pca",
        n_jobs=8,
        random_state=42, 
        verbose=True,  # Set verbose to True for detailed output
    )
    
    tsne_result = tsne.fit(features)
    return tsne_result

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="/user/azhar.akhmetova/corrViT/refactor/data/sensorium")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--device", type=str, default="cpu")

args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    raise FileNotFoundError(f"Cannot find {args.output_dir}.")

utils.get_device(args)
utils.set_random_seed(1234)

utils.load_args(args)

_, val_ds, test_ds = data.get_training_ds(
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

id_embeddings = model.neuron_id_tokenizer[args.mouse_ids[0]].embedding.weight.data 
try:
    features = model.readouts[args.mouse_ids[0]].features.weight.data
except AttributeError:
    use_features = False
    print("Model has no readout features.")

os.makedirs(os.path.join(args.output_dir, "analysis"), exist_ok=True)
opentsne_id_emb = perform_opentsne(id_embeddings)
np.save(os.path.join(args.output_dir, "analysis", "opentsne_id_emb.npy"), opentsne_id_emb)
if use_features:
    opentsne_features = perform_opentsne(features)
    np.save(os.path.join(args.output_dir, "analysis", "opentsne_features.npy"), opentsne_features)
import pandas as pd
import torch
import os

from data.go_preprocessing import construct_go_bp_layers
from model.Decoder import DenseBIDecoder, SparseBIDecoder
from model.Encoder import DenseBIEncoder, SparseBIEncoder


def make_layers(merge_conditions, dataset_name, print_go=True):
    print("\n----- START: GO preprocessing -----")
    data = pd.read_csv(f"../../GO_TCGA/{dataset_name}.csv.gz", usecols=["gene id"],
                       compression="gzip").sort_values("gene id")
    genes = list(data["gene id"])
    layers = construct_go_bp_layers(genes, merge_conditions, print_go=print_go)
    print("----- COMPLETED: GO preprocessing -----")
    return layers


def save_layers(layers, merge_conditions, dataset_name):
    layer_copy = [torch.zeros(len(layer)) for layer in layers]
    os.makedirs(f"../masks/layers/{str(merge_conditions)}", exist_ok=True)
    torch.save(layer_copy, f"../masks/layers/{str(merge_conditions)}/{dataset_name}_layers.pt")
    print("----- Saved layers to file -----")


def save_masks(layers, merge_conditions, dataset_name, dtype, model_type="sparse"):
    print(f"\n----- START: Generate {model_type} masks -----")
    if model_type == "sparse":
        encoder = SparseBIEncoder(layers, torch.nn.ReLU(), dtype)
        decoder = SparseBIDecoder(layers, torch.nn.ReLU(), dtype)
    else:
        encoder = DenseBIEncoder(layers, torch.nn.ReLU(), dtype)
        decoder = DenseBIDecoder(layers, torch.nn.ReLU(), dtype)
    print(f"----- COMPLETED: Generate {model_type} masks -----")
    os.makedirs(f"../masks/encoder/{merge_conditions}", exist_ok=True)
    torch.save(encoder.edge_masks, f"../masks/encoder/{merge_conditions}/{dataset_name}_{model_type}_edge_masks.pt")
    torch.save(encoder.proxy_masks, f"../masks/encoder/{merge_conditions}/{dataset_name}_{model_type}_proxy_masks.pt")
    os.makedirs(f"../masks/decoder/{merge_conditions}", exist_ok=True)
    torch.save(decoder.edge_masks, f"../masks/decoder/{merge_conditions}/{dataset_name}_{model_type}_edge_masks.pt")
    torch.save(decoder.proxy_masks, f"../masks/decoder/{merge_conditions}/{dataset_name}_{model_type}_proxy_masks.pt")
    print(f"----- Saved {model_type} masks to file -----")


def load_masks(module, merge_conditions, dataset_name, model_type):
    masks = []
    if (module == "encoder") or (module == "both"):
        masks.append(torch.load(f"../masks/encoder/{str(merge_conditions)}/{dataset_name}_{model_type}_edge_masks.pt",
                                weights_only=True))
        masks.append(torch.load(f"../masks/encoder/{str(merge_conditions)}/{dataset_name}_{model_type}_proxy_masks.pt",
                                weights_only=True))
    if (module == "decoder") or (module == "both"):
        masks.append(torch.load(f"../masks/decoder/{str(merge_conditions)}/{dataset_name}_{model_type}_edge_masks.pt",
                                weights_only=True))
        masks.append(torch.load(f"../masks/decoder/{str(merge_conditions)}/{dataset_name}_{model_type}_proxy_masks.pt",
                                weights_only=True))
    if len(masks) == 0:
        return None
    return masks


if __name__ == "__main__":
    merge_conditions = (1, 100)
    dataset_name = "GE_top1k_bp"
    dtype = torch.float64

    layers = make_layers(merge_conditions, dataset_name)
    save_layers(layers, merge_conditions, dataset_name)
    save_masks(layers, merge_conditions, dataset_name, dtype, model_type="sparse")
    save_masks(layers, merge_conditions, dataset_name, dtype, model_type="dense")

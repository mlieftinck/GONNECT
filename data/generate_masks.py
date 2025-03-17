import pandas as pd
import torch

from data.go_preprocessing import construct_go_bp_layers
from model.deprecated.OldDecoder import SparseDecoder, BIDecoder
from model.deprecated.OldEncoder import SparseEncoder, BIEncoder


def make_layers(merge_conditions, dataset_name):
    print("\n----- START: GO preprocessing -----")
    data = pd.read_csv(f"../../GO_TCGA/{dataset_name}.csv.gz", usecols=["gene id"],
                       compression="gzip").sort_values("gene id")
    genes = list(data["gene id"])
    layers = construct_go_bp_layers(genes, merge_conditions, print=True)
    print("----- COMPLETED: GO preprocessing -----")
    return layers


def save_layers(layers, merge_conditions, dataset_name):
    layer_copy = [torch.zeros(len(layer)) for layer in layers]
    torch.save(layer_copy, f"../masks/layers/{str(merge_conditions)}/{dataset_name}_layers.pt")
    print("----- Saved layers to file -----")


def save_masks(layers, merge_conditions, dataset_name, dtype, folder="encoder", model_type="sparse"):
    print(f"\n----- START: Generate {model_type} masks -----")
    if model_type == "sparse":
        encoder = SparseEncoder(layers, dtype)
        decoder = SparseDecoder(layers, dtype)
    else:
        encoder = BIEncoder(layers, dtype)
        decoder = BIDecoder(layers, dtype)
    print(f"----- COMPLETED: Generate {model_type} masks -----")
    torch.save(encoder.edge_masks, f"../masks/{folder}/{merge_conditions}/{dataset_name}_{model_type}_edge_masks.pt")
    torch.save(encoder.proxy_masks, f"../masks/{folder}/{merge_conditions}/{dataset_name}_{model_type}_proxy_masks.pt")
    torch.save(decoder.edge_masks, f"../masks/{folder}/{merge_conditions}/{dataset_name}_{model_type}_edge_masks.pt")
    torch.save(decoder.proxy_masks, f"../masks/{folder}/{merge_conditions}/{dataset_name}_{model_type}_proxy_masks.pt")
    print(f"----- Saved {model_type} masks to file -----")


if __name__ == "__main__":
    merge_conditions = (1, 100)
    dataset_name = "GE_top1k_bp"
    dtype = torch.float64

    layers = make_layers(merge_conditions, dataset_name)
    save_layers(layers, merge_conditions, dataset_name)
    save_masks(layers, merge_conditions, dataset_name, dtype, folder="encoder", model_type="sparse")
    save_masks(layers, merge_conditions, dataset_name, dtype, folder="decoder", model_type="sparse")
    save_masks(layers, merge_conditions, dataset_name, dtype, folder="encoder", model_type="dense")
    save_masks(layers, merge_conditions, dataset_name, dtype, folder="decoder", model_type="dense")

import torch

from data.dag_analysis import print_layers
from data.generate_masks import load_masks
from data.go_preprocessing import construct_go_bp_layers
from model.Autoencoder import Autoencoder
from model.Decoder import SparseBIDecoder, DenseBIDecoder, Decoder
from model.Encoder import SparseBIEncoder, DenseBIEncoder, Encoder


def build_model(model_type: str, biologically_informed: str, soft_links: bool, dataset_name: str,
                go_preprocessing: bool, merge_conditions, n_go_layers_used: int, activation_fn, dtype, genes=None):
    # GO processing
    if go_preprocessing:
        if not genes:
            raise Exception("Cannot perform GO preprocessing without gene list")

        print("\n----- START: GO preprocessing -----")
        go_layers = construct_go_bp_layers(genes, merge_conditions, print_go=True)
        masks = None
        print("----- COMPLETED: GO preprocessing -----")

    else:
        go_layers = torch.load(f"../masks/layers/{str(merge_conditions)}/{dataset_name}_layers.pt", weights_only=True)
        masks = load_masks(biologically_informed, merge_conditions, dataset_name, model_type)
        print("\n----- COMPLETED: Loading GO from file -----")

    # Model construction
    used_go_layers = go_layers[-min(n_go_layers_used, len(go_layers)):]
    print("Layers used in model:")
    print_layers(used_go_layers, show_genes=go_preprocessing)
    if (biologically_informed == "encoder") or (biologically_informed == "both"):
        if masks:
            # Discard masks of unused GO layers
            module_masks = (masks.pop(0)[:min(n_go_layers_used - 1, len(go_layers))],
                            masks.pop(0)[:min(n_go_layers_used - 1, len(go_layers))])
        else:
            module_masks = None
        if model_type == "sparse":
            encoder = SparseBIEncoder(used_go_layers, activation_fn, dtype, module_masks)
        else:
            encoder = DenseBIEncoder(used_go_layers, activation_fn, dtype, masks=module_masks, soft_links=soft_links)
    else:
        encoder = Encoder(used_go_layers, activation_fn, dtype)

    if (biologically_informed == "decoder") or (biologically_informed == "both"):
        if masks:
            # Discard masks of unused GO layers
            module_masks = (masks.pop(0)[-min(n_go_layers_used - 1, len(go_layers)):],
                            masks.pop(0)[-min(n_go_layers_used - 1, len(go_layers)):])
        else:
            module_masks = None
        if model_type == "sparse":
            decoder = SparseBIDecoder(used_go_layers, activation_fn, dtype, masks=module_masks)
        else:
            decoder = DenseBIDecoder(used_go_layers, activation_fn, dtype, masks=module_masks, soft_links=soft_links)
    else:
        decoder = Decoder(used_go_layers, activation_fn, dtype)

    return Autoencoder(encoder, decoder)

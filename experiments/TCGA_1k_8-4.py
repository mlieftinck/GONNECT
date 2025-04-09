import torch
import pandas as pd
from matplotlib import pyplot as plt
from data.dag_analysis import print_layers
from data.generate_masks import load_masks
from data.go_preprocessing import construct_go_bp_layers
from model.Autoencoder import Autoencoder
from model.Decoder import SparseBIDecoder, DenseBIDecoder, Decoder
from model.Encoder import SparseBIEncoder, Encoder, DenseBIEncoder
from train.loss import MSE, MSE_Soft_Link_Sum
from train.train import make_data_splits, train_with_validation, save_training_losses

if __name__ == "__main__":
    # Model params
    model_type = "dense"
    biologically_informed = "encoder"
    soft_links = False
    activation_fn = torch.nn.ReLU
    loss_fn = MSE_Soft_Link_Sum() if soft_links else MSE()
    # GO params
    go_preprocessing = True
    merge_conditions = (1, 10)
    n_go_layers_used = 5
    # Training params
    dataset_name = "TCGA_complete_bp_top1k"
    n_samples = 10000
    batch_size = 100
    n_epochs = 300
    learning_rate = 0.005
    momentum = 0.9
    patience = 5
    device = "cpu"
    save_losses = False
    loss_path = ""
    # Model storage params
    save_weights = False
    load_weights = False
    weights_path = ""
    # Additional params
    data_split = 0.7
    seed = 1
    dtype = torch.float64
    n_nan_cols = 5

    # Data processing
    data = pd.read_csv(f"../../GO_TCGA/{dataset_name}.csv.gz", nrows=min(n_samples, 9797),
                       usecols=range(n_nan_cols + 1000), compression="gzip")
    dataloader, trainloader, validationloader, testloader = make_data_splits(data, n_nan_cols, n_samples, batch_size,
                                                                             data_split, seed)

    # GO processing
    if go_preprocessing:
        print("\n----- START: GO preprocessing -----")
        genes = list(data.columns[n_nan_cols:])
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

    model = Autoencoder(encoder, decoder)
    if load_weights:
        model.load_state_dict(torch.load(f"../saved_weights/{dataset_name}/{weights_path}.pt", weights_only=True))

    # Training
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    epoch_losses = train_with_validation(n_epochs, trainloader, testloader, validationloader, model, optimizer, loss_fn,
                                         patience, device)
    if save_losses:
        save_training_losses(epoch_losses, loss_path + ".txt")

    # Debug: show network weights as colored grid
    # plt.imshow(encoder.net_layers._modules["0"].weight.data, cmap="RdYlGn")
    if save_weights:
        torch.save(model.state_dict(), f"../saved_weights/{dataset_name}/{weights_path}.pt")
        print(f"\n----- Saved weights to file ({weights_path}) -----")

    # Plot train and test loss
    plt.plot(epoch_losses)
    plt.title(f"Loss for {model_type} BI-module: {biologically_informed} (n = {int(n_samples * data_split)})")
    plt.xlabel("Epoch")
    plt.ylabel(loss_fn.name)
    plt.legend(["train", "test"])
    plt.show()

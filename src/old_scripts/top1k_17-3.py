import torch
import pandas as pd
import numpy as np
import time

from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from thesis_binn.data_processing.dag_analysis import print_layers
from thesis_binn.data_processing.data_preprocessing import split_data_deprecated as split_data
from thesis_binn.data_processing.generate_masks import load_masks
from thesis_binn.data_processing.go_preprocessing import construct_go_bp_layers
from thesis_binn.model.Autoencoder import Autoencoder
from thesis_binn.model.Decoder import SparseBIDecoder, DenseBIDecoder, Decoder
from thesis_binn.model.Encoder import SparseBIEncoder, Encoder, DenseBIEncoder
from thesis_binn.train.loss import MSE
from thesis_binn.train.train import train, test

if __name__ == "__main__":
    # Model params
    model_type = "dense"
    biologically_informed = "encoder"
    soft_links = False
    activation = torch.nn.ReLU
    loss_function = MSE()
    # GO params
    go_preprocessing = True
    merge_conditions = (1, 30)
    n_go_layers_used = 7
    # Training params
    dataset_name = "GE_top1k_bp"
    n_samples = 10000
    batch_size = 100
    n_epochs = 200
    learning_rate = 0.005
    momentum = 0.9
    device = "cpu"
    # Model storage params
    save_weights = False
    load_weights = False
    weights_path = ""
    # Additional params
    data_split = 0.7
    seed = 1
    dtype = torch.float64
    n_nan_cols = 2

    # Data processing
    data = pd.read_csv(f"../../GO_TCGA/{dataset_name}.csv.gz", usecols=range(2 + min(n_samples, 11499)),
                       compression="gzip").sort_values("gene id")
    train_set, validation_set, test_set = split_data(data, n_nan_cols, data_split, seed)
    data_np = data.iloc[:, n_nan_cols:].to_numpy()
    data_torch = TensorDataset(torch.from_numpy(np.transpose(data_np)))
    dataloader = DataLoader(data_torch, batch_size=min(n_samples, batch_size), shuffle=False)
    train_torch = TensorDataset(torch.from_numpy(np.transpose(train_set.to_numpy())))
    validation_torch = TensorDataset(torch.from_numpy(np.transpose(validation_set.to_numpy())))
    test_torch = TensorDataset(torch.from_numpy(np.transpose(test_set.to_numpy())))
    trainloader = DataLoader(train_torch, batch_size=min(n_samples, batch_size), shuffle=False)
    validationloader = DataLoader(validation_torch, batch_size=min(n_samples, batch_size), shuffle=False)
    testloader = DataLoader(test_torch, batch_size=min(n_samples, batch_size), shuffle=False)

    # GO processing
    if go_preprocessing:
        print("\n----- START: GO preprocessing -----")
        genes = list(data["gene id"])
        go_layers = construct_go_bp_layers(genes, merge_conditions, print_go=True)
        masks = None
        print("----- COMPLETED: GO preprocessing -----")

    else:
        go_layers = torch.load(f"../masks/layers/{str(merge_conditions)}/{dataset_name}_layers.pt", weights_only=True)
        masks = load_masks(biologically_informed, merge_conditions, dataset_name, model_type)
        print("\n----- COMPLETED: Loading GO from file -----")

    # Model construction
    used_go_layers = go_layers[-min(n_go_layers_used, len(go_layers)):]
    print_layers(used_go_layers)
    if (biologically_informed == "encoder") or (biologically_informed == "both"):
        if masks:
            # Discard masks of unused GO layers
            module_masks = (masks.pop(0)[:min(n_go_layers_used - 1, len(go_layers))],
                            masks.pop(0)[:min(n_go_layers_used - 1, len(go_layers))])
        else:
            module_masks = None
        if model_type == "sparse":
            encoder = SparseBIEncoder(used_go_layers, activation, dtype, module_masks)
        else:
            encoder = DenseBIEncoder(used_go_layers, activation, dtype, masks=module_masks, soft_links=soft_links)
    else:
        encoder = Encoder(used_go_layers, activation, dtype)

    if (biologically_informed == "decoder") or (biologically_informed == "both"):
        if masks:
            # Discard masks of unused GO layers
            module_masks = (masks.pop(0)[-min(n_go_layers_used - 1, len(go_layers)):],
                            masks.pop(0)[-min(n_go_layers_used - 1, len(go_layers)):])
        else:
            module_masks = None
        if model_type == "sparse":
            decoder = SparseBIDecoder(used_go_layers, activation, dtype, masks=module_masks)
        else:
            decoder = DenseBIDecoder(used_go_layers, activation, dtype, masks=module_masks, soft_links=soft_links)
    else:
        decoder = Decoder(used_go_layers, activation, dtype)

    model = Autoencoder(encoder, decoder)
    if load_weights:
        model.load_state_dict(torch.load(f"../saved_weights/{dataset_name}/{weights_path}.pt", weights_only=True))

    # Training
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    epoch_losses = []
    t_start = time.time()
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        train_loss = train(trainloader, model, optimizer, loss_fn=loss_function, device=device)
        test_loss = test(testloader, model, loss_fn=loss_function, device=device)
        print(f"Train loss after epoch {epoch + 1}:\t{train_loss}\t\t"
              f"Test loss after epoch {epoch + 1}:\t{test_loss}")
        epoch_losses.append([train_loss.item(), test_loss.item()])
    t_end = time.time() - t_start
    print(f"Total training time: {t_end // 60:.0f}m {t_end % 60:.0f}s")

    # Debug: show network weights as colored grid
    # plt.imshow(encoder.net_layers._modules["0"].weight.data, cmap="RdYlGn")
    if save_weights:
        torch.save(model.state_dict(), f"../saved_weights/{dataset_name}/{weights_path}.pt")
        print(f"\n----- Saved weights to file ({weights_path}) -----")

    # Plot train and test loss
    plt.plot(epoch_losses)
    plt.title(f"Loss for {model_type} BI-module: {biologically_informed} (n = {int(n_samples * data_split)})")
    plt.xlabel("Epoch")
    plt.ylabel(loss_function.name)
    plt.legend(["train", "test"])
    plt.show()

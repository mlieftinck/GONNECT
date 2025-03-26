import torch
import pandas as pd
import numpy as np
import time

from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from data.DAGGenerator import DAGGenerator
from data.dag_analysis import print_layers, create_layers
from data.data_preprocessing import split_data
from data.go_preprocessing import balance_until_convergence, pull_leaves_down
from model.Autoencoder import Autoencoder
from model.Decoder import SparseBIDecoder, DenseBIDecoder, Decoder
from model.Encoder import SparseBIEncoder, Encoder, DenseBIEncoder
from train.loss import MSE, MSE_Soft_Link_Sum
from train.train import train, test

if __name__ == "__main__":
    # Model params
    model_type = "sparse"
    biologically_informed = "both"
    soft_links = True
    activation = torch.nn.ReLU()
    loss_function = MSE_Soft_Link_Sum()
    # GO params
    go_preprocessing = False
    merge_conditions = (1, 10)
    n_go_layers_used = 500
    # Training params
    dataset_name = "mini_dag"
    n_samples = 1000
    batch_size = 50
    n_epochs = 100
    learning_rate = 0.001
    momentum = 0.9
    device = "cpu"
    # Model storage params
    save_weights = False
    load_weights = False
    weights_path = "25-3"
    # Additional params
    data_split = 0.7
    seed = 1
    dtype = torch.float64
    n_nan_cols = 0

    # Data processing
    data = pd.DataFrame(np.random.random((3, n_samples)), columns=list("ID" + str(i) for i in range(n_samples)))
    train_set, validation_set, test_set = split_data(data, data_split, seed)
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
    go = DAGGenerator.dag4()
    size = len(go)
    balance_until_convergence(go, "A")
    pull_leaves_down(go, size)
    go_layers = create_layers(go)
    masks = None

    # Model construction
    used_go_layers = go_layers[-min(n_go_layers_used, len(go_layers)):]
    print_layers(used_go_layers, show_genes=go_preprocessing)
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
    # Debug
    aa_enc_lay = model.encoder.net_layers._modules
    aa_dec_lay = model.decoder.net_layers._modules
    if load_weights:
        model.load_state_dict(torch.load(f"../saved_weights/{dataset_name}/{weights_path}.pt", weights_only=True))

    # Training
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    epoch_losses = []
    t_start = time.time()
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        train_loss = train(trainloader, model, optimizer, loss_fn=loss_function, device=device)
        print(f"Train loss after epoch {epoch + 1}:\t{train_loss}")
        test_loss = test(testloader, model, loss_fn=loss_function, device=device)
        print(f"Test  loss after epoch {epoch + 1}:\t{test_loss}")
        epoch_losses.append([train_loss.item(), test_loss.item()])
    t_end = time.time() - t_start
    print(f"Total training time: {t_end // 60:.0f}m {t_end % 60:.0f}s")

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

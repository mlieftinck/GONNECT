import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pandas as pd
from thesis_binn.data_processing.go_preprocessing import *
from thesis_binn.model.Autoencoder import Autoencoder
from thesis_binn.model.deprecated.OldDecoder import Decoder
from thesis_binn.model.deprecated.OldEncoder import Encoder, SparseEncoder
from thesis_binn.train.train import train

if __name__ == "__main__":
    go_preprocessing = True
    save = False
    data = pd.read_csv("../../GO_TCGA/GE_bp_100.csv.gz", compression='gzip').sort_values("gene id")
    n_nan_cols = 3
    genes = list(data["gene id"])
    merge_conditions = (1, 10)
    n_samples = data.shape[1] - n_nan_cols
    batch_size = 20
    n_epochs = 10
    dtype = torch.float64
    lr_bi, lr_fc = 1e-3, 1e-3
    if go_preprocessing:
        print("\n----- START: GO preprocessing -----")
        # Initialize GO layers, prune the top off
        layers = construct_go_bp_layers(genes, merge_conditions, print_go=True)
        layers = layers[2:]
        print([len(layer) for layer in layers])
        masks = None
        print("----- COMPLETED: GO preprocessing -----")
        if save:
            layer_copy = [torch.zeros(len(layer)) for layer in layers]
            torch.save(layer_copy, f"../masks/{str(merge_conditions)}/bp_100_layers.pt")
            print("----- Saved layers to file -----")
    else:
        print("\n----- START: Loading GO from file -----")
        layers = torch.load(f"../masks/{str(merge_conditions)}/bp_100_layers.pt", weights_only=True)
        mask_edge = torch.load(f"../masks/{str(merge_conditions)}/bp_100_sparse_edge_masks.pt", weights_only=True)
        mask_proxy = torch.load(f"../masks/{str(merge_conditions)}/bp_100_sparse_proxy_masks.pt", weights_only=True)
        masks = (mask_edge, mask_proxy)
        print("----- COMPLETED: Loading GO from file -----")

    # Convert dataset from pandas to torch
    data_np = data.iloc[:, n_nan_cols:].to_numpy()
    torch_dataset = TensorDataset(torch.from_numpy(np.transpose(data_np)))
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    # Construct model(s)
    model = Autoencoder(SparseEncoder(layers, dtype, masks=masks), Decoder(layers, dtype))
    optimizer = optim.SGD(model.parameters(), lr=lr_bi)
    model_vanilla = Autoencoder(Encoder(layers, dtype), Decoder(layers, dtype))
    optimizer_vanilla = optim.SGD(model_vanilla.parameters(), lr=lr_fc)

    # Set the number of epochs to for training
    epochs = n_epochs
    epoch_losses = []
    epoch_losses_vanilla = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = train(dataloader, model, optimizer)
        epoch_losses.append(train_loss.item())
        print(f"Training loss after epoch {epoch + 1}: {train_loss}")
        train_loss_vanilla = train(dataloader, model_vanilla, optimizer_vanilla)
        epoch_losses_vanilla.append(train_loss_vanilla.item())
        # print(f"Training loss after epoch {epoch + 1}: {train_loss_vanilla}")

    plt.plot(epoch_losses)
    plt.plot(epoch_losses_vanilla)
    plt.legend(["Training loss BI", "Training loss FC"])
    plt.show()

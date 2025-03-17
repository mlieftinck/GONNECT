import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pandas as pd
from data.go_preprocessing import *
from model.Autoencoder import Autoencoder
from model.OldDecoder import Decoder
from model.OldEncoder import BIEncoder, SparseEncoder
from train.train import train
import time

if __name__ == "__main__":
    data = pd.read_csv("../../GO_TCGA/GE_bp_100.csv.gz", compression='gzip').sort_values("gene id")
    n_nan_cols = 3
    genes = list(data["gene id"])
    merge_conditions = (1, 10)
    n_samples = data.shape[1] - n_nan_cols
    batch_size = 20
    n_epochs = 10
    dtype = torch.float64
    loss_fn = torch.nn.functional.mse_loss
    lr = 1e-2
    n_runs = 3

    # Initialize GO layers, prune the top off
    layers = construct_go_bp_layers(genes, merge_conditions=merge_conditions, print=True)
    layers = layers[2:]

    # Convert dataset from pandas to torch
    data_np = data.iloc[:, n_nan_cols:].to_numpy()
    torch_dataset = TensorDataset(torch.from_numpy(np.transpose(data_np)))
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    # TRAINING
    times_dense = []
    times_sparse = []
    epoch_loss_dense = []
    epoch_loss_sparse = []
    for run in range(n_runs):
        # DENSE TRAINING
        print(f"\n----- Start Dense Training ({run + 1}/{n_runs}) -----")
        t_start_dense = time.time()
        # Construct model
        model_dense = Autoencoder(BIEncoder(layers, dtype), Decoder(layers, dtype))
        optimizer_dense = optim.SGD(model_dense.parameters(), lr=lr)

        # Set the number of epochs for training
        epochs = n_epochs
        epoch_losses_dense = []
        for epoch in range(epochs):  # loop over the dataset multiple times
            train_loss = train(dataloader, model_dense, optimizer_dense, loss_fn=loss_fn)
            epoch_losses_dense.append(train_loss.item())
            print(f"Training loss after epoch {epoch + 1}: {train_loss}")
        t_dense = time.time() - t_start_dense

        # SPARSE TRAINING
        print(f"\n----- Start Sparse Training ({run + 1}/{n_runs}) -----")
        t_start_sparse = time.time()
        # Construct model
        model_sparse = Autoencoder(SparseEncoder(layers, dtype), Decoder(layers, dtype))
        optimizer_sparse = optim.SGD(model_sparse.parameters(), lr=lr)

        # Set the number of epochs for training
        epochs = n_epochs
        epoch_losses_sparse = []
        for epoch in range(epochs):  # loop over the dataset multiple times
            train_loss = train(dataloader, model_sparse, optimizer_sparse, loss_fn=loss_fn)
            epoch_losses_sparse.append(train_loss.item())
            print(f"Training loss after epoch {epoch + 1}: {train_loss}")
        t_sparse = time.time() - t_start_sparse

        times_dense.append(t_dense)
        times_sparse.append(t_sparse)
        epoch_loss_dense.append(epoch_losses_dense)
        epoch_loss_sparse.append(epoch_losses_sparse)

    mean_time_dense = np.mean(times_dense)
    mean_time_sparse = np.mean(times_sparse)
    mean_loss_dense = np.array(epoch_loss_dense).mean(axis=0)
    mean_loss_sparse = np.array(epoch_loss_sparse).mean(axis=0)
    print(f"\nMean training time dense Tensors: {mean_time_dense:.2f} seconds")
    print(f"Mean training time sparse Tensors: {mean_time_sparse:.2f} seconds")
    plt.plot(range(1, n_epochs + 1), mean_loss_dense)
    plt.plot(range(1, n_epochs + 1), mean_loss_sparse)
    plt.title(f"Mean training loss over {n_runs} runs for BI-FC")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend(["Dense Tensors", "Sparse Tensors"])
    plt.show()

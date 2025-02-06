import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pandas as pd
from data.go_preprocessing import *
from model.Autoencoder import Autoencoder
from model.Decoder import Decoder
from model.Encoder import BIEncoder, Encoder
from train import train

if __name__ == "__main__":
    data = pd.read_csv("../../GO_TCGA/GE_bp_100.csv.gz", compression='gzip')
    n_nan_cols = 3
    genes = list(data["gene id"])
    n_samples = data.shape[1] - n_nan_cols
    batch_size = 20
    n_epochs = 10
    dtype = torch.float64

    # Initialize GO layers, prune the top off
    layers = construct_go_bp_layers(genes)
    layers = layers[2:]

    # Convert dataset from pandas to torch
    data_np = data.iloc[:, n_nan_cols:].to_numpy()
    test = np.transpose(data_np)
    torch_dataset = TensorDataset(torch.from_numpy(np.transpose(data_np)))
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    # Construct model(s)
    model = Autoencoder(BIEncoder(layers, dtype), Decoder(layers, dtype))
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model_vanilla = Autoencoder(Encoder(layers, dtype), Decoder(layers, dtype))
    optimizer_vanilla = optim.Adam(model_vanilla.parameters(), lr=1e-5)

    # Set the number of epochs to for training
    epochs = n_epochs
    epoch_losses = []
    epoch_losses_vanilla = []
    # for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = train(dataloader, model, optimizer, loss_fn="broodrooster")
        epoch_losses.append(train_loss.item())
        print(f"Training loss after epoch {epoch + 1}: {train_loss}")
        train_loss_vanilla = train(dataloader, model_vanilla, optimizer_vanilla, loss_fn="broodrooster")
        epoch_losses_vanilla.append(train_loss_vanilla.item())

    plt.plot(epoch_losses)
    plt.plot(epoch_losses_vanilla)
    plt.legend(["Training loss BI", "Training loss vanilla"])
    plt.show()

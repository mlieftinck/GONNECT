import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pandas as pd
from data.go_preprocessing import *
from model.Autoencoder import Autoencoder
from model.Decoder import Decoder
from model.Encoder import Encoder, SparseEncoder
from train.train import train

if __name__ == "__main__":
    n_samples = 1
    n_nan_cols = 3
    data = pd.read_csv("../../GO_TCGA/GE_bp_100.csv.gz", usecols=range(n_nan_cols + n_samples),
                       compression='gzip').sort_values("gene id")
    genes = list(data["gene id"])
    merge_conditions = (1, 10)
    n_layers_used = 6
    batch_size = min(n_samples, 50)
    n_epochs = 200
    dtype = torch.float64
    lr = 1e-2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    go_layers = torch.load(f"../masks/{str(merge_conditions)}/bp_100_layers.pt", weights_only=True)
    encoder_layers = go_layers[-n_layers_used:]
    mask_edge = torch.load(f"../masks/{str(merge_conditions)}/bp_100_sparse_edge_masks.pt", weights_only=True)
    mask_proxy = torch.load(f"../masks/{str(merge_conditions)}/bp_100_sparse_proxy_masks.pt", weights_only=True)
    masks = (mask_edge, mask_proxy)
    decoder_layers = [torch.zeros(len(encoder_layers[0])), torch.zeros(75), torch.zeros(len(encoder_layers[-1]))]

    # Convert dataset from pandas to torch
    data_np = data.iloc[:, n_nan_cols:].to_numpy()
    torch_dataset = TensorDataset(torch.from_numpy(np.transpose(data_np)))
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    # Construct model(s)
    model = Autoencoder(SparseEncoder(encoder_layers, dtype, masks), Decoder(decoder_layers, dtype))
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Set the number of epochs to for training
    epochs = n_epochs
    epoch_losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = train(dataloader, model, optimizer, device=device)
        epoch_losses.append(train_loss.item())
        print(f"Training loss after epoch {epoch + 1}: {train_loss}")
    plt.plot(epoch_losses)
    plt.show()

    test_x = torch.tensor((data["ID0"])).reshape((1, 100)).to(device)
    test_y = model(test_x)
    print(test_y)

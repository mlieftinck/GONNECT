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
    biologically_informed = True
    n_samples = 1000
    n_nan_cols = 3
    data = pd.read_csv("../../../GO_TCGA/GE_bp_100.csv.gz", usecols=range(n_nan_cols + n_samples),
                       compression='gzip').sort_values("gene id")
    genes = list(data["gene id"])
    merge_conditions = (1, 10)
    n_layers_used = 6
    batch_size = min(n_samples, 50)
    n_epochs = 50
    dtype = torch.float64
    lr = 1e-2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

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
    if biologically_informed:
        model = Autoencoder(SparseEncoder(encoder_layers, dtype, masks), Decoder(decoder_layers, dtype))
    else:
        model = Autoencoder(Encoder(encoder_layers, dtype), Decoder(decoder_layers, dtype))
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Set the number of epochs to for training
    epochs = n_epochs
    epoch_losses = []
    t_start = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = train(dataloader, model, optimizer, device=device)
        epoch_losses.append(train_loss.item())
        print(f"Training loss after epoch {epoch + 1}: {train_loss}")
    t_end = time.time() - t_start
    print(f"Training time: {t_end:.2f} seconds")

    plt.plot(epoch_losses)
    if biologically_informed:
        plt.title(f"Training loss for BI-encoder (sparse) (n = {n_samples})")
    else:
        plt.title(f"Training loss for fully-connected encoder (dense) (n = {n_samples})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.show()

    # test_x = torch.tensor((data["ID0"])).reshape((1, 100)).to(device)
    # test_y = model(test_x)
    # print(test_y)

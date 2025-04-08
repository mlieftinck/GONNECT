import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pandas as pd

from data.data_preprocessing import split_data
from data.go_preprocessing import *
from model.Autoencoder import Autoencoder
from model.deprecated.OldDecoder import Decoder
from model.deprecated.OldEncoder import Encoder, SparseEncoder
from train.train import train, test

if __name__ == "__main__":
    # GO params
    go_preprocessing = False
    merge_conditions = (1, 10)
    n_go_layers_used = 6
    # Save model params
    save_architecture = False
    save_weights = False
    save_weights_path = "model_1"
    load_weights = False
    load_weights_path = "model_1"
    biologically_informed = True
    # Data params
    n_samples = 12000
    batch_size = 500
    split = 0.7
    seed = 10
    dataset_name = "GE_top1k_bp"
    n_nan_cols = 2
    dtype = torch.float64
    data = pd.read_csv(f"../../GO_TCGA/{dataset_name}.csv.gz", usecols=range(n_nan_cols + min(n_samples, 11499)),
                       compression="gzip").sort_values("gene id")
    # Training params
    n_epochs = 1000
    lr = 1e-4
    momentum = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cpu"
    print("device: ", device)

    if go_preprocessing:
        print("\n----- START: GO preprocessing -----")
        # Initialize GO layers
        genes = list(data["gene id"])
        layers = construct_go_bp_layers(genes, merge_conditions, print_go=True)
        masks = None
        go_layers = layers
        print("----- COMPLETED: GO preprocessing -----")
        if save_architecture:
            layer_copy = [torch.zeros(len(layer)) for layer in layers]
            torch.save(layer_copy, f"../masks/{str(merge_conditions)}/{dataset_name}_layers.pt")
            print("----- Saved layers to file -----")

    else:
        go_layers = torch.load(f"../masks/{str(merge_conditions)}/{dataset_name}_layers.pt", weights_only=True)

    encoder_layers = go_layers[-n_go_layers_used:]
    mask_edge = torch.load(f"../masks/{str(merge_conditions)}/{dataset_name}_sparse_edge_masks.pt", weights_only=True)
    mask_proxy = torch.load(f"../masks/{str(merge_conditions)}/{dataset_name}_sparse_proxy_masks.pt", weights_only=True)
    masks = (mask_edge, mask_proxy)
    decoder_layers = [torch.zeros(len(encoder_layers[0])), torch.zeros(75), torch.zeros(len(encoder_layers[-1]))]

    # Prepare data splits and pandas to torch conversion
    train_set, validation_set, test_set = split_data(data, n_nan_cols, split, seed)
    data_np = data.iloc[:, n_nan_cols:].to_numpy()
    data_torch = TensorDataset(torch.from_numpy(np.transpose(data_np)))
    dataloader = DataLoader(data_torch, batch_size=min(n_samples, batch_size), shuffle=False)
    train_torch = TensorDataset(torch.from_numpy(np.transpose(train_set.to_numpy())))
    validation_torch = TensorDataset(torch.from_numpy(np.transpose(validation_set.to_numpy())))
    test_torch = TensorDataset(torch.from_numpy(np.transpose(test_set.to_numpy())))
    trainloader = DataLoader(train_torch, batch_size=min(n_samples, batch_size), shuffle=False)
    validationloader = DataLoader(validation_torch, batch_size=min(n_samples, batch_size), shuffle=False)
    testloader = DataLoader(test_torch, batch_size=min(n_samples, batch_size), shuffle=False)

    # Construct model and optimizer
    if biologically_informed:
        model = Autoencoder(SparseEncoder(encoder_layers, dtype, masks), Decoder(decoder_layers, dtype))
    else:
        model = Autoencoder(Encoder(encoder_layers, dtype), Decoder(decoder_layers, dtype))
    if load_weights:
        model.load_state_dict(torch.load(f"../saved_weights/{dataset_name}/{save_weights_path}.pt", weights_only=True))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Set the number of epochs to for training
    epochs = n_epochs
    epoch_losses = []
    t_start = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = train(trainloader, model, optimizer, device=device)
        print(f"Train loss after epoch {epoch + 1}:\t{train_loss}")
        test_loss = test(testloader, model, device=device)
        print(f"Test  loss after epoch {epoch + 1}:\t{test_loss}")
        epoch_losses.append([train_loss.item(), test_loss.item()])
    t_end = time.time() - t_start
    print(f"Total training time: {t_end // 60:.0f}m {t_end % 60:.0f}s")

    if save_weights:
        torch.save(model.state_dict(), f"../saved_weights/{dataset_name}/{save_weights_path}.pt")
        print(f"\n----- Saved weights to file ({save_weights_path}) -----")

    plt.plot(epoch_losses)
    if biologically_informed:
        plt.title(f"Loss for BI-encoder (sparse) (n = {len(trainloader)})")
    else:
        plt.title(f"Loss for fully-connected encoder (dense) (n = {len(trainloader)})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend(["train", "test"])
    plt.show()

    # test_x = torch.tensor((data["ID0"])).reshape((1, 100)).to(device)
    # test_y = model(test_x)
    # print(test_y)

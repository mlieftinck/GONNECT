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
    batch_size = 200
    n_epochs = 100
    dtype = torch.float64

    # Initialize GO DAG
    go_main = create_dag("../data/go-basic.obo")
    go_bp = filter_by_namespace(go_main, {"biological_process"})
    go = copy_dag(go_bp)
    # Process GO DAG
    # Add genes
    link_genes_to_go_by_namespace(go, "../../GO_TCGA/goa_human.gaf", "biological_process", genes)
    print_layers(create_layers(go))
    remove_geneless_branches(go)
    print_layers(create_layers(go))
    # Merge-prune
    merge_prune_until_convergence(go, 1, 10)
    print_layers(create_layers(go))
    # Add proxies
    go_proxyless = copy_dag(go)
    balance_until_convergence(go)
    pull_leaves_down(go, len(go_proxyless))
    print_layers(create_layers(go))
    # Remove superroot
    remove_superroot(go)
    # Layerize DAG
    print_dag_info(go)
    layers = create_layers(go)

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

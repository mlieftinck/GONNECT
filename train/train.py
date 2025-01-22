from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from data.DAGGenerator import DAGGenerator
from model.VAE_vanilla import VarAutoencoder
from model.AE_vanilla import Autoencoder
from data.go_preprocessing import *
from torchsummary import summary


def loss_kl_divergence(inputs, outputs, net):
    return ((inputs - outputs) ** 2).sum() + net.encoder.kl


def train_vae(train_loader, net, optimizer, loss_fn=""):
    """Trains variational autoencoder network for one epoch in batches.
    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        loss_fn: Loss function."""

    avg_loss = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        inputs = data[0]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        if loss_fn == "kl":
            loss = loss_kl_divergence(inputs, outputs, net)
        else:
            loss = F.mse_loss(outputs, inputs)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss

    return avg_loss / len(train_loader)


if __name__ == "__main__":
    n_samples = 100
    batch_size = 10
    n_epochs = 100

    # DAG to layers
    dag = DAGGenerator.dag3()
    go = copy_dag(dag)
    balance_until_convergence(go, root_id="A")
    pull_leaves_down(go, len(dag))
    print_dag_info(go)
    layers = create_layers(go)
    # Load data (samples, genes)
    data = TensorDataset(torch.randn(n_samples, len(layers[-1])))

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    # model = Autoencoder(layers)
    model = VarAutoencoder(layers)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    # summary(model, (1, 100), 1, "cpu")

    # Set the number of epochs to for training
    epochs = n_epochs
    epoch_losses = []
    # for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = train_vae(dataloader, model, optimizer, loss_fn="kl")
        epoch_losses.append(train_loss.item())
        print(f"Training loss after epoch {epoch + 1}: {train_loss}")
    plt.plot(epoch_losses)
    plt.show()

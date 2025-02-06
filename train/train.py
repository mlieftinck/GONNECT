import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from data.go_preprocessing import *
from data.DAGGenerator import DAGGenerator
from model.Encoder import Encoder, BIEncoder, SparseEncoder
from model.Decoder import Decoder, BIDecoder, SparseDecoder
from model.Autoencoder import Autoencoder


def loss_kl_divergence(inputs, outputs, net):
    return ((inputs - outputs) ** 2).sum() + net.encoder.kl


def train(train_loader, net, optimizer, loss_fn=""):
    """Trains variational autoencoder network for one epoch in batches.
    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        loss_fn: Loss function."""

    avg_loss = 0
    # Iterate over batches
    for i, data in enumerate(train_loader):
        inputs = data[0]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Backward pass
        if loss_fn == "kl":
            loss = loss_kl_divergence(inputs, outputs, net)
        else:
            loss = F.mse_loss(outputs, inputs)
        loss.backward()

        # Force gradients (Option 1)
        if isinstance(net.encoder, BIEncoder):
            net.encoder.mask_gradients()

        test = net.encoder.layers[0]
        optimizer.step()

        # Force weights (Options 2)
        if isinstance(net.encoder, BIEncoder):
            net.encoder.mask_weights()

        # keep track of loss and accuracy
        avg_loss += loss

    return avg_loss / len(train_loader)


if __name__ == "__main__":
    n_samples = 1000
    batch_size = 10
    n_epochs = 10
    dtype=torch.float32

    # DAG to layers
    dag = DAGGenerator.dag4()
    go = copy_dag(dag)
    balance_until_convergence(go, root_id="A")
    pull_leaves_down(go, len(dag))
    print_dag_info(go)
    layers = create_layers(go)

    # Load data (samples, genes)
    data = TensorDataset(torch.randn(n_samples, len(layers[-1]), dtype=dtype))

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    model = Autoencoder(SparseEncoder(layers, dtype=dtype), SparseDecoder(layers, dtype=dtype))

    optimizer = optim.SGD(model.parameters(), lr=5e-3)

    # Set the number of epochs for training
    epochs = n_epochs
    epoch_losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = train(dataloader, model, optimizer, loss_fn="broodrooster")
        epoch_losses.append(train_loss.item())
        print(f"Training loss after epoch {epoch + 1}: {train_loss}")

    plt.plot(epoch_losses)
    plt.show()
    pass

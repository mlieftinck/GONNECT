import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from data.go_preprocessing import *
from data.DAGGenerator import DAGGenerator
from model.deprecated.OldEncoder import Encoder, BIEncoder
from model.deprecated.OldDecoder import Decoder
from model.Autoencoder import Autoencoder


def loss_kl_divergence(inputs, outputs, net):
    return ((inputs - outputs) ** 2).sum() + net.encoder.kl


def train(train_loader, net, optimizer, loss_fn, device="cpu"):
    """Trains variational autoencoder network for one epoch in batches.
    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        loss_fn: Loss function.
        device: Whether the network runs on CPU or GPU."""

    net.to(device)
    avg_loss = 0
    # Iterate over batches
    for i, data in enumerate(train_loader):
        inputs = data[0]
        inputs = inputs.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Backward pass
        if loss_fn == "kl":
            loss = loss_kl_divergence(inputs, outputs, net)
        else:
            loss = loss_fn(outputs, inputs, model=net)
        loss.backward()

        # Force gradients (optional as weight masking should be sufficient)
        if isinstance(net.encoder, BIEncoder):
            net.encoder.mask_gradients()

        optimizer.step()

        # Force biologically-informed weights
        net.encoder.mask_weights()

        # keep track of loss and accuracy
        avg_loss += loss

    return avg_loss / len(train_loader)


def test(test_loader, net, loss_fn, device="cpu"):
    net.to(device)
    avg_loss = 0

    with torch.no_grad():
        # Iterate over batches
        for i, data in enumerate(test_loader):
            inputs = data[0]
            inputs = inputs.to(device)

            # Forward pass
            outputs = net(inputs)
            loss = loss_fn(outputs, inputs, model=net)

            # keep track of loss and accuracy
            avg_loss += loss

    return avg_loss / len(test_loader)


if __name__ == "__main__":
    n_samples = 10
    batch_size = 10
    n_epochs = 1000
    lr = 1e-1
    dtype = torch.float32

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

    model = Autoencoder(Encoder(layers, dtype=dtype), Decoder(layers, dtype=dtype))

    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Set the number of epochs for training
    epochs = n_epochs
    epoch_losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = train(dataloader, model, optimizer)
        epoch_losses.append(train_loss.item())
        print(f"Training loss after epoch {epoch + 1}: {train_loss}")

    plt.plot(epoch_losses)
    plt.show()
    pass

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, go_layers):
        super(Encoder, self).__init__()
        self.layers = []
        self.relu = nn.ReLU()
        for i in range(len(go_layers) - 1):
            self.layers.append(nn.Linear(len(go_layers[i]), len(go_layers[i + 1])))

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return x


class Decoder(nn.Module):
    def __init__(self, go_layers):
        super(Decoder, self).__init__()
        self.layers = []
        self.relu = nn.ReLU()
        for i in range(len(go_layers) - 1):
            self.layers.append(nn.Linear(len(go_layers[i]), len(go_layers[i + 1])))

    def forward(self, z):
        for layer in self.layers:
            z = self.relu(layer(z))
        return z


class Autoencoder(nn.Module):
    def __init__(self, go_layers):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(go_layers)
        self.decoder = Decoder(list(reversed(go_layers)))

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

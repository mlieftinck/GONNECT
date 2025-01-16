import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, go_layers):
        super(Encoder, self).__init__()
        self.layers = []
        self.relu = nn.ReLU()
        for i in range(len(go_layers) - 1):
            self.layers.append(nn.Linear(len(go_layers[i]), len(go_layers[i + 1])))
            if i < len(go_layers) - 2:
                self.layers.append(self.relu)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, go_layers):
        super(Decoder, self).__init__()
        self.layers = []
        self.relu = nn.ReLU()
        for i in range(len(go_layers) - 1):
            self.layers.append(nn.Linear(len(go_layers[i]), len(go_layers[i + 1])))
            if i < len(go_layers) - 2:
                self.layers.append(self.relu)
            else:
                self.layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, z):
        return self.layers(z)


class Autoencoder(nn.Module):
    def __init__(self, go_layers):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(list(reversed(go_layers)))
        self.decoder = Decoder(go_layers)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

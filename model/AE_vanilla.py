import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dims, input_dims, hdim, dims=None):
        super(Encoder, self).__init__()

        # TODO: Generalize the vanilla implementations for variable model sizes
        # self.layers = []
        # for i in range(len(dims)-1):
        #     self.layers.append(nn.Linear(dims[i], dims[i+1]))

        self.linear1 = nn.Linear(input_dims, hdim[0])
        self.linear2 = nn.Linear(hdim[0], hdim[1])
        self.linear3 = nn.Linear(hdim[1], latent_dims)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

        # for layer in self.layers:
        #     x = self.relu(layer(x))

        return x


class Decoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(latent_dims, hdim[1])
        self.linear2 = nn.Linear(hdim[1], hdim[0])
        self.linear3 = nn.Linear(hdim[0], s_img * s_img)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.relu(self.linear1(z))
        z = self.relu(self.linear2(z))
        z = self.sigmoid(self.linear3(z))
        return z


class Autoencoder(nn.Module):
    def __init__(self, latent_dims, input_dims, hdim=[100, 50]):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(latent_dims, input_dims, hdim)
        self.decoder = Decoder(latent_dims, input_dims, hdim)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

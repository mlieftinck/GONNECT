import torch
import torch.nn as nn
from model.AE_vanilla import Decoder


class VarEncoder(nn.Module):
    def __init__(self, go_layers):
        super(VarEncoder, self).__init__()
        self.layers_mu = []
        self.layers_sigma = []
        self.relu = nn.ReLU()

        for i in range(len(go_layers) - 1):
            self.layers_mu.append(nn.Linear(len(go_layers[i]), len(go_layers[i + 1])))
            self.layers_sigma.append(nn.Linear(len(go_layers[i]), len(go_layers[i + 1])))
            if i < len(go_layers) - 2:
                self.layers_mu.append(self.relu)
                self.layers_sigma.append(self.relu)
        self.layers_mu = nn.Sequential(*self.layers_mu)
        self.layers_sigma = nn.Sequential(*self.layers_sigma)

        # distribution setup
        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.to(try_gpu())  # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.to(try_gpu())
        self.kl = 0

    # Kullback-Leibner loss
    def kull_leib(self, mu, sigma):
        return (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

    # Reparametrization trick
    def reparameterize(self, mu, sig):
        return mu + sig * self.N.sample(mu.shape)

    def forward(self, x):
        x_mu = self.layers_mu(x)
        x_sigma = self.layers_sigma(x)

        # Form distribution (exp() so encoder learns log(sigma))
        sigma = torch.exp(x_sigma)
        mu = x_mu

        # Reparameterize to find z
        z = self.reparameterize(mu, sigma)

        # loss between N(0,I) and learned distribution
        self.kl = self.kull_leib(mu, sigma)
        return z


class VarAutoencoder(nn.Module):
    def __init__(self, go_layers):
        super(VarAutoencoder, self).__init__()

        self.encoder = VarEncoder(list(reversed(go_layers)))
        self.decoder = Decoder(go_layers)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y

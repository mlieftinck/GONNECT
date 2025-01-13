import torch
import torch.nn as nn
from AE_vanilla import Decoder


class VarEncoder(nn.Module):
    def __init__(self, go_layers):
        super(VarEncoder, self).__init__()
        self.layers_mu = []
        self.layers_sigma = []
        self.relu = nn.ReLU()

        for i in range(len(go_layers) - 1):
            self.layers_mu.append(nn.Linear(len(go_layers[i]), len(go_layers[i + 1])))
            self.layers_sigma.append(nn.Linear(len(go_layers[i]), len(go_layers[i + 1])))

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
        x_mu = x.copy()
        x_sigma = x.copy()
        for i in range(len(self.layers_mu) - 1):
            x_mu = self.relu(self.layers_mu[i](x_mu))

        for j in range(len(self.layers_sigma) - 1):
            x_sigma = self.relu(self.layers_sigma[j](x_sigma))

        # Form distribution (exp() so encoder learns log(sigma))
        sigma = torch.exp(self.layers_sigma[-1](x_sigma))
        mu = self.layers_mu[-1](x_mu)

        # Reparameterize to find z
        z = self.reparameterize(mu, sigma)

        # loss between N(0,I) and learned distribution
        self.kl = self.kull_leib(mu, sigma)
        return z


class VarAutoencoder(nn.Module):
    def __init__(self, go_layers):
        super(VarAutoencoder, self).__init__()

        self.encoder = VarEncoder(go_layers)
        self.decoder = Decoder(list(reversed(go_layers)))

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y

import torch
import torch.nn as nn
from AE_vanilla import Decoder


class VarEncoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim):
        super(VarEncoder, self).__init__()

        # layers for g1
        self.linear1_1 = nn.Linear(s_img * s_img, hdim[0])
        self.linear2_1 = nn.Linear(hdim[0], hdim[1])
        self.linear3_1 = nn.Linear(hdim[1], latent_dims)

        # layers for g2
        self.linear1_2 = nn.Linear(s_img * s_img, hdim[0])
        self.linear2_2 = nn.Linear(hdim[0], hdim[1])
        self.linear3_2 = nn.Linear(hdim[1], latent_dims)

        self.relu = nn.ReLU()

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
        x1 = self.relu(self.linear1_1(x))
        x1 = self.relu(self.linear2_1(x1))

        x2 = self.relu(self.linear1_2(x))
        x2 = self.relu(self.linear2_2(x2))

        # Form distribution (exp() so encoder learns log(sig))
        sig = torch.exp(self.linear3_1(x1))
        mu = self.linear3_2(x2)

        # Reparameterize to find z
        z = self.reparameterize(mu, sig)

        # loss between N(0,I) and learned distribution
        self.kl = self.kull_leib(mu, sig)
        return z


class VarAutoencoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim=[100, 50]):
        super(VarAutoencoder, self).__init__()

        self.encoder = VarEncoder(latent_dims, s_img, hdim)
        self.decoder = Decoder(latent_dims, s_img, hdim)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y

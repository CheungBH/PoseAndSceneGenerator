from torch import nn
import torch
import numpy as np
from torch.autograd import Variable

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def reparameterization(mu, logvar, latent_dim):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z



class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 35),
            nn.Tanh(),
        )

    def forward(self, z):
        kps_gen = self.model(z)
        return kps_gen


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(35, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, kps):
        x = self.model(kps)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, self.latent_dim)
        return z


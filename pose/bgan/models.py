from torch import nn
import torch
import numpy as np
from torch.autograd import Variable

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Generator(nn.Module):
    def __init__(self, latent_dim, fn):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, fn),
            nn.Tanh()
        )

    def forward(self, z):
        kps_gen = self.model(z)
        return kps_gen


class Discriminator(nn.Module):
    def __init__(self, latent_dim, fn):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(fn, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # img_flat = img.view(img.shape[0], -1)
        # validity = self.model(img_flat)
        validity = self.model(z)
        return validity

from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, latent_dim, feature_num):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(1, 1)

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
            nn.Linear(1024, feature_num),
            nn.Tanh()
        )

    def forward(self, noise):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((noise), -1)
        img = self.model(gen_input)
        # img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, latent_dim, feature_num):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(1, 1)

        self.model = nn.Sequential(
            nn.Linear(feature_num, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class GeneratorKPS(nn.Module):
    def __init__(self, latent_dim, feature_num):
        super(GeneratorKPS, self).__init__()
        self.label_emb = nn.Embedding(1, 1)

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
            nn.Linear(1024, feature_num),
            nn.Tanh()
        )

    def forward(self, noise):
        # Concatenate label embedding and image to produce input
        gen_input = noise#torch.cat((noise), -1)
        img = self.model(gen_input)
        # img = img.view(img.size(0), *img_shape)
        return img


class DiscriminatorKPS(nn.Module):
    def __init__(self, latent_dim, feature_num):
        super(DiscriminatorKPS, self).__init__()

        # self.label_embedding = nn.Embedding(1, 1)

        self.model = nn.Sequential(
            nn.Linear(feature_num, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img):
        # Concatenate label embedding and image to produce input
        d_in = img.view(img.size(0), -1)
        validity = self.model(d_in)
        return validity

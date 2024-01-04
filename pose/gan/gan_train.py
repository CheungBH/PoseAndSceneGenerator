import argparse
import os
import numpy as np
import math
import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn
import torch.nn.functional as F
import torch
from models import Generator, Discriminator
from utils import KPSDataset

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--save_dir", type=str, default="../../exp/gan", help="interval between image sampling")
parser.add_argument("--action", type=str, required=True)
parser.add_argument("--data_path", type=str, default="", help="interval between image sampling")
parser.add_argument("--feature_num", type=int, default=34, help="")
opt = parser.parse_args()
print(opt)

# img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(opt.latent_dim, opt.feature_num)
discriminator = Discriminator(opt.latent_dim, opt.feature_num)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
dataset = KPSDataset(opt.data_path, opt.action)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_kps(n_row, batches_done, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    color_dict = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]

    connections = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 3), (2, 4), (3, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12)]

    def choose_color(coord_i):
        if coord_i <= 4:
            return color_dict[0]
        elif coord_i in [5, 7, 9]:
            return color_dict[1]
        elif coord_i in [6, 8, 10]:
            return color_dict[2]
        elif coord_i in [11, 13, 15]:
            return color_dict[3]
        elif coord_i in [12, 14, 16]:
            return color_dict[4]

    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    gen_kps = generator(z)
    imgs = []
    for gen_kp in gen_kps:
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        float_single_coord = [x * 400 for x in gen_kp]
        # print(label)
        for i in range(17):
            x = int(float_single_coord[i * 2])
            y = int(float_single_coord[i * 2 + 1])
            cv2.circle(image, (x, y), 5, choose_color(i), -1)

        for i, j in connections:
            x1, y1 = int(float_single_coord[i * 2]), int(float_single_coord[i * 2 + 1])
            x2, y2 = int(float_single_coord[j * 2]), int(float_single_coord[j * 2 + 1])
            cv2.line(image, (x1, y1), (x2, y2), choose_color(i), 2)
        imgs.append(image)

    concated_imgs = []
    for i in range(n_row):
        horizontal = np.concatenate(imgs[i*n_row:(i+1)*n_row], axis=1)
        concated_imgs.append(horizontal)
    concated_imgs = np.concatenate(concated_imgs, axis=0)
    cv2.imwrite(os.path.join("{}/{}.png").format(save_dir, batches_done), concated_imgs)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, kps in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(kps.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(kps.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(kps.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (kps.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_kps(n_row=int(math.sqrt(kps.shape[0])), batches_done=batches_done,
                       save_dir=os.path.join(opt.save_dir, "image"))

    if epoch % 10 == 0:
        torch.save(generator.state_dict(), os.path.join(opt.save_dir, "generator.pth"))
        torch.save(discriminator.state_dict(), os.path.join(opt.save_dir, "discriminator.pth"))

import numpy as np
from torch.autograd import Variable
import torch
import cv2

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



def generate_kps(latent_dim, model):
    color_dict = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]
    color_idx = [4, 7, 10, 13, 16]

    def choose_color(coord_i):
        c_idx = 0
        while True:
            if coord_i <= color_idx[c_idx]:
                return color_dict[c_idx]
            else:
                c_idx += 1

    while True:
        z = Variable(Tensor(np.random.normal(0, 1, (2, latent_dim))))
        gen_kps = model(z)
        for gen_kp in gen_kps:
            image = np.zeros((200, 200, 3), dtype=np.uint8)
            float_single_coord = [x * 200 for x in gen_kp]
            # print(label)
            for i in range(17):
                x = int(float_single_coord[i * 2])
                y = int(float_single_coord[i * 2 + 1])
                cv2.circle(image, (x, y), 5, choose_color(i), -1)
        cv2.imshow('image', image)
        cv2.waitKey(0)

        # concated_imgs = []
        # for i in range(n_row):
        #     horizontal = np.concatenate(imgs[i*n_row:(i+1)*n_row], axis=1)
        #     concated_imgs.append(horizontal)
        # concated_imgs = np.concatenate(concated_imgs, axis=0)


if __name__ == '__main__':
    from models import Decoder
    latent_dim = 10
    model = Decoder(latent_dim)
    model.load_state_dict(torch.load('decoder.pth'))
    generate_kps(latent_dim, model)

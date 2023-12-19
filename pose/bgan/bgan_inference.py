import numpy as np
from torch.autograd import Variable
import torch
import cv2
import csv

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

decoder_path = "/media/hkuit164/Backup/PoseAndSceneGenerator/exp/bgan_test2/generator.pth"
csv_path = ""


def write_csv(csv_path, modified_array):
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(modified_array)



def generate_kps(latent_dim, model):
    color_dict = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]
    # color_idx = [4, 7, 10, 13, 16]
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

    # while True:
    for idx in range(50):
        z = Variable(Tensor(np.random.normal(0, 1, (2, latent_dim))))
        gen_kps = model(z)
        for gen_kp in gen_kps:
            # kps_list = []
            # kps_list.append(gen_kp.tolist() + ["4", "throw"])
            # write_csv(csv_path, kps_list)

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

        cv2.imshow('image', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    from models import Generator
    latent_dim = 10
    model = Generator(latent_dim)
    if cuda:
        model = model.cuda()
    model.load_state_dict(torch.load(decoder_path))
    generate_kps(latent_dim, model)

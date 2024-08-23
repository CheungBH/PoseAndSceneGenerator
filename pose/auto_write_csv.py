import csv
import torch
import numpy as np
from torch.autograd import Variable
import os

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

generators_folders = ""
csv_path = ""
generated_data_num = 100
feature_num = 34


def write_csv(csv_path, modified_array):
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(modified_array)


def generate_kps(latent_dim, model, generated_data_num):

    for idx in range(generated_data_num/2):
        z = Variable(Tensor(np.random.normal(0, 1, (2, latent_dim))))
        gen_kps = model(z)
        for gen_kp in gen_kps:
            kps_list = []
            kps_list.append(gen_kp.tolist() + ["4", "throw"])
            write_csv(csv_path, kps_list)

if __name__ == '__main__':

    generator_folders = os.listdir(generators_folders)
    for generator_folder in generator_folders:

        from models import Generator
        latent_dim = 10
        model = Generator(latent_dim, feature_num)
        if cuda:
            model = model.cuda()
        model.load_state_dict(torch.load(generator_path))
        generate_kps(latent_dim, model, generated_data_num)

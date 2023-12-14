import torch
import csv


class KPSDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, target="fight"):
        self.data = []
        file = csv.reader(open(data_path))
        for line in file:
            if line[-1] == target:
                self.data.append([float(x) for x in line[:-2]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return torch.tensor(x)
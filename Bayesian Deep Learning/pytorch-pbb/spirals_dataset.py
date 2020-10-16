import torch
from torch.utils.data import Dataset
import numpy as np
import math


class SpiralsDataset(Dataset):
    def __init__(self, n_samples, noise_std=0., rotations=1.):
        self.rotations = rotations
        self.noise_std = noise_std
        ts = torch.linspace(0, 1, n_samples)
        rs = ts ** 0.5
        thetas = rs * rotations * 2 * np.pi
        signs = torch.randint(2, (n_samples,)) * 2 - 1
        # signs = (torch.linspace(0, n_samples - 1, n_samples) % 2 == 0) * 2 - 1

        xs = rs * signs * torch.cos(thetas) + torch.randn(n_samples) * noise_std
        ys = rs * signs * torch.sin(thetas) + torch.randn(n_samples) * noise_std
        self.points = torch.stack([xs, ys], axis=1)
        self.labels = (signs > 0).long()
        self.classes = ['0', '1']

    def __getitem__(self, index):
        return self.points[index], self.labels[index]

    def __len__(self):
        return len(self.points)

    def classify(self, data):
        xs, ys = data[:, 0], data[:, 1]
        rs = (xs**2 + ys**2) ** 0.5 * self.rotations * 2 * np.pi
        thetas = torch.atan2(ys, xs) + 2 * np.pi * (ys < 0)

        ks = torch.zeros(len(xs))
        dists = torch.ones(len(xs)) * 10000
        for i, (r, theta) in enumerate(zip(rs, thetas)):
            for k in range(-1, math.ceil(2 * self.rotations + 1e-5)):
                dist = (k * np.pi + theta - r).abs()
                if dist < dists[i]:
                    dists[i] = dist
                    ks[i] = k
        signs = (ks % 2 == 0).long()
        return signs

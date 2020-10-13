# ------------------------------------------------------------------------------
# Torchvision requires the mean and standard deviation to be calculated manually 
# for normalization. This method can be used for that. However, this is mainly 
# for colored 2D images and therefore rarely relevant for medical data.
# ------------------------------------------------------------------------------

import torch

def normalization_values(dataset):
    r"""Compute normalization values for a dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    count = 0
    mean = torch.empty(3)
    std = torch.empty(3)

    for data, _ in dataloader:
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        mean = (count * mean + sum_) / (count + nb_pixels)
        std = (count * std + sum_of_square) / (count + nb_pixels)
        count += nb_pixels

    return {'mean': mean, 'std': torch.sqrt(std - mean ** 2)}
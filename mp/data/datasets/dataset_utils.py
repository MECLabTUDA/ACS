# ------------------------------------------------------------------------------
# Utils for binding datasets as Dataset subclasses.
# ------------------------------------------------------------------------------

import numpy as np
import torch
from mp.paths import original_data_paths

def get_original_data_path(global_name):
    r"""Get the original path from mp.paths. The global name is the key."""
    try:
        data_path = original_data_paths[global_name]
    except:
        raise Exception('Data path for {} must be set in paths.py.'.format(global_name))
    return data_path

def get_dataset_name(global_name, subset=None):
    r"""Get name of dataset by adding the global name to the subset."""
    if subset is None:
        return global_name
    else:
        name = global_name
        for key, value in subset.items():
            name += '[' + key+':'+ value + ']'
        return name

def get_mean_std_shape(instances):
    r"""Returns the mean sheap as (channels, width, heigth, depth) for a 
    list of instances.

    Args:
        instances (list[SegmentationInstance]): a list of segmentation instances.

    Returns (tuple[int]) tuple with form (channels, width, heigth, depth)
    """
    shapes = [np.array(instance.shape) for instance in instances]
    mean = np.mean(shapes, axis=0)
    std = np.std(shapes, axis=0)
    return tuple(int(x) for x in mean), tuple(int(x) for x in std)

def get_mask_labels(instances):
    r"""Returns a set of integer labels which appear in segmentation masks in 
    a list of instances.

    Args:
        instances (list[SegmentationInstance)): a  list of segmentation instances.

    Returns (list[str]): list of the form ['0', '1', '2', etc.] as replacement 
        for not having the real label names.
    """
    labels = set()
    for instance in instances:
        instance_labels = list(np.unique(instance.y.tensor.numpy()))
        assert all(x == int(x) for x in instance_labels), "Mask contain non-integer values"
        labels = labels.union([int(x) for x in instance_labels])
    return [str(nr) for nr in range(max(labels)+1)]

def check_correct_nr_labels(labels, instances):
    r"""Check that the number of label names manually supplied is consistent 
    with the dataset masks.
    """
    nr_labels = len(get_mask_labels(instances))
    print(nr_labels)
    assert nr_labels <= len(labels), "There are mask indexes not accounted for in the manually supplied label list"
    if nr_labels < len(labels):
        print('Warning: Some labels are not represented in the data')

def get_normalization_values(instances):
    r"""Get normalization values for a dataset."""
    count = 0
    mean = torch.empty(3)
    std = torch.empty(3)
    for instance in instances:
        c, w, h, d = instance.shape
        nb_pixels = d * h * w
        data = instance.x.tensor
        sum_ = torch.sum(data, dim=[1, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[1, 2, 3])
        mean = (count * mean + sum_) / (count + nb_pixels)
        std = (count * std + sum_of_square) / (count + nb_pixels)
        count += nb_pixels
    return {'mean': mean, 'std': torch.sqrt(std - mean ** 2)}
# ------------------------------------------------------------------------------
# Dataset class meant to store general information about dataset and to divide 
# instances in folds, before converting to a torch.utils.data.Dataset. 
# All datasets descend from this SegmentationDataset class.
# ------------------------------------------------------------------------------

import os
import sys
from mp.data.datasets.dataset import Dataset, Instance
import mp.data.datasets.dataset_utils as du
import torchio

class SegmentationInstance(Instance):
    def __init__(self, x_path, y_path, name=None, class_ix=0, group_id=None):
        """A segmentation instance, using the TorchIO library.
        Arguments:
        x_path (str): path to image
        y_path (str): path to segmentation mask
        name (str): name of instance for case-wise evaluation
        class_ix (int): optinal "class" index. During splitting of the dataset, 
            the resulting subsets are stratesfied according to this value (i.e. 
            there are about as many examples from each class in each fold
        of each class on each fold).
        group_id (comparable): instances with same group_id (e.g. patient id)
            are always in the same fold

        Note that torchio images have the shape (channels, w, h, d)
        """
        assert isinstance(x_path, str)
        assert isinstance(y_path, str)
        x = torchio.Image(x_path, type=torchio.INTENSITY)
        y = torchio.Image(y_path, type=torchio.LABEL)
        self.shape = x.shape
        super().__init__(x=x, y=y, name=name, class_ix=class_ix, 
            group_id=group_id)

    def get_subject(self):
        return torchio.Subject(
            x=self.x,
            y=self.y
        )

class SegmentationDataset(Dataset):
    """A Dataset for segmentation tasks, that specific datasets descend from.
    Arguments:
    instances (list(SegmentationInstance)): a list of instances
    name (str): the dataset name
    mean_shape (tuple(int)): the mean input shape of the data, or None
    label_names (list(str)): list with label names, or None
    nr_channels int: number channels
    modality: modality of the data, e.g. MR, CT
    hold_out_ixs (list(int)): list of instance index to reserve for a separate 
        hold-out dataset
    """
    def __init__(self, instances, name, mean_shape=None, 
    label_names=None, nr_channels=1, modality='unknown', hold_out_ixs=[],
    check_correct_nr_labels=False):
        # Set mean input shape and mask labels, if these are not provided
        print('\nDATASET: {} with {} instances'.format(name, len(instances)))
        if mean_shape is None:
            mean_shape, shape_std = du.get_mean_std_shape(instances)
            print('Mean shape: {}, shape std: {}'.format(mean_shape, shape_std))
        if label_names is None:
            label_names = du.get_mask_labels(instances)
        else:
            if check_correct_nr_labels:
                du.check_correct_nr_labels(label_names, instances)
        print('Mask labels: {}\n'.format(label_names))
        self.mean_shape = mean_shape
        self.label_names = label_names
        self.nr_labels = len(label_names)
        self.modality = modality
        super().__init__(name=name, instances=instances, 
            mean_shape=mean_shape, output_shape=mean_shape, 
            hold_out_ixs=hold_out_ixs)


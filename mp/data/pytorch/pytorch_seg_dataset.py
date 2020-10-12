# ------------------------------------------------------------------------------
# From an mp.data.datasets.dataset_segmentation.SegmentationDataset, create a 
# mp.data.pytorch.PytorchDataset. There are different types of datasets:
#
# PytorchSeg2DDataset: the length of the dataset is the total number of slices 
# (forth dimension) in the data base. A resized slice is returned by __getitem__
#
# PytorchSeg3DDataset: __getitem__ returnes the next instance, resized to the
# specified size. The length is the number of instances.
#
# Pytorch3DQueue: 3D patches are sampled randomly from the entire dataset.
# Receives a torchio.data.Sampler. Is built on top of torchio.data.Queue
# See https://torchio.readthedocs.io/data/patch_training.html
# ------------------------------------------------------------------------------

import copy
import torch
import torchio
from mp.data.pytorch.pytorch_dataset import PytorchDataset
import mp.data.pytorch.transformation as trans
import mp.eval.inference.predictor as pred

class PytorchSegmnetationDataset(PytorchDataset):
    def __init__(self, dataset, ix_lst=None, size=None, norm_key='rescaling', 
        aug_key='standard', channel_labels=True):
        r"""A torch.utils.data.Dataset for segmentation data.
        Args:
            dataset (SegmentationDataset): a SegmentationDataset
            ix_lst (list[int)]): list specifying the instances of the dataset. 
                If 'None', all not in the hold-out dataset are incuded.
            size (tuple[int]): size as (channels, width, height, Opt(depth))
            norm_key (str): Normalization strategy, from 
                mp.data.pytorch.transformation
            aug_key (str): Augmentation strategy, from 
                mp.data.pytorch.transformation
            channel_labels (bool): if True, the output has one channel per label
        """
        super().__init__(dataset=dataset, ix_lst=ix_lst, size=size)
        self.norm = trans.NORMALIZATION_STRATEGIES[norm_key]
        self.aug = trans.AUGMENTATION_STRATEGIES[aug_key]
        self.nr_labels = dataset.nr_labels
        self.channel_labels = channel_labels
        self.predictor = None

    def get_instance(self, ix=None, name=None):
        r"""Get a particular instance from the ix or name"""
        assert ix is None or name is None
        if ix is None:
            instance = [ex for ex in self.instances if ex.name == name]
            assert len(instance) == 1
            return instance[0]
        else:
            return self.instances[ix]

    def get_ix_from_name(self, name):
        r"""Get ix from name"""
        return next(ix for ix, ex in enumerate(self.instances) if ex.name == name)

    def transform_subject(self, subject):
        r"""Tranform a subject by applying normalization and augmentation ops"""
        if self.norm is not None:
            subject = self.norm(subject)
        if self.aug is not None:
            subject = self.aug(subject)
        return subject

    def get_subject_dataloader(self, subject_ix):
        r"""Get a list of input/target pairs equivalent to those if the dataset
        was only of subject with index subject_ix. For evaluation purposes.
        """
        raise NotImplementedError

class PytorchSeg2DDataset(PytorchSegmnetationDataset):
    r"""Divides images into 2D slices. If resize=True, the slices are resized to
    the specified size, otherwise they are center-cropped and padded if needed.
    """
    def __init__(self, dataset, ix_lst=None, size=(1, 256, 256), 
        norm_key='rescaling', aug_key='standard', channel_labels=True, resize=False):
        if isinstance(size, int):
            size = (1, size, size)
        super().__init__(dataset=dataset, ix_lst=ix_lst, size=size, 
            norm_key=norm_key, aug_key=aug_key, channel_labels=channel_labels)
        assert len(self.size)==3, "Size should be 2D"
        self.resize = resize
        self.predictor = pred.Predictor2D(self.instances, size=self.size, 
            norm=self.norm, resize=resize)

        self.idxs = []
        for instance_ix, instance in enumerate(self.instances):
            for slide_ix in range(instance.shape[-1]):
                self.idxs.append((instance_ix, slide_ix))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        r"""Returns x and y values each with shape (c, w, h)"""
        instance_idx, slice_idx = self.idxs[idx]

        subject = copy.deepcopy(self.instances[instance_idx].get_subject())
        subject.load()

        subject = self.transform_subject(subject)

        x = subject.x.tensor.permute(3, 0, 1, 2)[slice_idx]
        y = subject.y.tensor.permute(3, 0, 1, 2)[slice_idx]

        if self.resize:
            x = trans.resize_2d(x, size=self.size)
            y = trans.resize_2d(y, size=self.size, label=True)
        else:
            x = trans.centre_crop_pad_2d(x, size=self.size)
            y = trans.centre_crop_pad_2d(y, size=self.size)

        if self.channel_labels:
            y = trans.per_label_channel(y, self.nr_labels)

        return x, y

    def get_subject_dataloader(self, subject_ix):
        dl_items = []
        idxs = [idx for idx, (instance_idx, slice_idx) in enumerate(self.idxs) 
            if instance_idx==subject_ix]
        for idx in idxs:
            x, y = self.__getitem__(idx)
            dl_items.append((x.unsqueeze_(0), y.unsqueeze_(0)))
        return dl_items

class PytorchSeg3DDataset(PytorchSegmnetationDataset):
    r"""Each 3D image is an item in the dataloader. If resize=True, the volumes
    are resized to the specified size, otherwise they are center-cropped and 
    padded if needed.
    """
    def __init__(self, dataset, ix_lst=None, size=(1, 56, 56, 10), 
        norm_key='rescaling', aug_key='standard', channel_labels=True, resize=False):
        if isinstance(size, int):
            size = (1, size, size, size)
        super().__init__(dataset=dataset, ix_lst=ix_lst, size=size, 
            norm_key=norm_key, aug_key=aug_key, channel_labels=channel_labels)
        assert len(self.size)==4, "Size should be 3D"
        self.resize=resize
        self.predictor = pred.Predictor3D(self.instances, size=self.size, 
            norm=self.norm, resize=resize)

    def __getitem__(self, idx):
        r"""Returns x and y values each with shape (c, w, h, d)"""

        subject = copy.deepcopy(self.instances[idx].get_subject())
        subject.load()

        subject = self.transform_subject(subject)

        x = subject['x'].data
        y = subject['y'].data

        if self.resize:
            x = trans.resize_3d(x, size=self.size)
            y = trans.resize_3d(y, size=self.size, label=True)
        else:
            x = trans.centre_crop_pad_3d(x, size=self.size)
            y = trans.centre_crop_pad_3d(y, size=self.size)

        if self.channel_labels:
            y = trans.per_label_channel(y, self.nr_labels)

        return x, y

    def get_subject_dataloader(self, subject_ix):
        x, y = self.__getitem__(subject_ix)
        return [(x.unsqueeze_(0), y.unsqueeze_(0))]

class Pytorch3DQueue(PytorchSegmnetationDataset):
    r"""Divides images into patches. If there are subjects with less depth than 
    self.size[-1], these are padded.
    """
    def __init__(self, dataset, ix_lst=None, size=(1, 56, 56, 10), sampler=None,
        max_length=300, samples_per_volume=10, norm_key='rescaling', 
        aug_key='standard', channel_labels=True):
        r"""The number of patches is determined by samples_per_volume """
        if isinstance(size, int):
            size = (1, size, size, size)
        super().__init__(dataset=dataset, ix_lst=ix_lst, size=size, 
            norm_key=norm_key, aug_key=aug_key, channel_labels=channel_labels)
        assert len(self.size)==4, "Size should be 3D"
        self.predictor = pred.GridPredictor(self.instances, size=self.size, norm=self.norm)

        # If there are subjects with less depth than self.size[-1], pad
        self.instances = [trans.pad_3d_if_required(instance, self.size) 
            for instance in self.instances]
        # Create an instance of torchio.data.SubjectsDataset
        subjects_dataset = torchio.data.SubjectsDataset(
            [instance.get_subject() for instance in self.instances])

        # Set Sampler
        if sampler is None:
            sampler = torchio.data.UniformSampler(self.size[1:])

        # Initialize queue
        self.queue = torchio.Queue(
            subjects_dataset,
            sampler=sampler,
            max_length=max_length,
            samples_per_volume=samples_per_volume,
            num_workers=0,
            shuffle_subjects=True,
            shuffle_patches=True,
            )

    def __len__(self):
        return self.queue.__len__()

    def __getitem__(self, idx):
        r"""Returns x and y values each with shape (c, w, h, d)
        Class torchio.Queue descends from torch.utils.data.Dataset."""

        subject = self.queue.__getitem__(idx)
        subject.load()

        subject = self.transform_subject(subject)

        x = subject['x'].data
        y = subject['y'].data

        if self.channel_labels:
            y = trans.per_label_channel(y, self.nr_labels)

        return x, y

    def get_subject_dataloader(self, subject_ix):

        subject = copy.deepcopy(self.instances[subject_ix].get_subject())
        subject.load()
        subject = self.transform_subject(subject)

        grid_sampler = torchio.inference.GridSampler(
            sample=subject,
            patch_size=self.size[1:],
            patch_overlap=(0,0,0))

        dl_items = []
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
        for patches_batch in patch_loader:            
            input_tensor = patches_batch['x'][torchio.DATA]
            target_tensor = patches_batch['y'][torchio.DATA]
            dl_items.append((input_tensor, target_tensor))
        return dl_items

# ------------------------------------------------------------------------------
# Tensor transformations. Mainly, transformations from the TorchIO library are 
# used (https://torchio.readthedocs.io/transforms).
# ------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import torchio

NORMALIZATION_STRATEGIES = {None:None,
    'rescaling': torchio.transforms.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.1, 99.)),
    'z_norm': torchio.transforms.ZNormalization(masking_method=None)
    # TODO
    #'histogram_norm': torchio.transforms.HistogramStandardization(landmarks)
}

AUGMENTATION_STRATEGIES = {'none':None,
    'standard': torchio.transforms.Compose([
        torchio.transforms.OneOf({
            torchio.transforms.RandomElasticDeformation(p=0.1,
                num_control_points=(7,7,7), 
                max_displacement=7.5): 0.7,
            torchio.RandomAffine(p=0.1,
                scales=(0.5, 1.5),
                degrees=(5),
                isotropic=False,
                default_pad_value='otsu',
                image_interpolation='bspline'): 0.1
        }),
        torchio.transforms.RandomFlip(p=0.1, 
            axes=(1, 0, 0)),
        torchio.transforms.RandomMotion(p=0.1,
            degrees=10, 
            translation=10, 
            num_transforms=2),
        torchio.transforms.RandomBiasField(p=0.1,
            coefficients=(0.5, 0.5), 
            order=3),
        torchio.transforms.RandomNoise(p=0.1,
            mean=(0,0), 
            std=(50, 50)),
        torchio.transforms.RandomBlur(p=0.1,
            std=(0, 1))
    ])
}

def per_label_channel(y, nr_labels, channel_dim=0, device='cpu'):
    r"""Trans. a one-channeled mask where the integers specify the label to a 
    multi-channel output with one channel per label, where 1 marks belonging to
    that label."""
    masks = []
    zeros = torch.zeros(y.shape, dtype=torch.float64).to(device)
    ones = torch.ones(y.shape, dtype=torch.float64).to(device)
    for label_nr in range(nr_labels):
        mask = torch.where(y == label_nr, ones, zeros)
        masks.append(mask)
    target = torch.cat(masks, dim=channel_dim)
    return target

def _one_output_channel_single(y):
    r"""Helper function."""
    channel_dim = 0
    target_shape = list(y.shape)
    nr_labels = target_shape[channel_dim]
    target_shape[channel_dim] = 1
    target = torch.zeros(target_shape, dtype=torch.float64)
    label_nr_mask = torch.zeros(target_shape, dtype=torch.float64)
    for label_nr in range(nr_labels):
        label_nr_mask.fill_(label_nr)
        target = torch.where(y[label_nr] == 1, label_nr_mask, target)
    return target

def one_output_channel(y, channel_dim=0):
    r"""Inverses the operation of 'per_label_channel'. The output is 
    one-channelled. It is stricter than making a prediction because the content 
    must be 1 and not the largest float."""
    if channel_dim == 0:
        return _one_output_channel_single(y)
    else:
        assert channel_dim == 1, "Not implemented for channel_dim > 1"
    batch = [_one_output_channel_single(x) for x in y]
    return torch.stack(batch, dim=0)

def resize_2d(img, size=(1, 128, 128), label=False):
    r"""2D resize."""
    img.unsqueeze_(0) # Add additional batch dimension so input is 4D
    if label:
        # Interpolation in 'nearest' mode leaves the original mask values.
        img = F.interpolate(img, size=size[1:], mode='nearest')
    else:
        img = F.interpolate(img, size=size[1:], mode='bilinear')
    return img[0]

def resize_3d(img, size=(1, 56, 56, 56), label=False):
    r"""3D resize."""
    img.unsqueeze_(0) # Add additional batch dimension so input is 5D
    if label:
        # Interpolation in 'nearest' mode leaves the original mask values.
        img = F.interpolate(img, size=size[1:], mode='nearest')
    else:
        img = F.interpolate(img, size=size[1:], mode='trilinear')
    return img[0]

def centre_crop_pad_2d(img, size=(1, 128, 128)):
    r"""Center-crops to the specified size, unless the image is to small in some
    dimension, then padding takes place.
    """
    img = torch.unsqueeze(img, -1)
    size = (size[1], size[2], 1)
    transform = torchio.transforms.CropOrPad(target_shape=size, padding_mode=0)
    device = img.device
    img = transform(img.cpu()).to(device)
    img = torch.squeeze(img, -1)
    return img

def centre_crop_pad_3d(img, size=(1, 56, 56, 56)):
    r"""Center-crops to the specified size, unless the image is to small in some
    dimension, then padding takes place. For 3D data.
    """
    transform = torchio.transforms.CropOrPad(target_shape=size[1:], padding_mode=0)
    device = img.device
    img = transform(img.cpu()).to(device)
    return img

def pad_3d_if_required(instance, size):
    r"""Pads if required in the last dimension, for 3D.
    """
    if instance.shape[-1] < size[-1]:
        delta = size[-1]-instance.shape[-1]
        subject = instance.get_subject()
        transform = torchio.transforms.Pad(padding=(0, 0, 0, 0, 0, delta), padding_mode = 0)
        subject = transform(subject)
        instance.x = torchio.Image(tensor=subject.x.tensor, type=torchio.INTENSITY)
        instance.y = torchio.Image(tensor=subject.y.tensor, type=torchio.LABEL)
        instance.shape = subject.shape
    return instance

from torchvision import transforms

def torchvision_rescaling(x, size=(3, 224, 224), resize=False):
    r"""To use pretrained torchvision models, three-channeled 2D images must 
    first be normalized between 0 and 1 and then noralized with predfined values
    (see https://pytorch.org/docs/stable/torchvision/models.html)
    """
    device = x.device
    # Images should be normalized between 0 and 1
    assert torch.min(x) >= 0.
    assert torch.max(x) <= 1.
    transform_ops = []
    # If images are one-channeled, triplicate
    if x.shape[1] == 1:
        transform_ops.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    # Resize or crop
    transform_ops.append(transforms.ToPILImage())
    if resize:
        transform_ops.append(transforms.Resize(size=(size[1], size[2])))
    else:
        transform_ops.append(transforms.CenterCrop(size=(size[1], size[2])))   
    # Apply pre-defined normalization 
    transform_ops.append(transforms.ToTensor())
    transform_ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
    # Apply transform operation
    transform = transforms.Compose(transform_ops)
    imgs = []
    for img in x:
        imgs.append(transform(img.cpu()).to(device))
    return torch.stack(imgs, dim=0)
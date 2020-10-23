# ------------------------------------------------------------------------------
# Visualize images (in Numpy, PyTorch or TorchIO formats) and dataloaders.
# ------------------------------------------------------------------------------

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk
import math
import random
from PIL import Image
from mp.data.pytorch.transformation import one_output_channel

def img_to_numpy_array(x):
    r"""Transform an image in several formats into a numpy array."""
    type_str = str(type(x))
    if 'SimpleITK.SimpleITK.Image' in type_str:
        return sitk.GetArrayFromImage(x)
    elif 'torchio.data.image.Image' in type_str:
        return x.tensor.numpy()
    elif 'torch.Tensor' in type_str:
        return x.detach().cpu().numpy()
    elif 'numpy.ndarray' in type_str:
        return x
    else:
        raise NotImplementedError

def ensure_channel_width_height_depth(np_array):
    r"""Ensure the dimensions go channels, width, height(, depth)"""
    # TODO keep width and height in initial dimensions, right now largest first
    axis_order = np.argsort(np_array.shape)
    if len(np_array.shape) == 3:
        np_array = np_array.transpose(
            axis_order[0], axis_order[2], axis_order[1])
    if len(np_array.shape) == 4:
        np_array = np_array.transpose(
            axis_order[0], axis_order[3], axis_order[2], axis_order[1])
    return np_array

def normalize_range(img_array, max_value=255.):
    r"""Normalize in range [0, 255]"""
    img_array /= (img_array.max()/max_value)
    return img_array.astype(np.uint8)

def overlay_images(base, overlay, alpha=0.5):
    r"""Add transparency to mask, and make composition of image overlayed by 
    transparent mask.
    """
    alpha = int(255*alpha)
    overlay.putalpha(alpha)
    return Image.alpha_composite(base, overlay)

def stretch_mask_range(mask):
    r"""Stretches the range of mask values to [0, 255] so that they are 
    differentiable, and converts to RGBA PIL Image.
    """
    if mask.max() != 0:
        mask *= (255.0/mask.max())
        mask = mask.astype(np.uint8)

segmask_colors = {1: {'red': 206, 'green': 24, 'blue': 30}, # Red
    2: {'red': 64, 'green': 201, 'blue': 204}, # Mint
    3: {'red': 250, 'green': 245, 'blue': 56}, # Yellow
    4: {'red': 193, 'green': 69, 'blue': 172}, # Purple
    5: {'red': 54, 'green': 71, 'blue': 217} # Blue
    }

def color_mask(mask):
    r"""Converts a mask with integer values that are typically < 5 to an RGBA
    PIL image which each integer is a differentiable color.
    """
    mask = mask.astype(np.uint8)
    mask = np.stack((mask,)*3, axis=-1)
    red, green, blue = mask.T
    for seg_value, new_color in segmask_colors.items():
        to_replace = (red == seg_value) & (blue == seg_value) & (green == seg_value)
        red[to_replace] = new_color['red']
        green[to_replace] = new_color['green']
        blue[to_replace] = new_color['blue']
    mask = np.array([red, green, blue]).T
    return mask

### Visualize images, masks and dataloaders using Pillow ###
    
def plot_3d_subject_gt(subject, save_path=None):
    r"""Plot a subject with input and ground truth"""
    inputs = subject['x'].data
    targets = subject['y'].data
    plot_3d_segmentation(inputs, targets, save_path=save_path)

def plot_3d_subject_pred(subject, pred, save_path=None):
    r"""Plot a subject with input and prediction"""
    inputs = subject['x'].data
    assert pred.shape == subject['y'].data.shape, "Prediction has the wrong size."
    plot_3d_segmentation(inputs, pred, save_path=save_path)

def plot_3d_img(img, save_path=None, img_size=(512, 512)):
    r"""Visualize a 3D image."""
    img = img_to_numpy_array(img)
    if len(img.shape) == 3:
        # Add channel dimension
        img = np.expand_dims(img, axis=0)
    # Ensure (channel, width, height, depth)
    img = ensure_channel_width_height_depth(img)
    assert len(img.shape) == 4 and int(img.shape[0]) == 1
    # Rotate axis to (depth, 1, width, height) from (1, width, height, depth)
    img = np.moveaxis(img, -1, 0)
    # Create 2D image list
    imgs = []
    for ix in range(len(img)):
        imgs.append(img[ix])
    grid_side = int(math.ceil(math.sqrt(len(imgs))))
    img_grid = get_img_grid(imgs, grid_side, grid_side)
    create_img_grid(
        img_grid=img_grid, save_path=save_path, img_size=img_size)

def plot_3d_segmentation(
        img, segmentation, save_path=None, img_size=(512, 512), alpha=0.5):
    r"""Visualize a 3D image with coresponding segmentation."""
    img = img_to_numpy_array(img)
    segmentation = img_to_numpy_array(segmentation)
    assert img.shape == segmentation.shape
    if len(img.shape) == 3:
        # Add channel dimension
        img = np.expand_dims(img, axis=0)
        segmentation = np.expand_dims(segmentation, axis=0)
    # Ensure (channel, width, height, depth)
    img = ensure_channel_width_height_depth(img)
    segmentation = ensure_channel_width_height_depth(segmentation)
    assert len(img.shape) == 4 and int(img.shape[0]) == 1
    # Rotate axis to (depth, 1, width, height) from (1, width, height, depth)
    img = np.moveaxis(img, -1, 0)
    segmentation = np.moveaxis(segmentation, -1, 0)
    # Create 2D image list
    imgs = []
    for ix in range(len(img)):
        imgs.append((img[ix], segmentation[ix]))
    grid_side = int(math.ceil(math.sqrt(len(imgs))))
    img_grid = get_img_grid(imgs, grid_side, grid_side)
    create_x_y_grid(
        img_grid=img_grid, save_path=save_path, img_size=img_size, alpha=alpha)

def get_img_grid(img_list, nr_rows, nr_cols, randomize=False):
    r"""Place list items in a gris format."""
    if randomize:
        random.shuffle(img_list)
    img_grid = [[None for i in range(nr_cols)] for j in range(nr_rows)]
    for j in range(nr_rows):
        for i in range(nr_cols):
            if i+j*nr_cols < len(img_list):
                img_grid[j][i] = img_list[i+j*nr_cols]
    return img_grid

def create_img_grid(img_grid = [[]], img_size = (512, 512), 
    margin = (5, 5), background_color = (255, 255, 255, 255), save_path=None):
    r"""Visualize a grid with 2d image slices, overlayed with masks."""
    bg_width = len(img_grid[0])*img_size[0] + (len(img_grid[0])+1)*margin[0]
    bg_height = len(img_grid)*img_size[1] + (len(img_grid)+1)*margin[1]
    new_img = Image.new('RGBA', (bg_width, bg_height), background_color)
    left = margin[0]
    top = margin[1]
    for row in img_grid:
        for img in row:
            if img is not None:
                if img.shape[0]==1: # Grayscale images
                    img = img[0]
                else: # Colored images
                    if np.argpartition(img.shape, 1)[0] == 0: # Channels first
                        img = np.moveaxis(img, 0, 2) 
                img = normalize_range(img)
                img = Image.fromarray(img).resize(img_size).convert('RGBA')
                new_img.paste(img, (left, top))
                left += img_size[0] + margin[0]
        top += img_size[1] + margin[1]
        left = margin[0]
    if save_path is None:
        new_img.show()
    else:
        new_img.save(save_path)

def create_x_y_grid(img_grid = [[]], img_size = (512, 512), alpha=0.5,
    margin = (5, 5), background_color = (255, 255, 255, 255), save_path=None):
    r"""Visualize a grid with 2d image slices, overlayed with masks."""
    bg_width = len(img_grid[0])*img_size[0] + (len(img_grid[0])+1)*margin[0]
    bg_height = len(img_grid)*img_size[1] + (len(img_grid)+1)*margin[1]
    new_img = Image.new('RGBA', (bg_width, bg_height), background_color)
    left = margin[0]
    top = margin[1]
    for row in img_grid:
        for img_mask_pair in row:
            if img_mask_pair is not None: # Is None if grid to large for img nr.
                img, mask = img_mask_pair
                if img.shape[0]==1: # Grayscale images
                    img = img[0]
                    # Normalize image values between 0 and 255
                    img = normalize_range(img)
                    img = Image.fromarray(img).resize(img_size).convert('RGBA')
                    # Stretch the mask values between 0 and 255
                    mask = mask[0]
                    mask = color_mask(mask)
                    Image.fromarray(mask)
                    Image.fromarray(mask).resize(img_size)
                    mask = Image.fromarray(mask).resize(img_size).convert('RGBA')
                else: # Colored images
                    if np.argpartition(img.shape, 1)[0] == 0: # If channels first
                        img = np.moveaxis(img, 0, 2) 
                        mask = np.moveaxis(mask, 0, 2) 
                    img = normalize_range(img)
                    img = Image.fromarray(
                        (img).astype(np.uint8)).resize(img_size).convert('RGBA')
                    mask = Image.fromarray(
                        (mask).astype(np.uint8)).resize(img_size).convert('RGBA')
                # Overlay images
                x_y_img = overlay_images(img, mask, alpha=alpha)
                # Paste into original image
                new_img.paste(x_y_img, (left, top))
                left += img_size[0] + margin[0]
        top += img_size[1] + margin[1]
        left = margin[0]
    if save_path is None:
        new_img.show()
    else:
        new_img.save(save_path)

def visualize_dataloader(
    dataloader, max_nr_imgs=100, save_path=None, img_size=(256, 256)):
    r"""Visualize images (inputs) from dataloader."""
    imgs = get_imgs_from_dataloader(dataloader, max_nr_imgs)
    grid_side = int(math.ceil(math.sqrt(len(imgs))))
    img_grid = get_img_grid(imgs, grid_side, grid_side)
    create_img_grid(img_grid=img_grid, save_path=save_path, img_size=img_size)

def get_imgs_from_dataloader(dataloader, nr_imgs):
    r"""Get images (inputs) from dataloader and place in list."""
    imgs = []
    for x, y in dataloader:
        x = x.cpu().detach().numpy()
        for img in x:
            if len(imgs) < nr_imgs:
                imgs.append(img)
        if len(imgs) == nr_imgs:
            break  
    return imgs

def visualize_dataloader_with_masks(dataloader, max_nr_imgs=100, save_path=None, 
    img_size=(256, 256), alpha=0.5):
    r"""Visualize images and masks from dataloader."""
    imgs = get_x_y_from_dataloader(dataloader, max_nr_imgs)
    grid_side = int(math.ceil(math.sqrt(len(imgs))))
    img_grid = get_img_grid(imgs, grid_side, grid_side)
    create_x_y_grid(
        img_grid=img_grid, save_path=save_path, img_size=img_size, alpha=alpha)

def get_x_y_from_dataloader(dataloader, nr_imgs):
    r"""Get images and masks from dataloader and place in list."""
    imgs = []
    for x, y in dataloader:
        x = x.cpu().detach().numpy()

        # If one channel per label, transform into one mask
        if y.shape[1] > 1:
            y = one_output_channel(y, channel_dim=1)

        y = y.cpu().detach().numpy()

        if len(x.shape) == 5: # If each x or y is a batch of volumes
            # Go from shape (batch, 1, width, height, depth) to 
            # (batch*depth, 1, width, height) by shifting the depth channel to
            # the beginning and concatenating all volumes.
            x_batch = [np.moveaxis(volume_x, -1, 0) for volume_x in x]
            y_batch = [np.moveaxis(volume_y, -1, 0) for volume_y in y]
            x = np.concatenate(x_batch)
            y = np.concatenate(y_batch)
        assert len(x.shape) == 4
        for ix, img in enumerate(x):
            if len(imgs) < nr_imgs:
                imgs.append((img, y[ix]))
        if len(imgs) == nr_imgs:
            break  
    return imgs

### Visualize using matplotlib, deprecated ###

def plot_overlay_mask(img, mask, save_path=None, figsize=(20, 20)):
    r"""
    Compare two 2d imgs, one on top of the other.
    TODO: background takes on blue tones.
    """
    if 'torch.Tensor' in str(type(img)):
        img, mask = img.cpu().detach().numpy(), mask.cpu().detach().numpy()
        while len(img.shape) > 2:
            img, mask = img[0], mask[0]
    assert img.shape == mask.shape
    plt.figure(figsize=figsize, frameon=False)
    plt.imshow(img, 'gray'), plt.axis('off')
    plt.imshow(mask, 'jet', alpha=0.7), plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_2d_img(img, save_path=None, figsize=(20, 20)):
    r"""Plot a 2d image"""
    if 'torch.Tensor' in  str(type(img)):
        img = img.cpu().detach().numpy()
    while img.shape[0] == 1:
        img = img[0]
    if len(img.shape) == 3:
        # If channels first, rotate so channels last
        if np.argpartition(img.shape, 1)[0] == 0:
            img = np.moveaxis(img, 0, 2)
    # Plot
    plt.figure(figsize=figsize, frameon=False)
    plt.imshow(img), plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

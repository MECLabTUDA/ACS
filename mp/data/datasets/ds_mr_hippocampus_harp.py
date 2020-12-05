# ------------------------------------------------------------------------------
# Hippocampus segmentation task for the HarP dataset
# (http://www.hippocampal-protocol.net/SOPs/index.php)
# ------------------------------------------------------------------------------

import os
import re

import SimpleITK as sitk
import nibabel as nib
import numpy as np

import mp.data.datasets.dataset_utils as du
from mp.data.datasets.dataset_segmentation import SegmentationDataset, SegmentationInstance
from mp.paths import storage_data_path
from mp.utils.load_restore import join_path


class HarP(SegmentationDataset):
    r"""Class for the segmentation of the HarP dataset,
    found at http://www.hippocampal-protocol.net/SOPs/index.php
    with the masks as .nii files and the scans as .mnc files.
    """

    def __init__(self, subset=None, hold_out_ixs=None, merge_labels=True):
        # Part is either: "Training", "Validation" or "All"
        default = {"Part": "All"}
        if subset is not None:
            default.update(subset)
            subset = default
        else:
            subset = default

        if hold_out_ixs is None:
            hold_out_ixs = []

        global_name = 'HarP'
        name = du.get_dataset_name(global_name, subset)
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)

        # Build instances
        instances = []
        folders = []
        if subset["Part"] in ["Training", "All"]:
            folders.append(("100", "Training"))
        if subset["Part"] in ["Validation", "All"]:
            folders.append(("35", "Validation"))

        for orig_folder, dst_folder in folders:
            # Paths with the sub-folder for the current subset
            dst_folder_path = os.path.join(dataset_path, dst_folder)

            # Copy the images if not done already
            if not os.path.isdir(dst_folder_path):
                _extract_images(original_data_path, dst_folder_path, orig_folder)

            # Fetch all patient/study names
            study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name
                              in os.listdir(os.path.join(dataset_path, dst_folder)))

            for study_name in study_names:
                instances.append(SegmentationInstance(
                    x_path=os.path.join(dataset_path, dst_folder, study_name + '.nii.gz'),
                    y_path=os.path.join(dataset_path, dst_folder, study_name + '_gt.nii.gz'),
                    name=study_name,
                    group_id=None
                ))

        label_names = ['background', 'hippocampus']

        super().__init__(instances, name=name, label_names=label_names,
                         modality='T1w MRI', nr_channels=1, hold_out_ixs=hold_out_ixs)


def _extract_images(source_path, target_path, subset):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """

    def bbox_3D(img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax

    # Folder 100 is for training (100 subjects), 35 subjects are left over for validation
    affine = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

    images_path = os.path.join(source_path, subset)
    labels_path = os.path.join(source_path, f'Labels_{subset}_NIFTI')

    # Create directories
    os.makedirs(os.path.join(target_path))

    # For each MRI, there are 2 segmentation (left and right hippocampus)
    for filename in os.listdir(images_path):
        # Loading the .mnc file and converting it to a .nii.gz file
        minc = nib.load(os.path.join(images_path, filename))
        x = nib.Nifti1Image(minc.get_data(), affine=affine)

        # We need to recover the study name of the image name to construct the name of the segmentation files
        match = re.match(r"ADNI_[0-9]+_S_[0-9]+_[0-9]+", filename)
        if match is None:
            raise Exception(f"A file ({filename}) does not match the expected file naming format")

        # For each side of the brain
        for side in ["_L", "_R"]:
            study_name = match[0] + side

            y = sitk.ReadImage(os.path.join(labels_path, study_name + ".nii"))
            y = sitk.GetArrayFromImage(y)

            # Shape expected: (189, 233, 197)
            # Average label shape (Training): (27.1, 36.7, 22.0)
            # Average label shape (Validation): (27.7, 35.2, 21.8)
            assert x.shape == y.shape
            # Disclaimer: next part is ugly and not many checks are made
            # BUGFIX: Some segmentation have some weird values eg {26896.988, 26897.988} instead of {0, 1}
            y = (y - np.min(y.flat)).astype(np.uint32)

            # So we first compute the bounding box
            rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(y)

            # Compute the start idx for each dim
            dr = (rmax - rmin) // 4
            dc = (cmax - cmin) // 4
            dz = (zmax - zmin) // 4

            # Reshaping
            y = y[rmin - dr: rmax + dr,
                cmin - dc: cmax + dc,
                zmin - dz: zmax + dz]

            x_cropped = x.get_data()[rmin - dr: rmax + dr,
                        cmin - dc: cmax + dc,
                        zmin - dz: zmax + dz]

            # Save new images so they can be loaded directly
            sitk.WriteImage(sitk.GetImageFromArray(y),
                            join_path([target_path, study_name + "_gt.nii.gz"]))
            sitk.WriteImage(sitk.GetImageFromArray(x_cropped),
                            join_path([target_path, study_name + ".nii.gz"]))

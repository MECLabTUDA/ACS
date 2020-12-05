# ------------------------------------------------------------------------------
# Hippocampus segmentation published by Dryad
# (https://datadryad.org/stash/dataset/doi:10.5061/dryad.gc72v)
# ------------------------------------------------------------------------------

import os

import SimpleITK as sitk

import mp.data.datasets.dataset_utils as du
from mp.data.datasets.dataset_segmentation import SegmentationDataset, SegmentationInstance
from mp.paths import storage_data_path
from mp.utils.load_restore import join_path
import re
import nibabel as nib
import numpy as np
import re


class DryadHippocampus(SegmentationDataset):
    r"""Class for the segmentation of the HarP dataset,
    https://datadryad.org/stash/dataset/doi:10.5061/dryad.gc72v.
    """

    def __init__(self, subset=None, hold_out_ixs=None, merge_labels=True):
        # Modality is either: "T1w" or "T2w"
        # Resolution is either: "Standard" or "Hires"
        # If you want to use different resolutions or modalities, please create another object with a different subset
        default = {"Modality": "T1w", "Resolution": "Standard"}
        if subset is not None:
            default.update(subset)
            subset = default
        else:
            subset = default

        # Hires T2w is not available
        assert not (subset["Resolution"] == "Standard" and subset["Modality"] == "T2w"), \
            "Hires T2w not available for the Dryad Hippocampus dataset"

        if hold_out_ixs is None:
            hold_out_ixs = []

        global_name = 'DryadHippocampus'
        name = du.get_dataset_name(global_name, subset)
        dataset_path = os.path.join(storage_data_path,
                                    global_name,
                                    "Merged Labels" if merge_labels else "Original",
                                    "".join([f"{key}[{subset[key]}]" for key in ["Modality", "Resolution"]])
                                    )
        original_data_path = du.get_original_data_path(global_name)

        # Copy the images if not done already
        if not os.path.isdir(dataset_path):
            _extract_images(original_data_path, dataset_path, merge_labels, subset)

        # Fetch all patient/study names
        study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name in os.listdir(dataset_path))

        # Build instances
        instances = []
        for study_name in study_names:
            instances.append(SegmentationInstance(
                x_path=os.path.join(dataset_path, study_name + '.nii.gz'),
                y_path=os.path.join(dataset_path, study_name + '_gt.nii.gz'),
                name=study_name,
                group_id=None
            ))

        if merge_labels:
            label_names = ['background', 'hippocampus']
        else:
            label_names = ['background', 'subiculum', 'CA1-3', 'CA4-DG']

        super().__init__(instances, name=name, label_names=label_names,
                         modality=subset["Modality"] + ' MRI', nr_channels=1, hold_out_ixs=hold_out_ixs)


def _extract_images(source_path, target_path, merge_labels, subset):
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

    # Create directories
    os.makedirs(os.path.join(target_path))

    # Patient folders s01, s02, ...
    for patient_folder in filter(lambda s: re.match(r"^s[0-9]+.*", s), os.listdir(source_path)):

        # Loading the image
        image_path = os.path.join(source_path, patient_folder,
                                  f"{patient_folder}_{subset['Modality'].lower()}_"
                                  f"{subset['Resolution'].lower()}_defaced_MNI.nii.gz")
        x = sitk.ReadImage(image_path)
        x = sitk.GetArrayFromImage(x)

        # For each MRI, there are 2 segmentation (left and right hippocampus)
        for side in ["L", "R"]:
            # Loading the label
            label_path = os.path.join(source_path, patient_folder,
                                      f"{patient_folder}_hippolabels_"
                                      f"{'hres' if subset['Resolution'] == 'Hires' else 't1w_standard'}"
                                      f"_{side}_MNI.nii.gz")

            y = sitk.ReadImage(label_path)
            y = sitk.GetArrayFromImage(y)

            # We need to recover the study name of the image name to construct the name of the segmentation files
            study_name = f"{patient_folder}_{side}"

            # Average label shape (T1w, standard): (37.0, 36.3, 26.7)
            # Average label shape (T1w, hires): (94.1, 92.1, 68.5)
            # Average label shape (T2w, hires): (94.1, 92.1, 68.5)
            assert x.shape == y.shape

            # Disclaimer: next part is ugly and not many checks are made

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

            if merge_labels:
                y[y > 1] = 1

            x_cropped = x[rmin - dr: rmax + dr,
                        cmin - dc: cmax + dc,
                        zmin - dz: zmax + dz]

            # Save new images so they can be loaded directly
            sitk.WriteImage(sitk.GetImageFromArray(y),
                            join_path([target_path, study_name + "_gt.nii.gz"]))
            sitk.WriteImage(sitk.GetImageFromArray(x_cropped),
                            join_path([target_path, study_name + ".nii.gz"]))

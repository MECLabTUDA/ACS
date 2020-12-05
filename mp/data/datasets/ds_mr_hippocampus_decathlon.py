# ------------------------------------------------------------------------------
# Hippocampus segmentation task from the Medical Segmentation Decathlon
# (http://medicaldecathlon.com/)
# ------------------------------------------------------------------------------

import os

import SimpleITK as sitk

import mp.data.datasets.dataset_utils as du
from mp.data.datasets.dataset_segmentation import SegmentationDataset, SegmentationInstance
from mp.paths import storage_data_path
from mp.utils.load_restore import join_path


class DecathlonHippocampus(SegmentationDataset):
    r"""Class for the hippocampus segmentation decathlon challenge,
    found at http://medicaldecathlon.com/.
    """

    def __init__(self, subset=None, hold_out_ixs=None, merge_labels=True):
        assert subset is None, "No subsets for this dataset."

        if hold_out_ixs is None:
            hold_out_ixs = []

        global_name = 'DecathlonHippocampus'
        dataset_path = os.path.join(storage_data_path, global_name, "Merged Labels" if merge_labels else "Original")
        original_data_path = du.get_original_data_path(global_name)

        # Copy the images if not done already
        if not os.path.isdir(dataset_path):
            _extract_images(original_data_path, dataset_path, merge_labels)

        # Fetch all patient/study names
        study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name
                          in os.listdir(dataset_path))

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
            label_names = ['background', 'hippocampus proper', 'subiculum']
        super().__init__(instances, name=global_name, label_names=label_names,
                         modality='T1w MRI', nr_channels=1, hold_out_ixs=hold_out_ixs)


def _extract_images(source_path, target_path, merge_labels):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """

    images_path = os.path.join(source_path, 'imagesTr')
    labels_path = os.path.join(source_path, 'labelsTr')

    # Filenames have the form 'hippocampus_XX.nii.gz'
    filenames = [x for x in os.listdir(images_path) if x[:5] == 'hippo']

    # Create directories
    os.makedirs(target_path)

    for filename in filenames:

        # Extract only T2-weighted
        x = sitk.ReadImage(os.path.join(images_path, filename))
        x = sitk.GetArrayFromImage(x)
        y = sitk.ReadImage(os.path.join(labels_path, filename))
        y = sitk.GetArrayFromImage(y)

        # Shape expected: (35, 51, 35)
        # Average label shape: (24.5, 37.8, 21.0)
        assert x.shape == y.shape

        # No longer distinguish between hippocampus proper and subiculum
        if merge_labels:
            y[y == 2] = 1

        # Save new images so they can be loaded directly
        study_name = filename.replace('_', '').split('.nii')[0]
        sitk.WriteImage(sitk.GetImageFromArray(x), join_path([target_path, study_name + ".nii.gz"]))
        sitk.WriteImage(sitk.GetImageFromArray(y), join_path([target_path, study_name + "_gt.nii.gz"]))

# ------------------------------------------------------------------------------
# Prostate segmentation task from the Medical Segmentation Decathlon 
# (http://medicaldecathlon.com/)
# ------------------------------------------------------------------------------

import os
import numpy as np
import SimpleITK as sitk
from mp.utils.load_restore import join_path
from mp.data.datasets.dataset_segmentation import SegmentationDataset, SegmentationInstance
from mp.paths import storage_data_path
import mp.data.datasets.dataset_utils as du

class DecathlonProstateT2(SegmentationDataset):
    r"""Class for the prostate segmentation decathlon challenge, only for T2,
    found at http://medicaldecathlon.com/.
    """
    def __init__(self, subset=None, hold_out_ixs=[], merge_labels=True):
        assert subset is None, "No subsets for this dataset."

        global_name = 'DecathlonProstateT2'
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)

        # Separate T2 images, if not already done
        if not os.path.isdir(dataset_path):
            _extract_t2_images(original_data_path, dataset_path, merge_labels)

        # Fetch all patient/study names
        study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
            in os.listdir(dataset_path))

        # Build instances
        instances = []
        for study_name in study_names:
            instances.append(SegmentationInstance(
                x_path=os.path.join(dataset_path, study_name+'.nii.gz'),
                y_path=os.path.join(dataset_path, study_name+'_gt.nii.gz'),
                name=study_name,
                group_id=None
                ))
        if merge_labels:
            label_names = ['background', 'prostate']
        else:
            label_names = ['background', 'peripheral zone', 'central gland']
        super().__init__(instances, name=global_name, label_names=label_names, 
            modality='MR', nr_channels=1, hold_out_ixs=[])
  
def _extract_t2_images(source_path, target_path, merge_labels):
    r"""Extracts T2 images, merges mask labels (if specified) and saves the
    modified images.
    """
    images_path = os.path.join(source_path, 'imagesTr')
    labels_path = os.path.join(source_path, 'labelsTr')

    # Filenames have the form 'prostate_XX.nii.gz'
    filenames = [x for x in os.listdir(images_path) if x[:8] == 'prostate']

    # Create directories
    os.makedirs(target_path)

    for filename in filenames:

        # Extract only T2-weighted
        x = sitk.ReadImage(os.path.join(images_path, filename))
        x = sitk.GetArrayFromImage(x)[0]
        y = sitk.ReadImage(os.path.join(labels_path, filename))
        y = sitk.GetArrayFromImage(y)
        assert x.shape == y.shape

        # No longer distinguish between central and peripheral zones
        if merge_labels:
            y = np.where(y==2, 1, y)

        # Save new images so they can be loaded directly
        study_name = filename.replace('_', '').split('.nii')[0]
        sitk.WriteImage(sitk.GetImageFromArray(x), 
            join_path([target_path, study_name+".nii.gz"]))
        sitk.WriteImage(sitk.GetImageFromArray(y), 
            join_path([target_path, study_name+"_gt.nii.gz"]))
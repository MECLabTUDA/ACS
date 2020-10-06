# ------------------------------------------------------------------------------
# Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation 
# Challenge (M&Ms) dataset.
# ------------------------------------------------------------------------------

import os
import numpy as np
import csv
import SimpleITK as sitk
from mp.data.datasets.dataset_segmentation import SegmentationDataset, SegmentationInstance
from mp.paths import storage_data_path
import mp.data.datasets.dataset_utils as du
from mp.utils.load_restore import join_path

class MM_Challenge(SegmentationDataset):
    r"""Class for importing the Multi-Centre, Multi-Vendor & Multi-Disease
    Cardiac Image Segmentation Challenge (M&Ms), found at www.ub.edu/mnms/."""

    def __init__(self, subset={'Vendor': 'A'}, hold_out_ixs=[]):

        global_name = 'MM_Challenge'
        name = du.get_dataset_name(global_name, subset)
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)

        # Extract ED and ES images, if not already done
        if not os.path.isdir(dataset_path):
            _extract_segmented_slices(original_data_path, dataset_path)
        
        # Fetch metadata
        csv_info = os.path.join(original_data_path, "M&Ms Dataset Information.csv")
        data_info = _get_csv_patient_info(csv_info, id_ix=0)

        # Fetch all patient/study names in the directory (the csv includes 
        # unlabeled data)
        study_names = set(file_name.split('_')[0] for file_name 
            in os.listdir(dataset_path))

        # Fetch image and mask for each study
        instances = []
        for study_name in study_names:
            # If study is part of the defined subset, add ED and ES images
            if subset is None or all(
                [data_info[study_name][key] == value for key, value 
                    in subset.items()]):
                instance_ed = SegmentationInstance(
                    x_path=os.path.join(dataset_path, study_name+'_ED.nii.gz'),
                    y_path=os.path.join(dataset_path, study_name+'_ED_gt.nii.gz'),
                    name=study_name+'_ED',
                    group_id=study_name
                    )
                instance_es = SegmentationInstance(
                    x_path=os.path.join(dataset_path, study_name+'_ES.nii.gz'),
                    y_path=os.path.join(dataset_path, study_name+'_ES_gt.nii.gz'),
                    name=study_name+'_ES',
                    group_id=study_name
                    )
                instances.append(instance_ed)
                instances.append(instance_es)
        label_names = ['background', 'left ventricle', 'myocardium', 'right ventricle']
        super().__init__(instances, name=name, label_names=label_names, 
            modality='MR', nr_channels=1, hold_out_ixs=[])
  

def _get_csv_patient_info(file_full_path, id_ix=0):
    r"""From a .csv file with the description in the top row, turn into a dict 
    where the keys are the dientifier entries and the values are a dictionary 
    with all other entries.
    """
    file_info = dict()
    with open(file_full_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        first_line = None
        for row in reader:
            if first_line is None:
                first_line = row
            else:
                file_info[row[id_ix]] = {key: row[key_ix] for key_ix, key in 
                enumerate(first_line)}
    return file_info

def _extract_segmented_slices(source_path, target_path):
    r"""The original dataset has the following structure:

    MM_Challenge_dataset
    ├── Training-corrected
    │ ├── Labeled
    │ │ ├── <study name>
    │ │ │ ├── <study name>_sa.nii.gz
    │ │ │ └── <study name>_sa_gt.nii.gz
    │ │ └── ...
    └──────── M&Ms Dataset Information.xlsx

    The "M&Ms Dataset Information.xlsx" file should first be converted to csv.
    Each image and mask have the dimension (timesteps, slices, width, height).
    This method extracts only the segmented time steps (ED and ES). The result
    of applying the method is:

    <storage_path>
    ├── data
    │ ├── MM_Challenge
    │ │ ├── <study name>_ED.nii.gz
    │ │ ├── <study name>_ED_gt.nii.gz
    │ │ ├── <study name>_ES.nii.gz
    │ │ ├── <study name>_ES_gt.nii.gz
    │ │ └── ...

    Arguments:
    original_data_path (str): path to MM_Challenge_dataset, where the metadata 
        file has been converted to csv.
    """
    # Fetch metadata
    csv_info = os.path.join(source_path, "M&Ms Dataset Information.csv")
    data_info = _get_csv_patient_info(csv_info, id_ix=0)

    # Create directories
    os.makedirs(target_path)

    # Extract segmented timestamps (ED and ES) and save
    img_path = join_path([source_path, 'Training-corrected', 'Labeled'])
    for study_name in os.listdir(img_path):
        x_path = join_path([img_path, study_name, study_name+"_sa.nii.gz"])
        mask_path = join_path([img_path, study_name, study_name+"_sa_gt.nii.gz"])
        x = sitk.ReadImage(x_path)
        x = sitk.GetArrayFromImage(x)
        mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask)
        assert x.shape == mask.shape
        assert len(x.shape) == 4
        # There are two times for which segmentation is performed, ED and ES.
        # These are specified in the metadata file
        ed_slice = int(data_info[study_name]["ED"])
        es_slice = int(data_info[study_name]["ES"])
        # Store new images
        sitk.WriteImage(sitk.GetImageFromArray(x[ed_slice]), 
            join_path([target_path, study_name+"_ED.nii.gz"]))
        sitk.WriteImage(sitk.GetImageFromArray(mask[ed_slice]), 
            join_path([target_path, study_name+"_ED_gt.nii.gz"]))
        sitk.WriteImage(sitk.GetImageFromArray(x[es_slice]), 
            join_path([target_path, study_name+"_ES.nii.gz"]))
        sitk.WriteImage(sitk.GetImageFromArray(mask[es_slice]), 
            join_path([target_path, study_name+"_ES_gt.nii.gz"]))


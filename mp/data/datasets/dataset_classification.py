# ------------------------------------------------------------------------------
# Classes for creating new classification datasets.
# ------------------------------------------------------------------------------

import os
from mp.data.datasets.dataset import Dataset, Instance
from mp.paths import original_data_paths

class ClassificationPathInstance(Instance):
    """Instance class where x is a path and y is an integer label corr. to
    an index of the dataset 'classes' field.
    """
    def __init__(self, x_path, y, name=None, group_id=None):
        assert isinstance(x_path, str)
        assert isinstance(y, int)
        super().__init__(x=x_path, y=y, class_ix=y, name=name, group_id=group_id)

class SplitClassImageDataset(Dataset):
    """Classification Dataset with the structure root/split/class/filename,
    where 'split' is test for the hold-out test dataset and train for the rest.
    The instances are of the type 'PathInstance'.
    """
    def __init__(self, name, root_path=None, input_shape=(1, 32, 32), x_norm=None):
        if root_path is None:
            root_path = original_data_paths.get(name)
        if root_path is None:
            raise Exception('Data path must be set in paths.py.')
        classes = []
        instances = []
        hold_out_start = None
        for split in ['train', 'test']:
            if split == 'test':
                hold_out_start = len(instances)
            split_path = os.path.join(root_path, split)
            for class_name in os.listdir(split_path):
                if class_name not in classes:
                    classes.append(class_name)
                class_path = os.path.join(split_path, class_name)
                for img_name in os.listdir(class_path):
                    instance = ClassificationPathInstance(name=img_name, x_path=os.path.join(class_path, img_name), y=classes.index(class_name))
                    instances.append(instance)
        super().__init__(name=name, classes=tuple(classes), instances=instances, 
        input_shape=input_shape, output_shape=len(classes), x_norm=x_norm,
        hold_out_ixs=list(range(hold_out_start, len(instances))))

class CIFAR10(SplitClassImageDataset):
    def __init__(self, root_path=None):
        super().__init__(name='cifar10', root_path=root_path, 
        input_shape=(3, 32, 32), 
        x_norm={'mean': (0.4914, 0.4822, 0.4465), 'std': (0.247, 0.243, 0.262)}
        )

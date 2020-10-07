import numpy as np
from mp.data.datasets.dataset_classification import CIFAR10
from mp.data.pytorch.pytorch_class_dataset import ImgClassificationDataset
from mp.utils.pytorch.compute_normalization_values import normalization_values

import pytest
pytest.mark.skip(reason="To test, download the CIFAR10 dataset in .png form.")
def test_normalization_values():
    dataset = CIFAR10()
    pt_dataset = ImgClassificationDataset(dataset)
    norm_values = normalization_values(pt_dataset)
    assert np.allclose(np.array(norm_values['mean']), np.array([0.491, 0.482, 0.446]), atol=0.01)
    assert np.allclose(np.array(norm_values['std']), np.array([0.247, 0.243, 0.262]), atol=0.01)
    normed_dataset = ImgClassificationDataset(dataset, norm=norm_values)
    norm_values = normalization_values(normed_dataset)
    assert np.allclose(np.array(norm_values['mean']), np.array([0., 0., 0.]), atol=1e-05)
    assert np.allclose(np.array(norm_values['std']), np.array([1, 1, 1]))

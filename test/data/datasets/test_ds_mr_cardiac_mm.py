from mp.data.datasets.ds_mr_cardiac_mm import MM_Challenge
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset

def test_ds():
    data = MM_Challenge(subset=None)
    assert data.label_names == ['background', 'left ventricle', 'myocardium', 'right ventricle']
    assert data.nr_labels == 4
    assert data.modality == 'MR'
    assert data.size == 300
    ds = PytorchSeg2DDataset(data, size=(1, 256, 256), aug_key='none', resize=False)
    instance = ds.get_instance(0)
    assert instance.name == 'A0S9V9_ED'
    subject_ix = ds.get_ix_from_name('A0S9V9_ED')
    assert subject_ix == 0

def test_ds_subset():
    data = MM_Challenge(subset={'Vendor': 'B'})
    print(data.size)
    assert data.size == 150
    ds = PytorchSeg2DDataset(data, size=(1, 256, 256), aug_key='none', resize=False)
    instance = ds.get_instance(0)
    assert instance.name == 'A1D0Q7_ED'
    subject_ix = ds.get_ix_from_name('A1D0Q7_ED')
    assert subject_ix == 0

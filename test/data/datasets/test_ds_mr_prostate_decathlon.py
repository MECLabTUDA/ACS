from mp.data.datasets.ds_mr_prostate_decathlon import DecathlonProstateT2

def test_ds_label_merging():
    data = DecathlonProstateT2(merge_labels=True)
    assert data.label_names == ['background', 'prostate']
    assert data.nr_labels == 2
    assert data.modality == 'MR'
    assert data.size == 32
    assert data.name == 'DecathlonProstateT2'
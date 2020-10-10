from mp.data.datasets.dataset import Dataset
from mp.data.datasets.dataset_classification import ClassificationPathInstance
from mp.experiments.data_splitting import split_instances, create_instance_folds, split_dataset

def test_split_instances():
    instances = []
    instances.append(ClassificationPathInstance(name='0A', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='1A', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1B', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='0B', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='0C', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='1C', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='0D', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='0E', y=0, x_path=''))
    ds = Dataset(name=None, classes=('0', '1'), instances=instances)
    # Split at 70%
    ixs_1, ixs_2 = split_instances(ds, 0.7, stratisfied=True, exclude_ixs=[3])
    class_dictribution = ds.get_class_dist(ixs_1)
    assert class_dictribution['0'] == 2
    assert class_dictribution['1'] == 2
    class_dictribution = ds.get_class_dist(ixs_2)
    assert class_dictribution['0'] == 2
    assert class_dictribution['1'] == 1
    # Split at 80%
    ixs_1, ixs_2 = split_instances(ds, 0.8, stratisfied=True, exclude_ixs=[3])
    class_dictribution = ds.get_class_dist(ixs_1)
    assert class_dictribution['0'] == 3
    assert class_dictribution['1'] == 2
    class_dictribution = ds.get_class_dist(ixs_2)
    assert class_dictribution['0'] == 1
    assert class_dictribution['1'] == 1
    # Split at 90%
    # not possible because of too few examples of class 0
    try:
        ixs_1, ixs_2 = split_instances(ds, 0.9, stratisfied=True)
        assert False
    except RuntimeError:
        pass

def test_cross_validation():
    instances = []
    instances.append(ClassificationPathInstance(name='0A', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='1A', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1B', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='0B', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='0C', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='1C', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='0D', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='0E', y=0, x_path=''))
    ds = Dataset(name=None, classes=('0', '1'), instances=instances)
    # Split into 2 folds
    folds = create_instance_folds(ds, k=2, exclude_ixs=[3], stratisfied=True)
    for fold in folds:
        assert len(fold) < 5
        class_dictribution = ds.get_class_dist(fold)
        assert class_dictribution['1']==1 or class_dictribution['1']==2
        assert class_dictribution['0']==2
    # Split into 3 folds
    folds = create_instance_folds(ds, k=3, exclude_ixs=[3], stratisfied=True)
    for fold in folds:
        assert len(fold) < 4
        class_dictribution = ds.get_class_dist(fold)
        assert class_dictribution['1']==1 
        assert class_dictribution['0']==1 or class_dictribution['0']==2
    # Split into 4 folds
    try:
        folds = create_instance_folds(ds, k=4, exclude_ixs=[3], stratisfied=True)
        assert False
    except RuntimeError:
        pass

def test_split_dataset():
    instances = []
    instances.append(ClassificationPathInstance(name='0A', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='0B', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='0C', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='0D', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='0E', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='1A', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1B', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1C', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1D', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1E', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1F', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1G', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1H', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='0A2', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='0B2', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='0C2', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='0D2', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='0E2', y=0, x_path=''))
    instances.append(ClassificationPathInstance(name='1A2', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1B2', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1C2', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1D2', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1E2', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1F2', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1G2', y=1, x_path=''))
    instances.append(ClassificationPathInstance(name='1H2', y=1, x_path=''))
    ds = Dataset(name=None, classes=('0', '1'), instances=instances)
    # Split into 2 folds. Then split into train, test and validation sets
    folds = split_dataset(ds, test_ratio=0.2, val_ratio=0.2, nr_repetitions=2, cross_validation=True)
    for fold in folds:
        assert len(set(fold['train'] + fold['val'] + fold['test'])) == len(fold['train'] + fold['val'] + fold['test'])
        class_dictribution = ds.get_class_dist(fold['train'])
        assert class_dictribution['0']==4
        assert class_dictribution['1']==6
        class_dictribution = ds.get_class_dist(fold['val'])
        assert class_dictribution['0']==1
        assert class_dictribution['1']==2
        class_dictribution = ds.get_class_dist(fold['test'])
        assert class_dictribution['0']==5
        assert class_dictribution['1']==8
    # Repetitions
    splits = split_dataset(ds, test_ratio=0.2, val_ratio=0.2, nr_repetitions=3, cross_validation=False)
    for split in splits:
        class_dictribution = ds.get_class_dist(split['train'])
        assert class_dictribution['0']==6
        assert class_dictribution['1']==10
        class_dictribution = ds.get_class_dist(split['val'])
        assert class_dictribution['0']==2
        assert class_dictribution['1']==2
        class_dictribution = ds.get_class_dist(split['test'])
        assert class_dictribution['0']==2
        assert class_dictribution['1']==4

def test_group_id_division():
    instances = []
    instances.append(ClassificationPathInstance(name='0Aa', y=0, group_id='a', x_path=''))
    instances.append(ClassificationPathInstance(name='0Bc', y=0, group_id='c', x_path=''))
    instances.append(ClassificationPathInstance(name='0Cd', y=0, group_id='d', x_path=''))
    instances.append(ClassificationPathInstance(name='0Dd', y=0, group_id='d', x_path=''))
    instances.append(ClassificationPathInstance(name='0Ee', y=0, group_id='e', x_path=''))
    instances.append(ClassificationPathInstance(name='1Aa', y=1, group_id='a', x_path=''))
    instances.append(ClassificationPathInstance(name='1Bg', y=1, group_id='g', x_path=''))
    instances.append(ClassificationPathInstance(name='1Cf', y=1, group_id='f', x_path=''))
    instances.append(ClassificationPathInstance(name='1Df', y=1, group_id='f', x_path=''))
    instances.append(ClassificationPathInstance(name='1Eb', y=1, group_id='b', x_path=''))
    instances.append(ClassificationPathInstance(name='1Fb', y=1, group_id='b', x_path=''))
    instances.append(ClassificationPathInstance(name='1Gc', y=1, group_id='c', x_path=''))
    instances.append(ClassificationPathInstance(name='1Ha', y=1, group_id='a', x_path=''))
    instances.append(ClassificationPathInstance(name='0A2a', y=0, group_id='a', x_path=''))
    instances.append(ClassificationPathInstance(name='0B2b', y=0, group_id='b', x_path=''))
    instances.append(ClassificationPathInstance(name='0C2b', y=0, group_id='b', x_path=''))
    instances.append(ClassificationPathInstance(name='0D2c', y=0, group_id='c', x_path=''))
    instances.append(ClassificationPathInstance(name='0E2e', y=0, group_id='e', x_path=''))
    instances.append(ClassificationPathInstance(name='1A2g', y=1, group_id='g', x_path=''))
    instances.append(ClassificationPathInstance(name='1B2h', y=1, group_id='h', x_path=''))
    instances.append(ClassificationPathInstance(name='1C2h', y=1, group_id='h', x_path=''))
    instances.append(ClassificationPathInstance(name='1D2e', y=1, group_id='e', x_path=''))
    instances.append(ClassificationPathInstance(name='1E2e', y=1, group_id='e', x_path=''))
    instances.append(ClassificationPathInstance(name='1F2d', y=1, group_id='d', x_path=''))
    instances.append(ClassificationPathInstance(name='1G2c', y=1, group_id='c', x_path=''))
    instances.append(ClassificationPathInstance(name='1H2d', y=1, group_id='d', x_path=''))
    ds = Dataset(name=None, classes=('0', '1'), instances=instances)
    # Split into 2 folds. Then split into train, test and validation sets
    folds = split_dataset(ds, test_ratio=0.2, val_ratio=0.4, nr_repetitions=2, 
        cross_validation=True, respecting_groups=True)
    for fold in folds:
        assert len(set(fold['train'] + fold['val'] + fold['test'])) == len(fold['train'] + fold['val'] + fold['test'])
        class_dictribution = ds.get_class_dist(fold['train'])
        assert class_dictribution['0']>=2
        assert class_dictribution['1']>=4
        assert len([instances[ix].group_id for ix in fold['train']]) % 2 == 0
        class_dictribution = ds.get_class_dist(fold['val'])
        assert class_dictribution['0']>=2
        assert class_dictribution['1']>=2
        assert len([instances[ix].group_id for ix in fold['val']]) % 2 == 0
        class_dictribution = ds.get_class_dist(fold['test'])
        assert class_dictribution['0']>=4
        assert class_dictribution['1']>=6
        assert len([instances[ix].group_id for ix in fold['test']]) % 2 == 0
    # Repetitions
    splits = split_dataset(ds, test_ratio=0.2, val_ratio=0.2, nr_repetitions=3, 
        cross_validation=False, respecting_groups=True)
    for split in splits:
        class_dictribution = ds.get_class_dist(split['train'])
        assert class_dictribution['0']==6
        assert class_dictribution['1']==10
        assert len([instances[ix].group_id for ix in fold['train']]) % 2 == 0
        class_dictribution = ds.get_class_dist(split['val'])
        assert class_dictribution['0']==2
        assert class_dictribution['1']==2
        assert len([instances[ix].group_id for ix in fold['val']]) % 2 == 0
        class_dictribution = ds.get_class_dist(split['test'])
        assert class_dictribution['0']==2
        assert class_dictribution['1']==4
        assert len([instances[ix].group_id for ix in fold['test']]) % 2 == 0

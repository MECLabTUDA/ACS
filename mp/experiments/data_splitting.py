# ------------------------------------------------------------------------------
# A dataset is split into train, validation and test sets. This splitting can be
# performed at random, for instance to perform n repetitions, or through
# cross-validation. Each repetition or fold is used for a different
# experiment run. The indexes for each split are stored in a 'splits.json' file.
# A hold-out test dataset may be kept which is always the same and initialized
# together with a Dataset instance.
# ------------------------------------------------------------------------------

import random
import math
  
def split_dataset(
    dataset, test_ratio=0.2, val_ratio=0.2, 
    nr_repetitions=5, cross_validation=True, 
    respecting_groups=True):
    r"""Splits a dataset into different index folds.

    Args:
        dataset (Dataset): a Dataset object
        test_ratio (float): ratio of instances for testing
        val_ratio (float): ratio of non-test instances for validation
        nr_repetitions (int): number of times the experiment should be repeated,
            i.e. number of index splits which are created.
        cross_validation (bool): are the repetitions cross-val folds?
        respecting_groups (bool): do not place examples with the same group in
            the same fold
    
    Returns (list[dict[str -> list[int]]]): A list of length 'nr_repetitions' 
        where each item is a dictionary with keys 'train', 'val' and 'test'.
    """
    splits = []
    if cross_validation:
        folds = create_instance_folds(dataset=dataset, k=nr_repetitions, 
            exclude_ixs=dataset.hold_out_ixs, stratisfied=True,
            respecting_groups=respecting_groups)
        for k in range(nr_repetitions):
            print('Cross-validation fold k {} of {}'.format(k+1, nr_repetitions))
            train, val = [], []
            for j in range(nr_repetitions):
                if j != k:
                    train += folds[j]
            if val_ratio > 0:
                train, val = split_instances(dataset=dataset, ratio=1-val_ratio, 
                    exclude_ixs=dataset.hold_out_ixs+folds[k],     
                    stratisfied=True, respecting_groups=respecting_groups)
            splits.append({'train': train, 'val': val, 'test': folds[k]})
    else:
        for k in range(nr_repetitions):
            print('Repetition k {} of {}'.format(k+1, nr_repetitions))
            train_val, test = split_instances(dataset=dataset, 
                ratio=1-test_ratio, exclude_ixs=dataset.hold_out_ixs, 
                stratisfied=True, respecting_groups=respecting_groups)
            if val_ratio> 0:
                train, val = split_instances(dataset=dataset, ratio=1-val_ratio, 
                    exclude_ixs=dataset.hold_out_ixs+test, stratisfied=True, 
                    respecting_groups=respecting_groups)
            else:
                train, val = train_val, []
            splits.append({'train': train, 'val': val, 'test': test})
    return splits


def split_instances(dataset, ratio=0.7, exclude_ixs=[], stratisfied=True, 
    respecting_groups=True):
    r"""Divides instances into two stratisfied sets. The stratification 
    operations prefers to give more examples of underrepresented classes
    to smaller sets (when the examples in a class cannot be split without
    a remainder).

    Args:
        ratio (float): ratio of instances which remain in the first set.
        exclude_ixs (list[int]): exclude these indexes from the splitting
        stratisfied (bool): should there be ca. as any examples for each class?
        respecting_groups (bool): do not place examples with the same group in
            the same fold
    
    Returns (tuple[list[int]]): 2 index lists with the indexes
    """
    ixs = range(dataset.size)
    ixs = [ix for ix in ixs if ix not in exclude_ixs]
    first_ds_len = math.floor(len(ixs)*ratio)
    if not stratisfied:
        random.shuffle(ixs)
        return _split_ixs(ixs, first_ds_len=first_ds_len, 
            instances=dataset.instances, respecting_groups=respecting_groups)
    else:
        ixs_1, ixs_2 = [], []
        class_instances = {class_name: dataset.get_class_instance_ixs(
                class_name=class_name, exclude_ixs=exclude_ixs) for class_name 
                in dataset.classes}
        classes = list(dataset.classes)
        classes.sort(key=lambda x: len(class_instances[x]))
        for class_name in classes:
            #print('Class: {}'.format(class_name))
            ixs = class_instances[class_name]
            random.shuffle(ixs)
            # The mayority class is used to fill to the desired number of 
            # examples for each split
            if class_name == classes[-1]:
                remaining_exs_nr = first_ds_len - len(ixs_1)
                if remaining_exs_nr == len(ixs):
                    raise RuntimeError(
                        'Not enough examples of class {}'.format(class_name))
                class_ixs_1, class_ixs_2 = _split_ixs(ixs, 
                    first_ds_len=remaining_exs_nr, instances=dataset.instances, 
                    respecting_groups=respecting_groups)
                ixs_1 += class_ixs_1
                ixs_2 += class_ixs_2
            # Otherwise, the operation makes sure less-represented classes
            # are as represented as possible in small sets
            else:
                nr_class_first_ds = math.floor(len(ixs)*ratio)
                if nr_class_first_ds == len(ixs):
                    raise RuntimeError(
                        'Not enough examples of class {}'.format(class_name))
                class_ixs_1, class_ixs_2 = _split_ixs(ixs, 
                    first_ds_len=nr_class_first_ds, instances=dataset.instances, 
                    respecting_groups=respecting_groups)
                ixs_1 += class_ixs_1
                ixs_2 += class_ixs_2
    assert len(set(ixs_1+ixs_2+exclude_ixs)) == len(dataset.instances)
    return ixs_1, ixs_2

def create_instance_folds(dataset, k=5, exclude_ixs=[], 
    stratisfied=True, respecting_groups=True):
    r"""Divides instances into k stratisfied sets. Always, the most examples of 
    a class (when not divisible) are added to the fold that currently has
    the least examples.

    Args:
        k (int): number of sets.
        exclude_ixs (list[int]): exclude these indexes from the splitting
        stratisfied (bool): should there be ca. as any examples for each class?
        respecting_groups (bool): do not place examples with the same group in
            the same fold
    
    Returns (tuple[list[int]]): k index lists with the indexes
    """
    ixs = range(dataset.size)
    ixs = [ix for ix in ixs if ix not in exclude_ixs]
    if not stratisfied:
        return _divide_sets_similar_length(dataset.instances, ixs, k, respecting_groups)
    else:
        folds = [[] for k_ix in range(k)]
        class_instances = {class_name: dataset.get_class_instance_ixs(
                class_name=class_name, exclude_ixs=exclude_ixs) for 
                class_name in dataset.classes}
        classes = list(dataset.classes)
        classes.sort(key=lambda x: len(class_instances[x]))
        for class_name in classes:
            #print('Class: {}'.format(class_name))
            exs = class_instances[class_name]
            # Sort so folds with least examples come first
            folds.sort(key=lambda x: len(x))
            divided_exs = _divide_sets_similar_length(dataset.instances, exs, k, respecting_groups)
            for i in range(len(divided_exs)):
                folds[i] += divided_exs[i]
    assert sum([len(fold) for fold in folds])+len(exclude_ixs) == len(dataset.instances)
    return folds

def _divide_sets_similar_length(instances, exs, k, respecting_groups=True):
    r"""Divides a list exs into k sets of similar length, with the initial 
    ones being longer.
    """
    random.shuffle(exs)
    folds = [[] for i in range(k)]
    # Add example indexes to folds
    if instances[0].group_id == None or not respecting_groups:
        # Calculate number of examples per fold
        nr_per_fold, remaining = divmod(len(exs), k)
        if nr_per_fold < 1:
            raise RuntimeError('Not enough examples.')
        nr_per_fold_final = []
        for _ in range(k):
            nr_exs = nr_per_fold
            if remaining > 0:
                nr_exs += 1
            nr_per_fold_final.append(nr_exs)
            remaining -= 1
        current_fold_ix = 0
        for ix in exs:
            folds[current_fold_ix].append(ix)
            if len(folds[current_fold_ix]) == nr_per_fold_final[current_fold_ix]:
                current_fold_ix += 1
    else:
        # Form groups
        ixs_groups = dict()
        for ix in exs:
            group_id = instances[ix].group_id
            if group_id not in ixs_groups:
                ixs_groups[group_id] = []
            ixs_groups[group_id].append(ix)
        # Calculate number of groups per fold
        nr_per_fold, remaining = divmod(len(ixs_groups.keys()), k)
        if nr_per_fold < 1:
            raise RuntimeError('Not enough groups.')
        nr_per_fold_final = []
        for _ in range(k):
            nr_exs = nr_per_fold
            if remaining > 0:
                nr_exs += 1
            nr_per_fold_final.append(nr_exs)
            remaining -= 1
        # Divide groups
        current_fold_ix = 0
        nr_fold_groups = 0
        for ix_lst in ixs_groups.values():
            folds[current_fold_ix] += ix_lst
            nr_fold_groups += 1
            if nr_fold_groups >= nr_per_fold_final[current_fold_ix]:
                current_fold_ix += 1
                nr_fold_groups = 0
        # Return divided indexes
        for fold in folds:
            assert len(fold)>0, 'Not enough examples'
    return folds

def _split_ixs(ixs, first_ds_len, instances, respecting_groups=True):
    r"""Returns two lists of indexes, which are subsets of ixs.
    """
    if not respecting_groups or instances[0].group_id == None:
        return ixs[:first_ds_len], ixs[first_ds_len:]
    else:
        ixs_1, ixs_2 = [], []
        # Form groups
        ixs_groups = dict()
        for ix in ixs:
            group_id = instances[ix].group_id
            if group_id not in ixs_groups:
                ixs_groups[group_id] = []
            ixs_groups[group_id].append(ix)
        # Divide groups
        for ix_lst in ixs_groups.values():
            if len(ixs_1) >= first_ds_len:
                ixs_2 += ix_lst
            else:
                ixs_1 += ix_lst
        # Return divided indexes
        assert len(ixs_1)>0 and len(ixs_2)>0, 'Not enough examples'
        return ixs_1, ixs_2
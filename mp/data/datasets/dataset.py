# ------------------------------------------------------------------------------
# Dataset class meant to store general information about dataset and to divide 
# instances in folds, before converting to a torch.utils.data.Dataset. 
# All datasets descend from this Dataset class.
# ------------------------------------------------------------------------------

class Dataset:
    r"""A dataset stores instances.

    Args:
        name (str): name of a dataset
        instances (list[mp.data.datasets.dataset.Instance]): list of Instances
        classes (tuple[str]): tuple with label names
        hold_out_ixs (list[int]): instances which are not evaluated until the end 
        mean_shape (list[int]): mean input shape
        output_shape (list[int]): output shape
        x_norm (tuple[float]): normalization values for the input
    """
    def __init__(self, name, instances, classes=('0'), hold_out_ixs=[], 
        mean_shape=(1, 32, 32), output_shape=(1, 32, 32), x_norm=None):
        self.name = name
        # Sort instances in terms of name
        self.instances = sorted(instances, key=lambda ex: ex.name)
        self.size = len(instances)
        self.classes = classes
        self.hold_out_ixs = hold_out_ixs
        self.mean_shape = mean_shape
        self.output_shape = output_shape
        self.x_norm = x_norm

    def get_class_dist(self, ixs=None):
        r"""Get class (category) distribution

        Args:
            ixs (list[int]): if not None, distribution for only these indexes. 
            Otherwise distribution for all indexes not part of the hold-out.
        """ 
        if ixs is None:
            ixs = [ix for ix in range(self.size) if ix not in self.hold_out_ixs]
        class_dist = {class_ix: 0 for class_ix in self.classes}
        for ex_ix, ex in enumerate(self.instances):
            if ex_ix in ixs:
                class_dist[self.classes[ex.class_ix]] += 1
        return class_dist

    def get_class_instance_ixs(self, class_name, exclude_ixs):
        r"""Get instances for a class, excluding those in exclude_ixs.""" 
        return [ix for ix, ex in enumerate(self.instances) if 
            ex.class_ix==self.classes.index(class_name) and ix not in exclude_ixs]

    def get_instance(self, name):
        r"""Get an instance from a name.""" 
        instances = [instance for instance in self.instances if instance.name == name]
        if len(instances) == 0:
            return None
        else:
            assert len(instances) == 1, "There are more than one instance with that name"
            return instances[0]

    def get_instance_ixs_from_names(self, name_lst):
        r"""Get instance ixs from a list of names.""" 
        ixs = [ix for ix, instance in enumerate(self.instances) if instance.name in name_lst]
        return ixs

class Instance:
    r"""A dataset instance.

    Args:
        x (Obj): input, can take different forms depending on the subclass
        y (Obj): ground truth
        name (str): instance name (e.g. file name) for case-wise evaluation
        class_ix (int): during splitting of the dataset, the resulting subsets 
            are stratesfied according to this value (i.e. there are about as 
            many examples of each class on each fold). For classification, 
            class_ix==y, but can also be used solely for splitting.
        group_id (int): instances with same 'group id' should always 
            remain on the same dataset split. A group id could be, for instance, 
            a patient identifier (the same patient should typically not be in 
            several different splits).
    """
    def __init__(self, x, y, name=None, class_ix=0, group_id=None):
        self.x = x
        self.y = y
        self.name = name
        self.class_ix = class_ix
        self.group_id = group_id
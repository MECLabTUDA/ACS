# ------------------------------------------------------------------------------
# An instance of data includes a dictionary of datasets.
# TODO: in the standard Dataset class, label_names are classes
# ------------------------------------------------------------------------------

class Data:
    def __init__(self):
        self.datasets = dict()
        self.label_names = None
        self.nr_labels = None

    def add_dataset(self, dataset):
        assert dataset.name not in self.datasets
        if len(self.datasets) > 0:
            for other_dataset in self.datasets.values():
                assert dataset.label_names == other_dataset.label_names, 'Datasets must have the same label names'
        else:
            self.label_names = dataset.label_names
            self.nr_labels = dataset.nr_labels
        self.datasets[dataset.name] = dataset
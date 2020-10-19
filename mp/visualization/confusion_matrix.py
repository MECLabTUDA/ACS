# ------------------------------------------------------------------------------
# A confusion matrix for classification tasks.
# ------------------------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ConfusionMatrix:
    r"""An internal representtaion for a confusion matrix.

    Args:
        nr_classes (int): number of classes/labels
    """
    def __init__(self, nr_classes, labels=None):
        self.cm = [[0 for i in range(nr_classes)] for i in range(nr_classes)]
        if labels is None:
            self.labels = list(range(nr_classes))
        else:
            self.labels = labels

    def add(self, predicted, actual, count=1):
        r"""Set an entry for the confusion matrix."""
        self.cm[self.labels.index(actual)][self.labels.index(predicted)] += count

    def plot(self, path, name='confusion_matrix', label_predicted='Predicted', 
        label_actual='Actual', figure_size=(7,5), annot=True):
        r"""Plot using seaborn."""
        cm = self.cm.copy()
        nr_rows = len(cm)
        cm.insert(0, [0]*nr_rows)
        df = pd.DataFrame(cm, columns=[c+1 for c in range(nr_rows)])
        df = df.drop([0])
        plt.figure()
        sns.set(rc={'figure.figsize':figure_size})
        ax = sns.heatmap(df, annot=annot)
        ax.set(xlabel=label_predicted, ylabel=label_actual)
        plt.savefig(os.path.join(path, name+'.png'), facecolor='w', 
            bbox_inches="tight", dpi = 300)
        
    def get_accuracy(self):
        r"""Get the accuracy."""
        correct = sum([self.cm[i][i] for i in range(len(self.cm))])
        all_instances = sum([sum(x) for x in self.cm])
        return correct/all_instances
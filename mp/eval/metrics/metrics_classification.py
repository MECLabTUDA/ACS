# ------------------------------------------------------------------------------
# Collection of metrics.
# ------------------------------------------------------------------------------

import torch
import numpy as np

def accuracy(outputs, targets):
    _, pred = torch.max(outputs.data, 1)
    total = outputs.size(0)
    correct = (pred == targets).sum().item()
    return correct/total

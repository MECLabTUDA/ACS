# ------------------------------------------------------------------------------
# Accumulates results from a minibatch.
# ------------------------------------------------------------------------------

import numpy as np
import torch

class Accumulator:
    def __init__(self, keys=None):
        self.values = dict()
        if keys is not None:
            self.init(keys)

    def update(self, acc):
        for key, value in acc.values.items():
            if key not in self.values:
                self.values[key] = value

    def init(self, keys):
        for key in keys:
            self.values[key] = []

    def ensure_key(self, key):
        if key not in self.values:
            self.values[key] = []
            
    def add(self, key, value, count=1):
        self.ensure_key(key)
        if isinstance(value, torch.Tensor):
            np_value = float(value.detach().cpu())
        else:
            np_value = value
        for _ in range(count):
            self.values[key].append(np_value)

    def mean(self, key):
        return np.mean(self.values[key])

    def std(self, key):
        return np.std(self.values[key])

    def sum(self, key):
        return sum(self.values[key])

    def get_keys(self):
        return sorted(list(self.values.keys()))



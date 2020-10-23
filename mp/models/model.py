# ------------------------------------------------------------------------------
# Class all model definitions should descend from.
# ------------------------------------------------------------------------------

import os
import torch.nn as nn
import numpy as np
from torchsummary import summary

from mp.utils.pytorch.pytorch_load_restore import load_model_state, save_model_state

class Model(nn.Module):
    r"""A model that descends from torch.nn.Model and includes methods to output
    a model summary, as well as the input_shape and output_shape fields used in
    other models and the logic to restore previous model states from a path.

    Args:
        input_shape tuple (int): Input shape with the form
            (channels, width, height, Opt(depth))
        output_shape (Obj): output shape, which takes different forms depending 
            on the problem
    """
    def __init__(self, input_shape=(1, 32, 32), output_shape=2):
        super(Model, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def preprocess_input(self, x):
        r"""E.g. pretrained features. Override if needed. """
        return x

    def initialize(self, weights_init_path, device):
        r"""Tries to restore a previous model. If no model is found, the initial 
        weights are saved.
        """
        path, name = os.path.split(weights_init_path) 
        restored = load_model_state(self, path=path, name=name, device=device)
        if restored:
            print('Initial parameters {} were restored'.format(weights_init_path))
        else:
            save_model_state(self, path=path, name=name)
            print('Initial parameters {} were saved'.format(weights_init_path))

    def get_param_list_static(self):
        r"""Returns a 1D array of parameter values
        """
        model_params_array = []
        for _, param in self.state_dict().items():
            model_params_array.append(param.reshape(-1).cpu().numpy())
        return np.concatenate(model_params_array)


    # Method to output model information

    def model_summary(self, verbose=False):
        r"""Return a Keras-style summary."""
        summary_str = str(summary(self, input_data=self.input_shape, verbose=0))
        if verbose:
            print(summary_str)
        return summary_str


    # Methods to calculate the feature size

    def num_flat_features(self, x):
        r"""Flattened view of all dimensions except the batch size.
        """
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def flatten(self, x):
        r"""Flatten x into 1 dimension."""
        return x.view(-1, self.num_flat_features(x))

    def size_before_lin(self, shape_input):
        r"""Size after linearization.

        Returns (int): integer of dense size
        """
        return shape_input[0]*shape_input[1]*shape_input[2]

    def size_after_conv(self, shape_input, output_channels, kernel):
        r"""Gives the number of output neurons after the conv operation.
        The first dimension is the channel depth and the other 2 are given by
        input volume (size - kernel size + 2*padding)/stride + 1
        """
        return (output_channels, shape_input[1]-kernel+1, shape_input[2]-kernel+1)

    def size_after_pooling(self, shape_input, shape_pooling):
        r"""Maintains the first input dimension, which is the output channels in 
        the previous conv layer. The others are divided by the shape of the 
        pooling.
        """
        return (shape_input[0], shape_input[1]//shape_pooling[0], shape_input[2]//shape_pooling[1])

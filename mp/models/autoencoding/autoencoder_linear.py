# ------------------------------------------------------------------------------
# A linear autoencoder. Note that the input must be normalized between
# 0 and 1.
# ------------------------------------------------------------------------------

from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from mp.models.autoencoding.autoencoder import Autoencoder

class AutoencoderLinear(Autoencoder):
    r"""An autoencoder with only linear layers."""
    def __init__(self, input_shape, hidden_dim = [128, 64]):
        super().__init__(input_shape=input_shape)
        in_dim = self.input_shape[0] if len(self.input_shape)<2 else reduce(lambda x, y: x*y, self.input_shape)
        dims = [in_dim] + hidden_dim

        # Encoder layers
        self.enc_layers = nn.ModuleList([nn.Linear(in_features=dims[i], out_features=dims[i+1]) 
            for i in range(len(dims)-1)])

        # Decoder layers
        self.dec_layers = nn.ModuleList([nn.Linear(in_features=dims[i+1], out_features=dims[i]) 
            for i in reversed(range(len(dims)-1))])

    def preprocess_input(self, x):
        r"""Flatten x into one dimension."""
        return torch.flatten(x, start_dim=1)

    def encode(self, x):
        for layer in self.enc_layers:
            x = F.relu(layer(x))
        return x

    def decode(self, x):
        for layer in self.dec_layers:
            x = F.relu(layer(x))
        return x
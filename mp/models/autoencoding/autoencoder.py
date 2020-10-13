# ------------------------------------------------------------------------------
# Basic class for autoencoder models that reconstruct the input.
# ------------------------------------------------------------------------------

from mp.models.model import Model

class Autoencoder(Model):
    r"""A superclass for autoencoder models.

    Args:
        input_shape tuple (int): (channels, width, height, Opt(depth))
    """
    def __init__(self, input_shape):
        # An autoencoder has the same input and output shapes
        super().__init__(input_shape, output_shape=input_shape)

    def encode(self, x):
        r"""Encode the input."""
        raise NotImplementedError

    def decode(self, x):
        r"""Decode the features into an output."""
        raise NotImplementedError

    def forward(self, x):
        initial_shape = x.shape
        x = self.encode(x)
        x = self.decode(x)
        assert x.shape == initial_shape
        return x

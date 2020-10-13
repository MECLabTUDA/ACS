# ------------------------------------------------------------------------------
# Basic class for segmentation models.
# ------------------------------------------------------------------------------

from mp.models.model import Model

class SegmentationModel(Model):
    r"""An abstract class for segmentation models that caluclates the output 
    shape from the input shape and the number of labels."""
    def __init__(self, input_shape, nr_labels):
        assert 2 < len(input_shape) < 5
        # The output shae is the same as the input shape, but instead of the 
        # input channels it has the number of labels as channels
        output_shape = tuple([nr_labels] + list(input_shape[1:]))
        super().__init__(input_shape, output_shape=output_shape)
        self.nr_labels = nr_labels


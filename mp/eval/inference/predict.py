# ------------------------------------------------------------------------------
# Transform a multi-channeled network output into a prediction, and similar 
# helper functions.
# ------------------------------------------------------------------------------

import torch

def arg_max(output, channel_dim=1):
    return torch.argmax(output, dim=channel_dim)

def softmax(output, channel_dim=1):
    f = torch.nn.Softmax(dim=channel_dim)
    return f(output)

# TODOs
def confidence(softmaxed_output, channel_dim=1):
    """Returns the confidence for each voxel (the highest value along the 
    channel dimension)."""
    pass

def ensable_prediction():
    pass

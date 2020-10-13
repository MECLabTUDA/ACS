# ------------------------------------------------------------------------------
# An abstract loss function to use during training. These are defined in the 
# project to output respective evaluation dictionaries that report all 
# components of the loss separatedly.
# ------------------------------------------------------------------------------

import torch.nn as nn

class LossAbstract(nn.Module):
    r"""A named loss function, that loss functions should inherit from.
        Args:
            device (str): device key
    """
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
        self.name = self.__class__.__name__

    def get_evaluation_dict(self, output, target):
        r"""Return keys and values of all components making up this loss.
        Args:
            output (torch.tensor): a torch tensor for a multi-channeled model 
                output
            target (torch.tensor): a torch tensor for a multi-channeled target
        """
        return {self.name: float(self.forward(output, target).cpu())}
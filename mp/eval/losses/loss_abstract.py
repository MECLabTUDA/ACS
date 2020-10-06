# ------------------------------------------------------------------------------
# An abstract loss.
# ------------------------------------------------------------------------------

import torch.nn as nn

class LossAbstract(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
        self.name = self.__class__.__name__

    def get_evaluation_dict(self, output, target):
        '''
        Return keys and values of all components making up this loss.
        '''
        return {self.name: float(self.forward(output, target).cpu())}
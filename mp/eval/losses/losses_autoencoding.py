# ------------------------------------------------------------------------------
# Similitude metrics between output and target.
# ------------------------------------------------------------------------------

import torch.nn as nn
from mp.eval.losses.loss_abstract import LossAbstract

class LossMSE(LossAbstract):
    r"""Mean Squared Error."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, output, target):
        return self.mse(output, target)

class LossL1(LossAbstract):
    r"""L1 distance loss."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, output, target):
        return self.l1(output, target)
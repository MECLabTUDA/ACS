# ------------------------------------------------------------------------------
# Collection of loss metrics that can be used during training, including binary
# cross-entropy and dice. Class-wise weights can be specified.
# Losses receive a 'target' array with shape (batch_size, channel_dim, etc.)
# and channel dimension equal to nr. of classes that has been previously 
# transformed (through e.g. softmax) so that values lie between 0 and 1, and an 
# 'output' array with the same dimension and values that are either 0 or 1.
# The results of the loss is always averaged over batch items (the first dim).
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from mp.eval.losses.loss_abstract import LossAbstract

class LossDice(LossAbstract):
    r"""Dice loss with a smoothing factor."""
    def __init__(self, smooth=1., device='cuda:0'):
        super().__init__(device=device)
        self.smooth = smooth
        self.name = 'LossDice[smooth='+str(self.smooth)+']'

    def forward(self, output, target):
        output_flat = output.view(-1)
        target_flat = target.view(-1)
        intersection = (output_flat * target_flat).sum()
        return 1 - ((2. * intersection + self.smooth) /
                (output_flat.sum() + target_flat.sum() + self.smooth))

class LossBCE(LossAbstract):
    r"""Binary cross entropy loss."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.bce = nn.BCELoss(reduction='mean')

    def forward(self, output, target):
        return self.bce(output, target)

class LossBCEWithLogits(LossAbstract):
    r"""More stable than following applying a sigmoid function to the output 
    before applying the loss (see 
    https://pytorch.org/docs/stable/generated/torch.nn.LossBCEWithLogits.html), 
    but only use if applicable."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.bce = nn.BCELossWithLogits(reduction='mean')

    def forward(self, output, target):
        return self.bce(output, target)

class LossCombined(LossAbstract):
    r"""A combination of several different losses."""
    def __init__(self, losses, weights, device='cuda:0'):
        super().__init__(device=device)
        self.losses = losses
        self.weights = weights
        # Set name
        self.name = 'LossCombined['
        for loss, weight in zip(self.losses, self.weights):
            self.name += str(weight)+'x'+loss.name + '+'
        self.name = self.name[:-1] + ']'

    def forward(self, output, target):
        total_loss = torch.tensor(0.0).to(self.device)
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight*loss(output, target)
        return total_loss

    def get_evaluation_dict(self, output, target):
        eval_dict = super().get_evaluation_dict(output, target)
        for loss, weight in zip(self.losses, self.weights):
            loss_eval_dict = loss.get_evaluation_dict(output, target)
            for key, value in loss_eval_dict.items():
                eval_dict[key] = value
        return eval_dict

class LossDiceBCE(LossCombined):
    r"""A combination of Dice and Binary cross entropy."""
    def __init__(self, bce_weight=1., smooth=1., device='cuda:0'):
        super().__init__(losses=[LossDice(smooth=smooth), LossBCE()], 
            weights=[1., bce_weight], device=device)

class LossClassWeighted(LossAbstract):
    r"""A loss that weights different labels differently. Often, weights should
    be set inverse to the ratio of pixels of that class in the data so that
    classes with high representation (e.g. background) do not monopolize the 
    loss."""
    def __init__(self, loss, weights=None, nr_labels=None, device='cuda:0'):
        super().__init__(device)

        self.loss = loss
        if weights is None:
            assert nr_labels is not None, "Specify either weights or number of labels."
            self.class_weights = [1 for label_nr in range(nr_labels)]
        else:
            self.class_weights = weights
        # Set name
        self.name = 'LossClassWeighted[loss='+loss.name+'; weights='+str(tuple(self.class_weights))+']'
        # Set tensor class weights
        self.class_weights = torch.tensor(self.class_weights).to(self.device)
        self.added_weights = self.class_weights.sum()
        
    def forward(self, output, target):
        batch_loss = torch.tensor(0.0).to(self.device)
        for instance_output, instance_target in zip(output, target):
            instance_loss = torch.tensor(0.0).to(self.device)
            for out_channel_output, out_channel_target, weight in zip(instance_output, instance_target, self.class_weights):
                instance_loss += weight * self.loss(out_channel_output, 
                    out_channel_target)
            batch_loss += instance_loss / self.added_weights
        return batch_loss / len(output)

    def get_evaluation_dict(self, output, target):
        eval_dict = super().get_evaluation_dict(output, target)
        weighted_loss_values = [0 for weight in self.class_weights]
        for instance_output, instance_target in zip(output, target):
            for out_channel_output, out_channel_target, weight_ix in zip(instance_output, instance_target, range(len(weighted_loss_values))):
                instance_weighted_loss = self.loss(out_channel_output, out_channel_target)
                weighted_loss_values[weight_ix] += float(instance_weighted_loss.cpu())
        for weight_ix, loss_value in enumerate(weighted_loss_values):
            eval_dict[self.loss.name+'['+str(weight_ix)+']'] = loss_value / len(output)
        return eval_dict
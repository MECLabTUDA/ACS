# ------------------------------------------------------------------------------
# An autoencoder that reconstructs extracted features.
# ------------------------------------------------------------------------------

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from mp.models.autoencoding.autoencoder_linear import AutoencoderLinear
from mp.data.pytorch.transformation import torchvision_rescaling

class AutoencoderFeatured(AutoencoderLinear):
    r"""An autoencoder that recontracts features."""
    def __init__(self, input_shape, hidden_dim = [128, 64], 
        feature_model_name='AlexNet'):

        extractor_size = (3, 224, 224) # For AlexNet, TODO clean up and others
        features_size = 9216

        super().__init__(input_shape=[features_size], hidden_dim=hidden_dim)
        self.extractor_size = extractor_size
        self.feature_extractor = self.get_feature_extractor(feature_model_name)
        
    def preprocess_input(self, x):
        r"""Preprocessing that is done to the input before performing the 
        autoencoding, which is to say also to the target."""
        # Instead of doing a forward pass, we exclude the classifier
        # See https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
        x = torchvision_rescaling(x, size=self.extractor_size, resize=False)
        x = self.feature_extractor.features(x)
        x = self.feature_extractor.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def get_feature_extractor(self, model_name='AlexNet'):
        r"""Features are extracted from the input data. These are normalized 
        with the ImageNet statistics."""
        # Fetch pretrained model
        if model_name == 'AlexNet':  # input_size = 224 x 224
            feature_extractor = models.alexnet(pretrained=True)
        # Freeze pretrained parameters
        for param in feature_extractor.parameters():
            param.requires_grad = False
        return feature_extractor

    def to(self, device):
        super().to(device)
        self.feature_extractor.to(device)

        

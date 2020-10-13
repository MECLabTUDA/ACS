# ------------------------------------------------------------------------------
# A 2D convolutional autoencoder. Note that the input must be normalized between
# 0 and 1.
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from mp.models.autoencoding.autoencoder import Autoencoder

class AutoencoderCNN(Autoencoder):
    r"""A simple CNN autoencoder."""
    def __init__(self, input_shape, hidden_ch = [16, 4]):
        super().__init__(input_shape=input_shape)
        in_channels = self.input_shape[0]

        # Encoder layers
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, 
            out_channels=hidden_ch[0], kernel_size=3, stride=1, padding=1)  
        self.enc_conv2 = nn.Conv2d(hidden_ch[0], hidden_ch[1], 
            kernel_size=3, stride=1, padding=1)
        self.enc_pool = nn.MaxPool2d(2, 2)

        # Decoder layers
        self.dec_conv1 = nn.ConvTranspose2d(hidden_ch[1], hidden_ch[0], 
            kernel_size=2, stride=2, padding=0)
        self.dec_conv2 = nn.ConvTranspose2d(hidden_ch[0], in_channels, 
            kernel_size=2, stride=2, padding=0)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.enc_pool(x)
        x = F.relu(self.enc_conv2(x))
        x = self.enc_pool(x)
        return x

    def decode(self, x):
        x = F.relu(self.dec_conv1(x))
        x = F.sigmoid(self.dec_conv2(x)) # Input should be normed to [0, 1]
        return x

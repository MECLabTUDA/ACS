# ------------------------------------------------------------------------------
# This UNet model is modified from https://github.com/fepegar/unet
# (see https://zenodo.org/record/3522306#.X0FJnhmxVhE).
# ------------------------------------------------------------------------------

from typing import Optional
import torch.nn as nn
from mp.models.segmentation.segmentation_model import SegmentationModel
from mp.models.segmentation.model_utils import Encoder, EncodingBlock, Decoder, ConvolutionalBlock

class UNet(SegmentationModel):
    def __init__(
            self,
            input_shape,
            nr_labels,
            dimensions: int = 2,
            num_encoding_blocks: int = 5,
            out_channels_first_layer: int = 64,
            normalization: Optional[str] = None,
            pooling_type: str = 'max',
            upsampling_type: str = 'conv',
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            initial_dilation: Optional[int] = None,
            dropout: float = 0,
            monte_carlo_dropout: float = 0,
            ):
        super().__init__(input_shape=input_shape, nr_labels=nr_labels)

        in_channels = input_shape[0]

        depth = num_encoding_blocks - 1

        # Force padding if residual blocks
        if residual:
            padding = 1

        # Encoder
        self.encoder = Encoder(
            in_channels,
            out_channels_first_layer,
            dimensions,
            pooling_type,
            depth,
            normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=initial_dilation,
            dropout=dropout,
        )

        # Bottom (last encoding block)
        in_channels = self.encoder.out_channels
        if dimensions == 2:
            out_channels_first = 2 * in_channels
        else:
            out_channels_first = in_channels

        self.bottom_block = EncodingBlock(
            in_channels,
            out_channels_first,
            dimensions,
            normalization,
            pooling_type=None,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=self.encoder.dilation,
            dropout=dropout,
        )

        # Decoder
        if dimensions == 2:
            power = depth - 1
        elif dimensions == 3:
            power = depth
        in_channels = self.bottom_block.out_channels
        in_channels_skip_connection = out_channels_first_layer * 2**power
        num_decoding_blocks = depth
        self.decoder = Decoder(
            in_channels_skip_connection,
            dimensions,
            upsampling_type,
            num_decoding_blocks,
            normalization=normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=self.encoder.dilation,
            dropout=dropout,
        )

        # Monte Carlo dropout
        self.monte_carlo_layer = None
        if monte_carlo_dropout:
            dropout_class = getattr(nn, 'Dropout{}d'.format(dimensions))
            self.monte_carlo_layer = dropout_class(p=monte_carlo_dropout)

        # Classifier
        if dimensions == 2:
            in_channels = out_channels_first_layer
        elif dimensions == 3:
            in_channels = 2 * out_channels_first_layer
        self.classifier = ConvolutionalBlock(
            dimensions, in_channels, nr_labels,
            kernel_size=1, activation=None,
        )

    def forward(self, x):
        skip_connections, encoding = self.encoder(x)
        encoding = self.bottom_block(encoding)
        x = self.decoder(skip_connections, encoding)
        if self.monte_carlo_layer is not None:
            x = self.monte_carlo_layer(x)
        return self.classifier(x)

class UNet2D(UNet):
    def __init__(self, *args, **kwargs):
        assert len(args[0]) == 3, "Input shape must have dimensions channels, width, height. Received: {}".format(args[0])
        predef_kwargs = {}
        predef_kwargs['dimensions'] = 2
        predef_kwargs['num_encoding_blocks'] = 5
        predef_kwargs['out_channels_first_layer'] = 16 #64
        predef_kwargs['normalization'] = 'batch'
        preactivation = True
        # Added this so there is no error between the skip connection and 
        # feature mas shapes
        predef_kwargs['padding'] = True
        predef_kwargs.update(kwargs)
        super().__init__(*args, **predef_kwargs)

class UNet3D(UNet):
    def __init__(self, *args, **kwargs):
        assert len(args[0]) == 4, "Input shape must have dimensions channels, width, height, depth. Received: {}".format(args[0])
        predef_kwargs = {}
        predef_kwargs['dimensions'] = 3
        predef_kwargs['num_encoding_blocks'] = 4
        predef_kwargs['out_channels_first_layer'] = 8
        predef_kwargs['normalization'] = 'batch'
        predef_kwargs['upsampling_type'] = 'linear'
        predef_kwargs['padding'] = True
        predef_kwargs.update(kwargs)
        super().__init__(*args, **predef_kwargs)
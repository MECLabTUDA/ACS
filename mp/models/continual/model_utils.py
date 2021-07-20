import torch
import torch.nn as nn
import torch.nn.functional as F
from mp.models.segmentation.unet_fepegar import UNet2D

### UNet Wrapper ###
class UNet2D_dis(UNet2D):
    r"""Wrapper for UNet2D to access encoder and decoder seperately.
    """
    def __init__(self, *args, **kwargs):
        super(UNet2D_dis, self).__init__(*args, **kwargs)

    def forward_enc(self, x):
        skip_connections, encoding = self.encoder(x)
        encoding = self.bottom_block(encoding)
        return skip_connections, encoding

    def forward_dec(self, skip_connections, encoding):
        x = self.decoder(skip_connections, encoding)
        if self.monte_carlo_layer is not None:
            x = self.monte_carlo_layer(x)
        return self.classifier(x)

### MODULES ###
class EncoderStyle(nn.Module):
    r"""Style Encoder (VAE).
    """
    def __init__(self, in_channels):
        super(EncoderStyle, self).__init__()

        layers = []
        layers += [ConvBlock(in_channels=in_channels, out_channels=256)]
        layers += [ConvPoolBlock(in_channels=256, out_channels=64, pooling=False)]
        layers += [ConvPoolBlock(in_channels=64, out_channels=128, pooling=True)]
        layers += [ConvPoolBlock(in_channels=128, out_channels=128, pooling=False)]
        layers += [ConvPoolBlock(in_channels=128, out_channels=192, pooling=True)]
        layers += [ConvPoolBlock(in_channels=192, out_channels=192, pooling=False)]
        layers += [ConvPoolBlock(in_channels=192, out_channels=256, pooling=True)]

        global_pool = [nn.LeakyReLU(), nn.AdaptiveMaxPool2d(output_size=(3,3))]
        self.global_pool = nn.Sequential(*global_pool)

        self.layers = nn.Sequential(*layers)

        self.dense_mu = nn.Linear(in_features=3*3*256, out_features=1)
        self.dense_var = nn.Linear(in_features=3*3*256, out_features=1)
    
    def forward(self, x):
        x = self.layers(x)
        x = self.global_pool(x)
        mu = self.dense_mu(x.view(x.shape[0], -1))
        log_var = self.dense_var(x.view(x.shape[0], -1))
        return [mu, log_var]

class LatentScaler(nn.Module):
    r"""Scales samples from style encoding to be injected into the generator.
    """
    def __init__(self, in_features):
        super(LatentScaler, self).__init__()

        layers = [nn.Linear(in_features=in_features, out_features=500), nn.LeakyReLU()]
        layers += [nn.Linear(in_features=500, out_features=1024), nn.LeakyReLU()]

        for _ in range(0, 2):
            layers += [nn.Linear(in_features=1024, out_features=1024), nn.LeakyReLU()]

        layers += [nn.Linear(in_features=1024, out_features=2560), nn.Tanh()]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x).reshape(x.shape[0],10,-1) # 10 occurences a 256 filters
        return x

class Generator(nn.Module):
    r"""Generator using content encoding, scaled style encoding (see LatentScaler) and domain_code to generate images.
    """
    def __init__(self, in_channels, out_channels, domain_code_size):
        super(Generator, self).__init__()

        layers_BCIN = [ResBlockBCIN(in_channels=in_channels, out_channels=in_channels, layer_id=0, stride=1, padding=1, domain_code_size=domain_code_size)]
        for i in range(0,4):
            layers_BCIN += [ResBlockBCIN(in_channels=in_channels, out_channels=in_channels, layer_id=i+1, stride=1, padding=1, domain_code_size=domain_code_size)]

        layers = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU()]
        layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU()]
        layers += [nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU()]
        layers += [nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU()]
        layers += [nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=7, stride=1, padding=3), nn.Sigmoid()]
        

        self.layers_BCIN = MultiInSequential(*layers_BCIN)
        self.layers = nn.Sequential(*layers)

    def forward(self, content, latent_scale, domain_code):
        content, latent_scale, domain_code = self.layers_BCIN(content, latent_scale, domain_code)
        x = self.layers(content)
        return x

class DiscriminatorDomain(nn.Module):
    r"""Domain Discriminator.
    """
    def __init__(self, in_channels, domain_code_size, max_channels=512, kernel_size=4, stride=2):
        super(DiscriminatorDomain, self).__init__()

        layers = [ConvBlockBCIN(in_channels=in_channels, out_channels=64, kernel_size=kernel_size, stride=stride, domain_code_size=domain_code_size)]
        layers += [ConvBlockBCIN(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, domain_code_size=domain_code_size)]
        layers += [ConvBlockBCIN(in_channels=128, out_channels=max_channels//2, kernel_size=kernel_size, stride=stride, domain_code_size=domain_code_size)]
        layers += [ConvBlockBCIN(in_channels=max_channels//2, out_channels=max_channels, kernel_size=kernel_size, stride=stride, domain_code_size=domain_code_size)]
        layers += [ConvBlockBCIN(in_channels=max_channels, out_channels=1, kernel_size=kernel_size, stride=stride, domain_code_size=domain_code_size, normalization='None')]
        self.layers = MultiInSequential(*layers)

        self.linear = nn.Linear(in_features=7**2, out_features=1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x, domain_code):
        x, domain_code = self.layers(x, domain_code)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)
        return x

class DiscriminatorContent(nn.Module):
    r"""Unet-style Content Discriminator.
    """
    def __init__(self, in_channels, domain_code_size, max_channels=512, kernel_size=3, stride=2):
        super(DiscriminatorContent, self).__init__()

        self.in_channels = 16
        self.in_channels_max = 128
        self.out_channels = 32
        self.out_channels_max = 256
        padding = 1

        self.conv_0 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels*2**1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm_0 = nn.BatchNorm2d(self.in_channels*2**1)
        self.activation_0 = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels=self.in_channels*2**1, out_channels=self.in_channels*2**2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm_1 = nn.BatchNorm2d(self.in_channels*2**2)
        self.activation_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=self.in_channels*2**2, out_channels=self.in_channels*2**3, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm_2 = nn.BatchNorm2d(self.in_channels*2**3)
        self.activation_2 = nn.ReLU()
        self.conv_3 = nn.Conv2d(in_channels=self.in_channels*2**3, out_channels=self.in_channels*2**4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm_3 = nn.BatchNorm2d(self.in_channels*2**4)
        self.activation_3 = nn.ReLU()
        self.conv_4 = nn.Conv2d(in_channels=self.in_channels*2**4, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm_4 = nn.BatchNorm2d(1)
        self.activation_4 = nn.ReLU()
        
        self.dense = nn.Linear(in_features = 8**2, out_features=domain_code_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, skip_connections, content_x):
        out = self.conv_0(skip_connections[0])
        out = self.norm_0(out)
        out = self.activation_0(out)
        out = self.conv_1(skip_connections[1] + out)
        out = self.norm_1(out)
        out = self.activation_1(out)
        out = self.conv_2(skip_connections[2] + out)
        out = self.norm_2(out)
        out = self.activation_2(out)
        out = self.conv_3(skip_connections[3] + out)
        out = self.norm_3(out)
        out = self.activation_3(out)
        out = self.conv_4(content_x + out)
        out = self.norm_4(out)
        out = self.activation_4(out)
        out = self.dense(out.reshape(content_x.shape[0], -1))
        out = self.softmax(out)
        return out
    
    def center_crop(self, skip_connection, x):
        skip_shape = torch.tensor(skip_connection.shape)
        x_shape = torch.tensor(x.shape)
        crop = skip_shape[2:] - x_shape[2:]
        half_crop = crop // 2
        # If skip_connection is 10, 20, 30 and x is (6, 14, 12)
        # Then pad will be (-2, -2, -3, -3, -9, -9)
        pad = -torch.stack((half_crop, half_crop)).t().flatten()
        skip_connection = F.pad(skip_connection, pad.tolist())
        return skip_connection

### BUILDING BLOCKS ###
class ConvBlock(nn.Module):
    r"""Convolutional Block with normalization and activation.
    """
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, activation=nn.LeakyReLU, normalization='Instance'):
        super(ConvBlock, self).__init__() 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.normalization = normalization
        if self.normalization == 'Instance':
            self.norm = nn.InstanceNorm2d(num_features=out_channels, affine=False) # not learnable
        if self.normalization =='BatchNorm':
            self.norm = nn.BatchNorm2d(num_features=out_channels)
        
        self.activation = activation()

    def forward(self,x):
        x = self.conv(x)
        if self.normalization in ['Instance', 'BatchNorm']:
            x = self.norm(x)
        x = self.activation(x)
        return x

class ConvPoolBlock(nn.Module):
    r"""Convolutional Block with normalization, activation and pooling.
    """
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, pooling=True, activation=nn.LeakyReLU):
        super(ConvPoolBlock, self).__init__()

        self.pooling = pooling

        self.norm= nn.InstanceNorm2d(num_features=out_channels, affine=False) # not learnable
        self.activation = activation()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)

        if self.pooling:
            x = self.pool(x)
        return x

class ConvBlockBCIN(nn.Module):
    r"""Convolutional Block with BCIN normalization and activation.
    """
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, activation=nn.LeakyReLU, domain_code_size=10, normalization='BCIN'):
        super(ConvBlockBCIN, self).__init__() 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = BCIN(out_channels, domain_code_size) # not learnable
        self.activation = activation()

        self.normalization = normalization

    def forward(self, x, domain_code):
        x = self.conv(x)
        if self.normalization == 'BCIN': 
            x = self.norm(x, domain_code)
        x = self.activation(x)
        return x, domain_code

class ResBlockIN(nn.Module):
    r"""Residual Block consisting of two convolutions with skip connection, instance normalization and activation.
    """
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, activation=nn.ReLU):
        super(ResBlockIN, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.norm0 = nn.InstanceNorm2d(num_features=out_channels, affine=False) # not learnable
        self.norm1 = nn.InstanceNorm2d(num_features=out_channels, affine=False) # not learnable
        self.activation = activation()
    
    def forward(self, x):
        x_in = x
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x += self.center_crop(x_in, x)
        return x

    def center_crop(self, skip_connection, x):
        skip_shape = torch.tensor(skip_connection.shape)
        x_shape = torch.tensor(x.shape)
        crop = skip_shape[2:] - x_shape[2:]
        half_crop = crop // 2
        # If skip_connection is 10, 20, 30 and x is (6, 14, 12)
        # Then pad will be (-2, -2, -3, -3, -9, -9)
        pad = -torch.stack((half_crop, half_crop)).t().flatten()
        skip_connection = F.pad(skip_connection, pad.tolist())
        return skip_connection

class ResBlockBCIN(nn.Module):
    r"""Residual Block consisting of two convolutions with skip connection, BCIN normalization and activation.
    """
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, activation=nn.ReLU, domain_code_size=10, layer_id=0):
        super(ResBlockBCIN, self).__init__()
        self.conv0 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.norm0 = BCIN(num_features=out_channels, domain_code_size=domain_code_size, affine=True) # learnable
        self.norm1 = BCIN(num_features=out_channels, domain_code_size=domain_code_size, affine=True) # learnable
        self.activation = activation()

        self.layer_id = layer_id

    def forward(self, x, latent_scale, domain_code):
        
        x_in = x
        x = self.conv0(x)
        x = torch.mul(x, latent_scale[:,self.layer_id*2,:][:,:,None,None])
        x = self.norm0(x, domain_code)
        
        x = self.activation(x)

        x = self.conv1(x)
        x = torch.mul(x, latent_scale[:,self.layer_id*2+1,:][:,:,None,None])
        x = self.norm1(x, domain_code)

        x += self.center_crop(x_in, x)

        return x, latent_scale, domain_code

    def center_crop(self, skip_connection, x):
        skip_shape = torch.tensor(skip_connection.shape)
        x_shape = torch.tensor(x.shape)
        crop = skip_shape[2:] - x_shape[2:]
        half_crop = crop // 2
        # If skip_connection is 10, 20, 30 and x is (6, 14, 12)
        # Then pad will be (-2, -2, -3, -3, -9, -9)
        pad = -torch.stack((half_crop, half_crop)).t().flatten()
        skip_connection = F.pad(skip_connection, pad.tolist())
        return skip_connection

### NORMALIZATION ###
class BCIN(nn.Module):
    r"""Central Biasing Instance Normalization
    https://arxiv.org/abs/1806.10050
    """
    def __init__(self, num_features, domain_code_size, affine=True, instance_norm=False, batch_norm=False):
        super(BCIN, self).__init__()
        self.W = nn.Parameter(torch.rand(domain_code_size), requires_grad=affine)
        self.b = nn.Parameter(torch.rand(1), requires_grad=affine)
        self.activation = nn.Tanh()

        self.instance_norm = instance_norm
        if self.instance_norm:
            print('Using instance_norm instead of BCIN')
        self.i_norm = torch.nn.InstanceNorm2d(num_features=num_features)

        self.batch_norm = batch_norm
        if self.instance_norm:
            print('Using batch_norm instead of BCIN')
        self.b_norm = torch.nn.BatchNorm2d(num_features=num_features)

    def forward(self, x, domain_code):
        x_var = torch.sqrt(torch.var(x, (1,2,3))) # instance std
        x_mean = torch.mean(x, (1,2,3)) # instance mean
        bias = torch.matmul(domain_code, self.W) * self.b
        bias_scaled = self.activation(bias)


        if self.instance_norm:
            return self.i_norm(x)
        if self.batch_norm:
            return self.b_norm(x)

        return ((x-x_mean[:,None,None,None]) / x_var[:,None,None,None]) + bias_scaled[:,None,None,None]

### HELPER MODULES ###
class MultiInSequential(nn.Sequential):
    r"""Sequential class that allows multiple inputs for forward function
    """
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input

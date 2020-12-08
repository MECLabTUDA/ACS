import torch
import torch.nn as nn
import torch.nn.functional as F

# modules
class EncoderContent(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super(EncoderContent, self).__init__()
        
        layers = [ConvBlock(in_channels=in_channels, normalization='Instance', kernel_size=3, stride=2, padding=1)]
        for _ in range(0, 2):
            layers += [ConvBlock(normalization='Instance', kernel_size=3, stride=2, padding=1)]
        
        for _ in range(0, 4):
            layers += [ResBlockIN(out_channels=out_channels)]

        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        x = self.layers(x)
        return x
        
class EncoderStyle(nn.Module):
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

        # TODO: GlobalPool2d
        global_pool = [nn.LeakyReLU(), nn.AdaptiveMaxPool2d(output_size=(3,3))]
        self.global_pool = nn.Sequential(*global_pool)

        self.layers = nn.Sequential(*layers)

        self.dense_mu = nn.Linear(in_features=3*3*256, out_features=1)
        self.dense_var = nn.Linear(in_features=3*3*256, out_features=1)
    
    def forward(self, x):
        x = self.layers(x)
        x = self.global_pool(x)
        mu = self.dense_mu(x.view(x.shape[0], -1))
        var = self.dense_var(x.view(x.shape[0], -1))
        return [mu, var]

class LatentScaler(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        layers_BCIN = [ResBlockBCIN(in_channels=in_channels, out_channels=in_channels, layer_id=0, stride=1, padding=0)]

        for i in range(0,3):
            layers_BCIN += [ResBlockBCIN(in_channels=in_channels, out_channels=in_channels, layer_id=i+1, stride=1, padding=0)]

        layers = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=128, kernel_size=2, stride=2), nn.ReLU()]
        layers += [nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2), nn.ReLU()]

        layers += [nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=2, stride=2), nn.Tanh()]

        self.layers_BCIN = nn.Sequential(*layers_BCIN)
        self.layers = nn.Sequential(*layers)

    def forward(self, content, latent_scale, domain_code):
        x = content
        x = self.layers_BCIN([[content, latent_scale, domain_code]])[0][0]
        x = self.layers(x)
        return x

class DiscriminatorContent(nn.Module):
    def __init__(self, in_channels, max_channels=512, kernel_size=4, stride=2):
        super(DiscriminatorContent, self).__init__()

        layers = [ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=kernel_size, stride=stride, normalization='None')]
        layers += [ConvBlock(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, normalization='None')]
        layers += [ConvBlock(in_channels=128, out_channels=max_channels//2, kernel_size=kernel_size, stride=stride, normalization='None')]
        layers += [ConvBlock(in_channels=max_channels//2, out_channels=max_channels, kernel_size=kernel_size, stride=1, normalization='None')]
        layers += [ConvBlock(in_channels=max_channels, out_channels=1, kernel_size=kernel_size, stride=1, normalization='None')]
        self.layers = nn.Sequential(*layers)
        
        # added TODO look up PatchGAN
        self.linear = nn.Linear(in_features=6**2, out_features=1) # 21**2
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)
        x = self.activation(x)
        return x

class DiscriminatorStructureMulti(nn.Module):
    def __init__(self, in_channels, domain_code_size, max_channels=512, kernel_size=4, stride=2):
        super(DiscriminatorStructureMulti, self).__init__()

        layers = [ConvBlockBCIN(in_channels=in_channels, out_channels=64, kernel_size=kernel_size, stride=stride, domain_code_size=domain_code_size)]
        layers += [ConvBlockBCIN(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, domain_code_size=domain_code_size)]
        layers += [ConvBlockBCIN(in_channels=128, out_channels=max_channels//2, kernel_size=kernel_size, stride=stride, domain_code_size=domain_code_size)]
        layers += [ConvBlockBCIN(in_channels=max_channels//2, out_channels=max_channels, kernel_size=kernel_size, stride=stride, domain_code_size=domain_code_size)]
        layers += [ConvBlockBCIN(in_channels=max_channels, out_channels=1, kernel_size=kernel_size, stride=stride, domain_code_size=domain_code_size, normalization='None')]
        self.layers = nn.Sequential(*layers)

        # added TODO look up PatchGAN
        self.linear = nn.Linear(in_features=7**2, out_features=1) # 24**2
        self.activation = nn.Sigmoid()

    def forward(self, x, domain_code):
        x = self.layers([[x, domain_code]])[0][0]
        x = x.view(x.shape[0],-1)
        x = self.linear(x)
        x = self.activation(x)
        return x

# layers / blocks
class ConvBlock(nn.Module):
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
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, activation=nn.LeakyReLU, domain_code_size=10, normalization='BCIN'):
        super(ConvBlockBCIN, self).__init__() 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = BCIN(out_channels, domain_code_size) # not learnable
        self.activation = activation()

        self.normalization = normalization

    def forward(self, inputs):
        x, domain_code = inputs[0][0], inputs[0][1]
        x = self.conv(x)
        if self.normalization == 'BCIN': 
            x = self.norm(x, domain_code)
        x = self.activation(x)
        return [[x, domain_code]]

class ResBlockIN(nn.Module):
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
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, activation=nn.ReLU, domain_code_size=10, layer_id=0):
        super(ResBlockBCIN, self).__init__()
        self.conv0 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        # self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.norm0 = BCIN(num_features=out_channels, domain_code_size=domain_code_size, affine=True) # learnable
        self.norm1 = BCIN(num_features=out_channels, domain_code_size=domain_code_size, affine=True) # learnable
        self.activation = activation()

        self.layer_id = layer_id

    def forward(self, inputs):
        x, latent_scale, domain_code = inputs[0][0], inputs[0][1], inputs[0][2]
        
        x_in = x
        x = self.conv0(x)
        x = torch.mul(x, latent_scale[:,self.layer_id*2,:][:,:,None,None])
        x = self.norm0(x, domain_code)
        
        x = self.activation(x)

        x = self.conv1(x)
        x = torch.mul(x, latent_scale[:,self.layer_id*2+1,:][:,:,None,None])
        x = self.norm1(x, domain_code)

        x += self.center_crop(x_in, x)

        return [[x, latent_scale, domain_code]]

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

class LatentScaleLayer(nn.Module):
    def __init__(self):
        super(LatentScaleLayer, self).__init__()
    
    def forward(self, x, latent_scale):
        return torch.matmul(x, latent_scale[:,None,None,None])
    
# normalization

class BCIN(nn.Module):
    def __init__(self, num_features, domain_code_size, affine=True):
        super(BCIN, self).__init__()
        self.domain_code_size = domain_code_size
        self.W = nn.Parameter(torch.rand(domain_code_size), requires_grad=affine)
        self.b = nn.Parameter(torch.rand(1), requires_grad=affine)

        self.activation = nn.Tanh()

    def forward(self, x, domain_code):
        x_var = torch.sqrt(torch.var(x, (1,2,3))) # instance std
        x_mean = torch.mean(x, (1,2,3)) # instance mean

        bias = torch.matmul(domain_code, self.W) * self.b
        bias_scaled = self.activation(bias)

        return ((x-x_mean[:,None,None,None]) / x_var[:,None,None,None]) + bias_scaled[:,None,None,None]
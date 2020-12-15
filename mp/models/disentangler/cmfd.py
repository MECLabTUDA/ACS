from mp.models.model import Model
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.models.disentangler.model_utils import * # EncoderContent, DiscriminatorContent, EncoderStyle, LatentScaler, BCIN, Generator
from torch.autograd import Variable
import torch.optim as optim

class CMFD(Model):
    def __init__(self,
            # self,
            input_shape=(3,256,256),
            nr_labels=2,
            latent_channels=128,
            domain_code_size=10,
            latent_scaler_sample_size=250
            # nr_labels,
            # dimensions: int = 2,
            # num_encoding_blocks: int = 5,
            # out_channels_first_layer: int = 64,
            # normalization: Optional[str] = None,
            # pooling_type: str = 'max',
            # upsampling_type: str = 'conv',
            # preactivation: bool = False,
            # residual: bool = False,
            # padding: int = 0,
            # padding_mode: str = 'zeros',
            # activation: Optional[str] = 'ReLU',
            # initial_dilation: Optional[int] = None,
            # dropout: float = 0,
            # monte_carlo_dropout: float = 0,
            ):

        super(CMFD, self).__init__()#input_shape=input_shape, nr_labels=nr_labels)

        self.input_shape = input_shape
        self.latent_scaler_sample_size = latent_scaler_sample_size
        self.domain_code_size = domain_code_size
        self.device = 'cpu'

        # encoder
        self.enc_con = EncoderContent(in_channels=self.input_shape[0], out_channels=latent_channels)
        self.enc_sty = EncoderStyle(in_channels=self.input_shape[0])

        # discriminator
        self.dis_con = DiscriminatorContent(in_channels=latent_channels, max_channels=256, kernel_size=3, stride=1)
        self.dis_struc = DiscriminatorStructureMulti(in_channels=self.input_shape[0], domain_code_size=self.domain_code_size, max_channels=256, kernel_size=3, stride=2)
        self.dis_mul = DiscriminatorStructureMulti(in_channels=self.input_shape[0], domain_code_size=self.domain_code_size, max_channels=256, kernel_size=3, stride=2)
        
        # generator
        self.gen = Generator(in_channels=latent_channels, out_channels=self.input_shape[0])

        # segmentor
        self.nr_labels = nr_labels
        self.unet = UNet2D(self.input_shape, nr_labels)

        # latent scaler
        self.latent_scaler = LatentScaler(in_features=self.latent_scaler_sample_size)
    
    def parallel(self, device_ids=[6,7]):
        self.enc_con = nn.DataParallel(self.enc_con, device_ids=device_ids)
        self.enc_sty = nn.DataParallel(self.enc_sty, device_ids=device_ids)

        self.dis_con = nn.DataParallel(self.dis_con, device_ids=device_ids)
        self.dis_struc = nn.DataParallel(self.enc_con, device_ids=device_ids)
        self.dis_mul = nn.DataParallel(self.dis_mul, device_ids=device_ids)

        self.gen = nn.DataParallel(self.gen, device_ids=device_ids)
        self.unet = nn.DataParallel(self.unet, device_ids=device_ids)
    
    def set_optimizers(self, optimizer=optim.Adam, lr=1e-4):
        self.enc_con_optim = optimizer(self.enc_con.parameters(),lr=lr)
        self.enc_sty_optim = optimizer(self.enc_sty.parameters(),lr=lr)

        self.dis_con_optim = optimizer(self.dis_con.parameters(),lr=lr)
        self.dis_struc_optim = optimizer(self.dis_struc.parameters(),lr=lr)
        self.dis_mul_optim = optimizer(self.dis_mul.parameters(),lr=lr)

        self.gen_optim = optimizer(self.gen.parameters(),lr=lr)
        self.unet_optim = optimizer(self.unet.parameters(),lr=lr)
    
    def zero_grad_optimizers(self):
        self.enc_con_optim.zero_grad()
        self.enc_sty_optim.zero_grad()

        self.dis_con_optim.zero_grad()
        self.dis_struc_optim.zero_grad()
        self.dis_mul_optim.zero_grad()

        self.gen_optim.zero_grad()
        self.unet_optim.zero_grad()

    def step_optimizers(self):
        self.enc_con_optim.step()
        self.enc_sty_optim.step()

        self.dis_con_optim.step()
        self.dis_struc_optim.step()
        self.dis_mul_optim.step()

        self.gen_optim.step()
        self.unet_optim.step()

    def forward(self, x, domain_code):
        x_hat = self.forward_encoder_generator(x, domain_code.to(self.device))
        x_seg = self.forward_segmentation(x)
        return x_seg

    def forward_encoder(self, x, sample_size=0):
        if sample_size <= 0:
            sample_size = self.latent_scaler_sample_size
        content = self.enc_con(x)
        style_mu_var = self.enc_sty(x)
        eps = Variable(torch.randn(len(style_mu_var[0]),sample_size)).to(self.device)
        style_sample = style_mu_var[0] + torch.exp(style_mu_var[1] / 2) * eps

        return content, style_sample

    def forward_generator(self, content, latent_scale, domain_code):
        x_hat = self.gen.forward(content, latent_scale, domain_code)
        return x_hat

    def forward_encoder_generator(self, x, domain_code):
        content, style_sample = self.forward_encoder(x)
        latent_scale = self.latent_scaler(style_sample)
        x_hat = self.gen.forward(content, latent_scale, domain_code)

        return x_hat
    
    def forward_segmentation(self, x):
        x_seg = self.unet(x)
        return x_seg

    def forward_joint_density_match(self, x):
        x_hat = self.forward_segmentation(x)
        return torch.cat((x, x_hat), 1) 

    def forward_struct_discriminator(self, x, domain_code):
        x = self.dis_struc(x, domain_code)
        return x

    def forward_multi_discriminator(self, x, domain_code):
        x = self.dis_mul(x, domain_code)
        return x

    def forward_content_discriminator(self, x):
        x = self.dis_con(x)
        return x
    
    def sample_z(self, shape):
        x = torch.rand(shape).to(self.device)
        return x

    def to_device(self, device):
        self.device = device
        self.to(device)

if __name__ == '__main__':
    import torch
    in_channels = 3
    sample = torch.rand(8, in_channels, 256, 256)

    # enc_con = EncoderContent(in_channels=3)
    # content = enc_con.forward(sample)
    # print('CONTENT SHAPE', content.shape)
    # dis_con = DiscriminatorContent(in_channels=256)
    content = torch.rand(8,256,230,230)
    # content_label = dis_con.forward(content)
    # print('DISC CONTENT LABEL', content_label.shape)
    # exit(32)
    enc_sty = EncoderStyle(in_channels=in_channels)
    style_mu_var = enc_sty.forward(sample)
    # print(style_mu_var, style_mu_var.shape)
    # print('MU VAR SHAPE', style_mu.shape, style_var.shape)

    # Using reparameterization trick to sample from a gaussian
    # from torch.autograd import Variable
    eps = Variable(torch.randn(len(style_mu_var[0]),250))
    style_sample = style_mu_var[0] + torch.exp(style_mu_var[1] / 2) * eps

    gen = Generator(in_channels=256, out_channels=4)
    domain_code = torch.zeros(10)
    domain_code[3] = 1
    gen_out = gen.forward(content, domain_code, style_sample)
    print(gen_out.shape)
    # # ls = LatentScaler(in_features=28)
    # # ls.forward(torch.rand(8, 28))

    # from mp.models.segmentation.unet_fepegar import UNet2D

    # unet = UNet2D((4,25 6,256), 2)
    # print('UNET SHAPE', unet.forward(torch.rand(8,4,256,256)).shape)

    # bcin = BCIN(256, 10)
    # norm = bcin.forward(torch.rand(8,256,30,30), torch.rand(8,10))
    # print(norm.shape)

    dis_strmul = DiscriminatorStructureMulti(in_channels=3, domain_code_size=10)
    labels = dis_strmul.forward(sample, domain_code)
    print(labels.shape)
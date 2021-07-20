from mp.models.model import Model
from mp.models.continual.model_utils import *
from torch.autograd import Variable
import torch.optim as optim

class CAS(Model):
    r""" Continual Adversarial Segmenter as proposed in Adversarial Continual Learning for Multi-Domain Segmentation 
    by Memmel et. al, 2021
    """
    def __init__(self,
            input_shape=(1,256,256),
            nr_labels=2,
            domain_code_size=10,
            latent_scaler_sample_size=250,
            unet_dropout = 0,
            unet_monte_carlo_dropout = 0,
            unet_preactivation= False
            ):
        r"""Constructor
        
        Args:
            input_shape (tuple of int): input shape of the images
            nr_labels (int): number of labels for the segmentation
            domain_code_size (int): size of domain code vector
            latent_scaler_sample_size (int): number of samples to be used to generate latent scale
            unet_dropout (float): dropout probability for the U-Net
            unet_monte_carlo_dropout (float): monte carlo dropout probability for the U-Net
            unet_preactivation (boolean): whether to use U-Net pre-activations
        """
        super(CAS, self).__init__()

        self.input_shape = input_shape
        self.latent_scaler_sample_size = latent_scaler_sample_size
        self.domain_code_size = domain_code_size

        # UNet -> segmentor and content encoder
        self.nr_labels = nr_labels
        self.unet = UNet2D_dis(self.input_shape, self.nr_labels, dropout=unet_dropout, monte_carlo_dropout=unet_monte_carlo_dropout, preactivation=unet_preactivation)
        self.enc_con_out_dim = self.unet.bottom_block.out_channels

        # encoder
        self.enc_sty = EncoderStyle(in_channels=self.input_shape[0])

        # discriminator
        self.dis_dom = DiscriminatorDomain(in_channels=self.input_shape[0], domain_code_size=self.domain_code_size, max_channels=256, kernel_size=3, stride=2)
        self.dis_con = DiscriminatorContent(in_channels=self.input_shape[0], domain_code_size=self.domain_code_size, max_channels=256, kernel_size=3, stride=2)
        
        # generator
        self.gen = Generator(in_channels=self.enc_con_out_dim, out_channels=self.input_shape[0], domain_code_size=self.domain_code_size)

        # latent scaler
        self.latent_scaler = LatentScaler(in_features=self.latent_scaler_sample_size)
    
    def set_data_parallel(self, device_ids):
        r"""Wrap each module in data parallel structure

        Args:
            device_ids (list): device ids of the GPUs
        """
        self.unet.encoder = nn.DataParallel(self.unet.encoder, device_ids)
        self.unet.bottom_block = nn.DataParallel(self.unet.bottom_block, device_ids)
        self.unet.decoder = nn.DataParallel(self.unet.decoder, device_ids)
        self.unet.classifier = nn.DataParallel(self.unet.classifier, device_ids)
        if self.unet.monte_carlo_layer is not None:
            self.unet.monte_carlo_layer = nn.DataParallel(self.unet.monte_carlo_layer, device_ids)

        self.enc_sty = nn.DataParallel(self.enc_sty, device_ids)
        self.dis_con = nn.DataParallel(self.dis_con, device_ids)
        self.dis_dom = nn.DataParallel(self.dis_dom, device_ids)
        self.gen = nn.DataParallel(self.gen, device_ids)
        self.latent_scaler = nn.DataParallel(self.latent_scaler, device_ids)
        
    def set_optimizers(self, optimizer=optim.Adam, lr=1e-4):
        r"""Set optimizers for all modules
        
        Args:
            optimizer (torch.nn.optim): optimizer to use
            lr (float): learning rate to use
        """
        self.enc_sty_optim = optimizer(self.enc_sty.parameters(),lr=lr)
        self.dis_con_optim = optimizer(self.dis_con.parameters(),lr=lr)
        self.dis_dom_optim = optimizer(self.dis_dom.parameters(),lr=lr)
        self.gen_optim = optimizer(self.gen.parameters(),lr=lr)
        
        self.ls_optim = optimizer(self.latent_scaler.parameters(), lr=lr)
        
        self.unet_optim = optimizer(self.unet.parameters(),lr=lr)
        self.unet_decoder_optim = optimizer(self.unet.decoder.parameters(),lr=lr)
        self.unet_classifier_optim = optimizer(self.unet.classifier.parameters(),lr=lr)

    def step_optim_disc(self):
        r"""Step discriminator optimizers"""
        self.dis_con_optim.step()
        self.dis_dom_optim.step()
    
    def step_optim_enc_misc(self):
        r"""Step encoder and misc. optimizers"""
        self.enc_sty_optim.step()
        self.gen_optim.step()
        self.unet_optim.step()

    def zero_grad_optim_enc_misc(self):
        """Zero grad encoder and misc. optimizers"""
        self.enc_sty_optim.zero_grad()
        self.gen_optim.zero_grad()
        self.unet_optim.zero_grad()

    def zero_grad_optim_disc(self):
        """Zero grad discriminator optimizers"""
        self.dis_con_optim.zero_grad()
        self.dis_dom_optim.zero_grad()

    def forward(self, x):
        r"""Full forward pass (segmentation)
        
        Args:
            x (torch.Tensor): input batch
        
        Returns:
            (torch.Tensor): segmentated batch
        """
        skip_connections, content, style_sample = self.forward_enc(x)
        x_seg = self.forward_dec(skip_connections, content)
        return x_seg

    def forward_enc(self, x, sample_size=0):
        r"""Forward encoding structure
        
        Args:
            x (torch.Tensor): input batch
            sample_size (int): how many samples to draw for the latent scale
        Returns:
            (torch.Tensor): skip connections of the U-Net
            (torch.Tensor): content encoding
            (torch.Tensor): style sample
        """

        # if no sample size is selected, use latent_scaler_sample_size
        if sample_size <= 0:
            sample_size = self.latent_scaler_sample_size
        
        skip_connections, content = self.unet.forward_enc(x)

        # forward style encoder and sample from distribution
        style_mu_var = self.enc_sty(x)
        eps = Variable(torch.randn(len(style_mu_var[0]),sample_size))
        if style_mu_var[0].is_cuda:
            eps = eps.to(style_mu_var[0].get_device())
            
        style_sample = style_mu_var[0] + torch.exp(style_mu_var[1] / 2) * eps
        
        return skip_connections, content, style_sample

    def forward_style_enc(self, x):
        r"""Forward style encoder
        
        Args:
            x (torch.Tensor): input batch
        
        Returns:
            (torch.Tensor): style encoding [mu, log_var]
        """
        return self.enc_sty(x)

    def forward_dec(self, skip_connections, content):
        r"""Forward U-Net decoder
        
        Args:
            skip_connections (torch.Tensor): skip_connections of the U-Net
            content (torch.Tensor): content encoding

        Returns:
            (torch.Tensor): segmentation
        """
        return self.unet.forward_dec(skip_connections, content)

    def forward_gen(self, content, latent_scale, domain_code):
        r"""Forward generator
        
        Args:
            content (torch.Tensor): content encoding
            latent_scale (torch.Tensor): latentscale
            domain_code (torch.Tensor): domain code

        Returns:
            (torch.Tensor): generated image
        """
        return self.gen.forward(content, latent_scale, domain_code)

    def forward_dom_dis(self, x, domain_code):
        r"""Forward domain discriminator
        
        Args:
            x (torch.Tensor): input batch
            domain_code (torch.Tensor): domain code

        Returns:
            (torch.Tensor): decision if domain is present
        """
        x = self.dis_dom(x, domain_code)
        return x

    def forward_con_dis(self, skip_connections_x, x):
        r"""Forward content discriminator
        
        Args:
            skip_connections_x (torch.Tensor): skip connections of the U-Net
            x (torch.Tensor): input batch

        Returns:
            (torch.Tensor): decision if content is present
        """
        x = self.dis_con(skip_connections_x, x)
        return x
    
    def sample_z(self, shape):
        r"""Sample from normal distribution"""
        return torch.rand(shape)
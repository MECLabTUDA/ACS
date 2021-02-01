from mp.models.model import Model
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.models.disentangler.model_utils import * # EncoderContent, DiscriminatorContent, EncoderStyle, LatentScaler, BCIN, Generator
from torch.autograd import Variable
import torch.optim as optim

class CMFD(Model):
    r''' Cross Modality Feature Disentangler.
    '''
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
        
        """
        super(CMFD, self).__init__()

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
        self.dis_con = DiscriminatorContent(in_channels=self.enc_con_out_dim, max_channels=256, kernel_size=3, stride=1, domain_code_size=self.domain_code_size)
        # self.dis_struc = DiscriminatorStructureMulti(in_channels=self.input_shape[0], domain_code_size=self.domain_code_size, max_channels=256, kernel_size=3, stride=2)
        self.dis_dom = DiscriminatorStructureMulti(in_channels=self.input_shape[0], domain_code_size=self.domain_code_size, max_channels=256, kernel_size=3, stride=2)
        
        self.dis_con = DiscriminatorUnet(in_channels=self.input_shape[0], domain_code_size=self.domain_code_size, max_channels=256, kernel_size=3, stride=2)
        
        # generator
        self.gen = Generator(in_channels=self.enc_con_out_dim, out_channels=self.input_shape[0], domain_code_size=self.domain_code_size)

        # latent scaler
        self.latent_scaler = LatentScaler(in_features=self.latent_scaler_sample_size)
    
    def set_data_parallel(self, device_ids):
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
        self.enc_sty_optim = optimizer(self.enc_sty.parameters(),lr=lr)
        self.dis_con_optim = optimizer(self.dis_con.parameters(),lr=lr)
        self.dis_dom_optim = optimizer(self.dis_dom.parameters(),lr=lr)
        self.gen_optim = optimizer(self.gen.parameters(),lr=lr)

        self.unet_optim = optimizer(self.unet.parameters(),lr=lr)
        # self.unet_encoder_optim = optimizer(self.unet.encoder.parameters(),lr=lr)
        # self.unet_bottom_block_optim = optimizer(self.unet.bottom_block.parameters(),lr=lr)
        self.unet_decoder_optim = optimizer(self.unet.decoder.parameters(),lr=lr)
        self.unet_classifier_optim = optimizer(self.unet.classifier.parameters(),lr=lr)

    def zero_grad_optimizers(self):
        self.enc_sty_optim.zero_grad()
        self.dis_con_optim.zero_grad()
        self.dis_dom_optim.zero_grad()
        self.gen_optim.zero_grad()
        self.unet_optim.zero_grad()

    def step_optimizers(self):
        self.enc_sty_optim.step()
        self.dis_con_optim.step()
        self.dis_dom_optim.step()
        self.gen_optim.step()
        self.unet_optim.step()

    def zero_grad_optim_vae_gen(self):
        self.enc_sty_optim.zero_grad()
        self.gen_optim.zero_grad()
        
        self.unet_optim.zero_grad()
        # self.unet_encoder_optim.zero_grad()
        # self.unet_bottom_block_optim.zero_grad()
        # self.unet_decoder_optim.zero_grad()
        # self.unet_classifier_optim.zero_grad()

    def step_optim_vae_gen(self):
        self.enc_sty_optim.step()
        self.gen_optim.step()

        self.unet_optim.step()
        # self.unet_encoder_optim.step()
        # self.unet_bottom_block_optim.step()
        # self.unet_decoder_optim.step()
        # self.unet_classifier_optim.step()
        
    
    def zero_grad_optim_dis_seg(self):
        self.dis_con_optim.zero_grad()
        self.dis_dom_optim.zero_grad()
        # self.unet_decoder_optim.zero_grad()

    def step_optim_dis_seg(self):
        self.dis_con_optim.step()
        self.dis_dom_optim.step()

        # self.unet_decoder_optim.step()

    def forward(self, x):
        # x_seg = self.unet(x)
        skip_connections, content, style_sample = self.forward_enc(x)
        x_seg = self.forward_dec(skip_connections, content)
        return x_seg

    def forward_enc(self, x, sample_size=0):
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
        return self.enc_sty(x)

    def forward_dec(self, skip_connections, content):
        return self.unet.forward_dec(skip_connections, content)

    def forward_gen(self, content, latent_scale, domain_code):
        return self.gen.forward(content, latent_scale, domain_code)

    def forward_struct_dis(self, x, domain_code):
        x = self.dis_struc(x, domain_code)
        return x

    def forward_dom_dis(self, x, domain_code):
        x = self.dis_dom(x, domain_code)
        return x

    def forward_con_dis(self, skip_connections_x, x):
        x = self.dis_con(skip_connections_x, x)
        return x
    
    def sample_z(self, shape):
        # sample from normal distribution
        return torch.rand(shape)

if __name__ == '__main__':
    import torch
    in_channels = 3
    sample = torch.rand(8, in_channels, 256, 256)

    # enc_con = EncoderContent(in_channels=3)
    # content = enc_con.forward(sample)
    # print('CONTENT SHAPE', content.shape)
    # dis_con = DiscriminatorContent(in_channels=256)
    input_shape=(3,256,256)
    unet = UNet2D_dis(input_shape, nr_labels=2)
    skip_connections, encoding = unet.forward_enc(sample)
    print('UNET', len(skip_connections), encoding.shape)
    # content = torch.rand(8,256,230,230)
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

    gen = Generator(in_channels=256, out_channels=4, domain_code_size=10)
    domain_code = torch.zeros(10)
    domain_code[3] = 1

    latent_scaler = LatentScaler(in_features=250)
    latent_scale = latent_scaler.forward(style_sample)

    gen_out = gen.forward(encoding, latent_scale, domain_code)
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
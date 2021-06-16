from mp.models.model import Model
from mp.models.segmentation.unet_fepegar import UNet2D
from torch.autograd import Variable
import torch.optim as optim

class Distillor(Model):
    r''' Cross Modality Feature Disentangler.
    '''
    def __init__(self,
            input_shape=(1,256,256),
            nr_labels=2,
            unet_dropout = 0,
            unet_monte_carlo_dropout = 0,
            unet_preactivation= False
            ):
        r"""Constructor
        
        Args:
            input_shape (tuple of int): input shape of the images
            nr_labels (int): number of labels for the segmentation
        """
        super(Distillor, self).__init__()

        self.input_shape = input_shape
        self.nr_labels = nr_labels

        self.unet_dropout = unet_dropout
        self.unet_monte_carlo_dropout = unet_monte_carlo_dropout
        self.unet_preactivation = unet_preactivation

        self.unet_new = UNet2D(self.input_shape, self.nr_labels, dropout=self.unet_dropout, monte_carlo_dropout=self.unet_monte_carlo_dropout, preactivation=self.unet_preactivation)
        self.unet_old = None # UNet2D(self.input_shape, self.nr_labels, dropout=self.unet_dropout, monte_carlo_dropout=self.unet_monte_carlo_dropout, preactivation=self.unet_preactivation)
    
    def forward(self, x):
        return self.unet_new(x)
    
    def forward_old(self, x):
        return self.unet_old(x)

    def freeze_unet(self, unet):
        for param in unet.parameters():
            param.requires_grad = False
        return unet

    def freeze_decoder(self, unet):
        for param in unet.decoder.parameters():
            param.requires_grad = False
        for param in unet.classifier.parameters():
            param.requires_grad = False
        return unet
    
    def finish(self):
                
        unet_new_state_dict = self.unet_new.state_dict()
        if next(self.unet_new.parameters()).is_cuda:
            device = next(self.unet_new.parameters()).device

        self.unet_old = UNet2D(self.input_shape, self.nr_labels, dropout=self.unet_dropout, monte_carlo_dropout=self.unet_monte_carlo_dropout, preactivation=self.unet_preactivation)
        self.unet_old.load_state_dict(unet_new_state_dict)
        self.unet_old = self.freeze_unet(self.unet_old)

        self.unet_old.to(device)
        

    def set_optimizers(self, optimizer=optim.SGD, lr=1e-4, weight_decay=1e-4):
        if optimizer == optim.SGD:
            self.unet_optim = optimizer(self.unet_new.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.unet_optim = optimizer(self.unet_new.parameters(), lr=lr)

    def set_scheduler(self, scheduler=optim.lr_scheduler.ExponentialLR, power=0.9):
        self.unet_scheduler = scheduler(self.unet_optim, power, last_epoch=-1)
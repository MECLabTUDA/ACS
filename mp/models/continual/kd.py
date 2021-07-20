from mp.models.model import Model
from mp.models.segmentation.unet_fepegar import UNet2D
import torch.optim as optim

class KD(Model):
    r"""Knowledge Distillation as porposed in Incremental learning techniques for semantic segmentation 
    by Michieli, U., Zanuttigh, P., 2019
    """
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
            unet_dropout (float): dropout probability for the U-Net
            unet_monte_carlo_dropout (float): monte carlo dropout probability for the U-Net
            unet_preactivation (boolean): whether to use U-Net pre-activations
        """
        super(KD, self).__init__()

        self.input_shape = input_shape
        self.nr_labels = nr_labels

        self.unet_dropout = unet_dropout
        self.unet_monte_carlo_dropout = unet_monte_carlo_dropout
        self.unet_preactivation = unet_preactivation

        self.unet_new = UNet2D(self.input_shape, self.nr_labels, dropout=self.unet_dropout, monte_carlo_dropout=self.unet_monte_carlo_dropout, preactivation=self.unet_preactivation)
        self.unet_old = None

    def forward(self, x):
        r"""Forward pass of current U-Net
        
        Args:
            x (torch.Tensor): input batch
        
        Returns:
            (torch.Tensor): segmentated batch
        """
        return self.unet_new(x)
    
    def forward_old(self, x):
        r"""Forward pass of previous U-Net
        
        Args:
            x (torch.Tensor): input batch
        
        Returns:
            (torch.Tensor): segmentated batch
        """
        return self.unet_old(x)

    def freeze_unet(self, unet):
        r"""Freeze U-Net
        
        Args:
            unet (nn.Module): U-Net
        
        Returns:
            (nn.Module): U-Net with frozen weights
        """
        for param in unet.parameters():
            param.requires_grad = False
        return unet

    def freeze_decoder(self, unet):
        r"""Freeze U-Net decoder
        
        Args:
            unet (nn.Module): U-Net
        
        Returns:
            (nn.Module): U-Net with frozen decoder weights
        """
        for param in unet.decoder.parameters():
            param.requires_grad = False
        for param in unet.classifier.parameters():
            param.requires_grad = False
        return unet
    
    def finish(self):
        r"""Finish training, store current U-Net as old U-Net
        """  
        unet_new_state_dict = self.unet_new.state_dict()
        if next(self.unet_new.parameters()).is_cuda:
            device = next(self.unet_new.parameters()).device

        self.unet_old = UNet2D(self.input_shape, self.nr_labels, dropout=self.unet_dropout, monte_carlo_dropout=self.unet_monte_carlo_dropout, preactivation=self.unet_preactivation)
        self.unet_old.load_state_dict(unet_new_state_dict)
        self.unet_old = self.freeze_unet(self.unet_old)

        self.unet_old.to(device)
        

    def set_optimizers(self, optimizer=optim.SGD, lr=1e-4, weight_decay=1e-4):
        r"""Set optimizers for all modules
        
        Args:
            optimizer (torch.nn.optim): optimizer to use
            lr (float): learning rate to use
            weight_decay (float): weight decay
        """
        if optimizer == optim.SGD:
            self.unet_optim = optimizer(self.unet_new.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.unet_optim = optimizer(self.unet_new.parameters(), lr=lr)


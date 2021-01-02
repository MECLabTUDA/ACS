from torch.utils.tensorboard import SummaryWriter

def create_writer(path, init_epoch=0):
    ''' Creates tensorboard SummaryWriter.'''

    return SummaryWriter(path, purge_step=init_epoch)
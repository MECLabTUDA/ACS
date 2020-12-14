from torch.utils.tensorboard import SummaryWriter

def create_writer(config, path):
    print(config)
    writer = SummaryWriter(path, purge_step=0)

    writer.add_scalar('Hyperparameters/learningrate', config['lr'], 0)
    writer.add_scalar('Hyperparameters/batch_size', config['batch_size'], 0)

    return writer
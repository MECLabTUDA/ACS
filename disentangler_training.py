
# ------------------------------------------------------------------------------
# The same code as in example_training.ipynp but as a python module instead of 
# jupyter notebook. See that file for more detailed explainations.
# ------------------------------------------------------------------------------

# 1. Imports

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.experiments.experiment import Experiment
from mp.data.data import Data
from mp.data.datasets.ds_mr_hippocampus_decathlon import DecathlonHippocampus
from mp.data.datasets.ds_mr_hippocampus_dryad import DryadHippocampus
from mp.data.datasets.ds_mr_hippocampus_harp import HarP

import mp.visualization.visualize_imgs as vis
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset, PytorchSeg2DDatasetDomain
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.eval.losses.losses_segmentation import LossClassWeighted, LossDiceBCE
from mp.agents.disentangler_agent import DisentanglerAgent
from mp.eval.result import Result
from mp.utils.load_restore import nifty_dump
from mp.utils.tensorboard import create_writer

from mp.models.disentangler.cmfd import CMFD

# TODO: detect anomaly
torch.autograd.set_detect_anomaly(True)

# 2. Define configuration

config = {'experiment_name':'test_exp', 'device':'cuda:5',
    'nr_runs': 1, 'cross_validation': False, 'val_ratio': 0.0, 'test_ratio': 0.2,
    'input_shape': (1, 256, 256), 'resize': False, 'augmentation': 'none', 
    'class_weights': (0.,1.), 'lr': 1e-4, 'batch_size': 16, 'domain_code_size':10, 'n_samples':5 # # samples per dataloader -> n_samples = None -> all data is used
    }
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(device)
print('Device name: {}'.format(device_name))
input_shape = config['input_shape']  

assert config['batch_size'] % 2 == 0

# 3. Create experiment directories
exp = Experiment(config=config, name='', notes='', reload_exp=True) # config['experiment_name']

# 4. Define data
# data = Data()
# data.add_dataset(DecathlonHippocampus(merge_labels=True))
# nr_labels = data.nr_labels
# label_names = data.label_names
# train_ds = ('DecathlonHippocampus', 'train')
# test_ds = ('DecathlonHippocampus', 'test')

data = Data()

dataset_domain_a = DecathlonHippocampus(merge_labels=True)
dataset_domain_a.name = 'DecathlonHippocampus'
data.add_dataset(dataset_domain_a)

dataset_domain_b = DryadHippocampus(merge_labels=True)
dataset_domain_b.name = 'DryadHippocampus'
data.add_dataset(dataset_domain_b)

# dataset_domain_c = HarP(merge_labels=True)
# dataset_domain_c.name = 'HarP'
# data.add_dataset(dataset_domain_c)

nr_labels = data.nr_labels
label_names = data.label_names
train_ds_a = ('DecathlonHippocampus', 'train')
train_ds_b = ('DryadHippocampus', 'train')
test_ds_c = ('HarP', 'test')
# test_ds_b = ('DecathlonHippocampus1', 'test')

# 5. Create data splits for each repetition
exp.set_data_splits(data)

# Now repeat for each repetition
for run_ix in range(config['nr_runs']):
    exp_run = exp.get_run(run_ix=0)

    # 6. Bring data to Pytorch format and add domain_code
    datasets = dict()
    for idx, item in enumerate(data.datasets.items()):
        ds_name, ds = item
        for split, data_ixs in exp.splits[ds_name][exp_run.run_ix].items():
            data_ixs = data_ixs[:config['n_samples']]
            if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                aug = config['augmentation'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = PytorchSeg2DDatasetDomain(ds, 
                    ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                    resize=config['resize'], domain_code=idx, domain_code_size=config['domain_code_size'])

    # 6.1 combine datasets
    # multi_domain_dataset = torch.utils.data.ConcatDataset((datasets[(train_ds_0)], datasets[(train_ds_1)]))

    # TODO: maybe use ConcatDataset and draw double/split batch and cross domain samples
    # i.e. batch of 8 -> two batches of 4
    # transfer batch 0 to batch 1 and reconstruct
    # 7. Build train dataloader, and visualize
    # dl = DataLoader(multi_domain_dataset, 
    #     batch_size=config['batch_size'], shuffle=True)

    # 7. Build two train dataloaders -> two domains
    # dl_a = DataLoader(datasets[(train_ds_a)], batch_size=config['batch_size'], shuffle=True)
    # dl_b = DataLoader(datasets[(train_ds_b)], batch_size=config['batch_size'], shuffle=True)
   
    multi_domain_dataset = torch.utils.data.ConcatDataset((datasets[(train_ds_a)], datasets[(train_ds_b)]))
    dl = DataLoader(multi_domain_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    # 8. Initialize model
    model = CMFD(input_shape, latent_channels=256, domain_code_size=config['domain_code_size'], latent_scaler_sample_size=250)
    model.to_device(device)
    model.parallel(device_ids=[5,6])

    # 9. Define loss and optimizer
    loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=device)
    loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'], 
        device=device)
    # TODO: fix CELoss
    loss_c = torch.nn.CrossEntropyLoss()
    model.set_optimizers(optim.Adam, lr=config['lr'])
    # 9.1 Set optimizer, pass None because optimizers are saved in model
    optimizer = None

    # 10.1 Create tensorboard SummaryWriter
    writer = create_writer(config, exp.path)
    # 10. Train model
    results = Result(name='training_trajectory')   
    agent = DisentanglerAgent(model=model, label_names=label_names, device=device, summary_writer=writer)
    agent.train(results, optimizer, loss_g, dl,
        init_epoch=0, nr_epochs=10, run_loss_print_interval=1,
        eval_datasets=datasets, eval_interval=1, 
        save_path=exp_run.paths['states'], save_interval=1,
        display_interval=1)

    # 11. Save and print results for this experiment run
    exp_run.finish(results=results, plot_metrics=['Mean_ScoreDice', 'Mean_ScoreDice[prostate]'])
    test_ds_key = '_'.join(test_ds_c)
    metric = 'Mean_ScoreDice[prostate]'
    last_dice = results.get_epoch_metric(
        results.get_max_epoch(metric, data=test_ds_key), metric, data=test_ds_key)
    print('Last Dice score for prostate class: {}'.format(last_dice))


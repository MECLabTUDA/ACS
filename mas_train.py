
# ------------------------------------------------------------------------------
# The same code as in example_training.ipynp but as a python module instead of 
# jupyter notebook. See that file for more detailed explainations.
# ------------------------------------------------------------------------------

# Imports
import os
import sys
from args import parse_args_as_dict
from mp.utils.helper_functions import seed_all

import torch
torch.set_num_threads(6)
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler

from mp.experiments.experiment import Experiment
from mp.data.data import Data
from mp.data.datasets.ds_mr_hippocampus_decathlon import DecathlonHippocampus
from mp.data.datasets.ds_mr_hippocampus_dryad import DryadHippocampus
from mp.data.datasets.ds_mr_hippocampus_harp import HarP

import mp.visualization.visualize_imgs as vis
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset, PytorchSeg2DDatasetDomain
from mp.models.segmentation.unet_milesial import UNet
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.eval.losses.losses_segmentation import LossClassWeighted, LossDiceBCE
from mp.agents.mas_agent import MASAgent
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.result import Result
from mp.utils.load_restore import nifty_dump
from mp.utils.tensorboard import create_writer

from mp.models.mas.mas import MAS

# torch.autograd.set_detect_anomaly(True)

# Get configuration from arguments
config = parse_args_as_dict(sys.argv[1:])
seed_all(42)

config['class_weights'] = (0., 1.)

# Create experiment directories
exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=(config['resume_epoch'] is not None))

# Datasets
data = Data()

dataset_domain_a = DecathlonHippocampus(merge_labels=True)
dataset_domain_a.name = 'DecathlonHippocampus'
data.add_dataset(dataset_domain_a)

dataset_domain_b = DryadHippocampus(merge_labels=True)
dataset_domain_b.name = 'DryadHippocampus'
data.add_dataset(dataset_domain_b)

dataset_domain_c = HarP(merge_labels=True)
dataset_domain_c.name = 'HarP'
data.add_dataset(dataset_domain_c)

nr_labels = data.nr_labels
label_names = data.label_names

if config['combination'] == 0:
    ds_a = ('DecathlonHippocampus', 'train')
    ds_b = ('DryadHippocampus', 'train')
    ds_c = ('HarP', 'train')
elif config['combination'] == 1:
    ds_a = ('DecathlonHippocampus', 'train')
    ds_c = ('DryadHippocampus', 'train')
    ds_b = ('HarP', 'train')
elif config['combination'] == 2:
    ds_c = ('DecathlonHippocampus', 'train')
    ds_b = ('DryadHippocampus', 'train')
    ds_a = ('HarP', 'train')

# ds_test = [('DecathlonHippocampus', 'test'), ('DryadHippocampus', 'test'), ('HarP', 'test')]
# ds_val = [('DecathlonHippocampus', 'val'), ('DryadHippocampus', 'val'), ('HarP', 'val')]

# Create data splits for each repetition
exp.set_data_splits(data)

# Now repeat for each repetition
for run_ix in range(config['nr_runs']):
    exp_run = exp.get_run(run_ix=0, reload_exp_run=(config['resume_epoch'] is not None))

    # Bring data to Pytorch format and add domain_code
    datasets = dict()
    for idx, item in enumerate(data.datasets.items()):
        ds_name, ds = item
        for split, data_ixs in exp.splits[ds_name][exp_run.run_ix].items():
            data_ixs = data_ixs[:config['n_samples']]
            if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                aug = config['augmentation'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = PytorchSeg2DDataset(ds, 
                    ix_lst=data_ixs, size=config['input_shape']  , aug_key=aug, 
                    resize=(not config['no_resize']))

    dataset = torch.utils.data.ConcatDataset((datasets[(ds_a)], datasets[(ds_b)]))
    train_dataloader_0 = DataLoader(dataset, batch_size=config['batch_size'], drop_last=False, pin_memory=True, num_workers=len(config['device_ids'])*config['n_workers'])
    train_dataloader_1 = DataLoader(datasets[(ds_c)], batch_size=config['batch_size'], shuffle=True, drop_last=False, pin_memory=True, num_workers=len(config['device_ids'])*config['n_workers'])

    if config['eval']:
        drop = []
        for key in datasets.keys():
            if 'train' in key or 'val' in key:
                drop += [key]
        for d in drop:
            datasets.pop(d)
    elif config['lambda_eval']:
        drop = []
        for key in datasets.keys():
            if 'train' in key or 'test' in key:
                drop += [key]
        for d in drop:
            datasets.pop(d)

    model = MAS(input_shape=config['input_shape'], nr_labels=nr_labels,
                    unet_dropout=config['unet_dropout'], unet_monte_carlo_dropout=config['unet_monte_carlo_dropout'], unet_preactivation=config['unet_preactivation'])
    
    model.to(config['device'])
  
    # Define loss and optimizer
    loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=config['device'])
    loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'], device=config['device'])

    # Set optimizers
    model.set_optimizers(optim.Adam, lr=config['lr'])

    # Train model
    results = Result(name='training_trajectory')

    agent = MASAgent(model=model, label_names=label_names, device=config['device'])
    agent.summary_writer = create_writer(os.path.join(exp_run.paths['states'], '..'), 0)

    init_epoch = 0
    nr_epochs = config['epochs'] // 2

    config['continual'] = False

    # Resume training
    if config['resume_epoch'] is not None:
        agent.restore_state(exp_run.paths['states'], config['resume_epoch'])
        init_epoch = agent.agent_state_dict['epoch'] + 1

    # Train on A and B
    if init_epoch < config['epochs'] / 2:
    # if init_epoch < config['epochs'] * 2/3:
        agent.train(results, loss_f, train_dataloader_0, train_dataloader_1, config,
            init_epoch=init_epoch, nr_epochs=nr_epochs, run_loss_print_interval=1,
            eval_datasets=datasets, eval_interval=config['eval_interval'],
            save_path=exp_run.paths['states'], save_interval=config['save_interval'],
            display_interval=config['display_interval'],
            resume_epoch=config['resume_epoch'], device_ids=config['device_ids'])

        print('Finished training on A and B, starting training on C')

    init_epoch = config['epochs'] // 2
    nr_epochs = config['epochs']

    # Resume training
    if config['resume_epoch'] is not None:
        agent.restore_state(exp_run.paths['states'], config['resume_epoch'])
        init_epoch = agent.agent_state_dict['epoch'] + 1
    
    # Train on C
    if init_epoch >= config['epochs'] / 2:
        
        if config['unet_only']:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.unet.decoder.decoding_blocks[-2].parameters():
                param.requires_grad = True
            for param in model.unet.decoder.decoding_blocks[-1].parameters():
                param.requires_grad = True
            for param in model.unet.classifier.parameters():
                param.requires_grad = True

        # model.unet_optim = optim.Adam(model.unet.parameters(), lr=config['lr'] / 3)
        # model.unet_scheduler = torch.optim.lr_scheduler.StepLR(model.unet_optim, (nr_epochs-init_epoch), gamma=0.1, last_epoch=-1)
        
        # Set optimizers
        model.set_optimizers(optim.Adam, lr=config['lr_2'])
        config['continual'] = True
        model.unet_scheduler = torch.optim.lr_scheduler.StepLR(model.unet_optim, (nr_epochs-init_epoch), gamma=0.1, last_epoch=-1)
        
        print('Freezing everything but last 2 layers of segmentor')
        agent.train(results, loss_f, train_dataloader_1, train_dataloader_0, config,
            init_epoch=init_epoch, nr_epochs=nr_epochs, run_loss_print_interval=1,
            eval_datasets=datasets, eval_interval=config['eval_interval'],
            save_path=exp_run.paths['states'], save_interval=config['save_interval'],
            display_interval=config['display_interval'],
            resume_epoch=config['resume_epoch'], device_ids=[0])

        print('Finished training on C')

    # Save and print results for this experiment run
    exp_run.finish(results=results, plot_metrics=['Mean_LossBCEWithLogits', 'Mean_LossDice[smooth=1.0]', 'Mean_LossCombined[1.0xLossDice[smooth=1.0]+1.0xLossBCEWithLogits]'])


# ------------------------------------------------------------------------------
# The same code as in example_training.ipynp but as a python module instead of 
# jupyter notebook. See that file for more detailed explainations.
# ------------------------------------------------------------------------------

# Imports
import sys
from args import parse_args_as_dict
from mp.utils.helper_functions import seed_all

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
# torch.autograd.set_detect_anomaly(True)

# Get configuration from arguments
config = parse_args_as_dict(sys.argv[1:])
seed_all(42)

config['class_weights'] = (0., 1.)

# Create experiment directories
exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)

# Datasets
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
# test_ds_c = ('HarP', 'test')

# Create data splits for each repetition
exp.set_data_splits(data)

# Now repeat for each repetition
for run_ix in range(config['nr_runs']):
    exp_run = exp.get_run(run_ix=0)

    # Bring data to Pytorch format and add domain_code
    datasets = dict()
    for idx, item in enumerate(data.datasets.items()):
        ds_name, ds = item
        for split, data_ixs in exp.splits[ds_name][exp_run.run_ix].items():
            data_ixs = data_ixs[:config['n_samples']]
            if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                aug = config['augmentation'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = PytorchSeg2DDatasetDomain(ds, 
                    ix_lst=data_ixs, size=config['input_shape']  , aug_key=aug, 
                    resize=config['resize'], domain_code=idx, domain_code_size=config['domain_code_size'])

    # Combine datasets
    multi_domain_dataset = torch.utils.data.ConcatDataset((datasets[(train_ds_a)], datasets[(train_ds_b)]))
    dl = DataLoader(multi_domain_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    # Initialize model
    model = CMFD(config['input_shape'], domain_code_size=config['domain_code_size'], latent_scaler_sample_size=250)
    model.to(config['device'])
    # if len(config['device_ids']) > 1:
    #     print('Using data parallel on devices', config['device_ids'])
        # model.parallel(device_ids=config['device_ids'])
        # model = nn.DataParallel(model, device_ids=config['device_ids'])

    # Define loss and optimizer
    loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=config['device'])
    loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'], device=config['device'])
    # TODO: fix CELoss
    loss_c = torch.nn.CrossEntropyLoss()

    # Set optimizers
    model.set_optimizers(optim.Adam, lr=config['lr'])

    # Create tensorboard SummaryWriter
    writer = create_writer(config, exp.path)
    
    # Train model
    results = Result(name='training_trajectory')   
    agent = DisentanglerAgent(model=model, label_names=label_names, device=config['device'], summary_writer=writer)
    agent.train(results, loss_g, dl,
        init_epoch=0, nr_epochs=config['epochs'], run_loss_print_interval=1,
        eval_datasets=datasets, eval_interval=5, 
        save_path=exp_run.paths['states'], save_interval=5,
        display_interval=1)

    # Save and print results for this experiment run
    exp_run.finish(results=results, plot_metrics=['Mean_LossBCEWithLogits', 'Mean_LossDice[smooth=1.0]', 'Mean_LossCombined[1.0xLossDice[smooth=1.0]+1.0xLossBCEWithLogits]'])
    # test_ds_key = '_'.join(test_ds_c)
    # metric = 'Mean_LossDice[smooth=1.0]'
    
    # print(results.results.keys())
    
    # last_dice = results.get_epoch_metric(
    #     results.get_max_epoch(metric, data=test_ds_key), metric, data=test_ds_key)
    # print('Last Dice score for hippocampus class: {}'.format(last_dice))
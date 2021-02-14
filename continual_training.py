
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
from mp.agents.disentangler_agent import DisentanglerAgent
from mp.agents.whatever_agent import WhateverAgent
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.result import Result
from mp.utils.load_restore import nifty_dump
from mp.utils.tensorboard import create_writer

from mp.models.disentangler.cmfd import CMFD

# torch.autograd.set_detect_anomaly(True)

# Get configuration from arguments
config = parse_args_as_dict(sys.argv[1:])
seed_all(42)

config['class_weights'] = (0., 1.) # -> inverse of label ratios -> try: (0.3, 0.7)
# config['class_weights'] = (0.8, 0.2)

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
                # TODO
                datasets[(ds_name, split)] = PytorchSeg2DDatasetDomain(ds, 
                    ix_lst=data_ixs, size=config['input_shape']  , aug_key=aug, 
                    resize=(not config['no_resize']), domain_code=idx, domain_code_size=config['domain_code_size'])

                # datasets[(ds_name, split)] = PytorchSeg2DDataset(ds, 
                #     ix_lst=data_ixs, size=config['input_shape'], aug_key=aug, 
                #     resize=(not config['no_resize']))

    # Combine datasets
    if config['single_ds']:
        dataset = datasets[(ds_a)]
        train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, pin_memory=True, num_workers=len(config['device_ids'])*config['n_workers'])

    else:
        len_a = len(datasets[(ds_a)])
        len_b = len(datasets[(ds_b)])
        ratio = len_a / (len_a+len_b)
        print('length dataset A:', len_a, 'B:', len_b, 'weights A:', 1-ratio, 'B:', ratio)

        samples_weight_a = torch.full((len_a,), 1-ratio)
        samples_weight_b = torch.full((len_b,), ratio)

        # use WeightedRandomSampler to counter class balance
        # https://github.com/ptrblck/pytorch_misc/blob/master/weighted_sampling.py 
        samples_weight_both = torch.cat((samples_weight_a, samples_weight_b))
        sampler = WeightedRandomSampler(samples_weight_both, len(samples_weight_both))
        
        dataset = torch.utils.data.ConcatDataset((datasets[(ds_a)], datasets[(ds_b)]))
        
        if config['sampler']:
            train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], drop_last=True, pin_memory=True, num_workers=len(config['device_ids'])*config['n_workers'], sampler=sampler)
        else:
            print('Not using a sampler')
            train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], drop_last=True, pin_memory=True, num_workers=len(config['device_ids'])*config['n_workers'])
   
    test_dataloader = DataLoader(datasets[(ds_c)], batch_size=config['batch_size'], shuffle=True, drop_last=True, pin_memory=True, num_workers=len(config['device_ids'])*config['n_workers'])

    model = CMFD(input_shape=config['input_shape'], nr_labels=nr_labels, domain_code_size=config['domain_code_size'], latent_scaler_sample_size=250,
                    unet_dropout=config['unet_dropout'], unet_monte_carlo_dropout=config['unet_monte_carlo_dropout'], unet_preactivation=config['unet_preactivation'])
    
    model.to(config['device'])
  
    # Define loss and optimizer
    loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=config['device'])
    loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'], device=config['device'])

    # Set optimizers
    model.set_optimizers(optim.Adam, lr=config['lr'])
    
    # Train model
    results = Result(name='training_trajectory')

    agent = WhateverAgent(model=model, label_names=label_names, device=config['device'])

    init_epoch = 0
    nr_epochs = config['epochs'] // 2
    # nr_epochs = int(config['epochs'] * 2/3)

    # Resume training
    if config['resume_epoch'] is not None:
        agent.restore_state(exp_run.paths['states'], config['resume_epoch'])
        init_epoch = agent.agent_state_dict['epoch'] + 1

    # Train on A and B
    if init_epoch < config['epochs'] / 2:
    # if init_epoch < config['epochs'] * 2/3:
        config['continual'] = False
        agent.train(results, loss_f, train_dataloader, test_dataloader, config,
            init_epoch=init_epoch, nr_epochs=nr_epochs, run_loss_print_interval=1,
            eval_datasets=datasets, eval_interval=config['eval_interval'],
            save_path=exp_run.paths['states'], save_interval=config['save_interval'],
            display_interval=config['display_interval'],
            resume_epoch=config['resume_epoch'], device_ids=config['device_ids'])

        print('Finished training on A and B, starting training on C')

    init_epoch = config['epochs'] // 2
    # init_epoch = int(config['epochs'] * 2/3)
    nr_epochs = config['epochs']

    # Resume training
    if config['resume_epoch'] is not None:
        agent.restore_state(exp_run.paths['states'], config['resume_epoch'])
        init_epoch = agent.agent_state_dict['epoch'] + 1
    
    # Train on C
    if init_epoch >= config['epochs'] / 2:
    # if init_epoch >= config['epochs'] * 1/3:
        # config['continual'] = True

        # TODO
        # config['lr'] /= 2
        # config['continual'] = False

        # ################
        # len_c = len(datasets[(ds_c)])
        # len_b = len(datasets[(ds_b)])
        # ratio = len_c / (len_a+len_b)
        # print('length dataset C:', len_c, 'B:', len_b, 'weights C:', 1-ratio, 'B:', ratio)

        # samples_weight_c = torch.full((len_c,), 1-ratio)
        # samples_weight_b = torch.full((len_b,), ratio)

        # # use WeightedRandomSampler to counter class balance
        # # https://github.com/ptrblck/pytorch_misc/blob/master/weighted_sampling.py 
        # samples_weight_both = torch.cat((samples_weight_c, samples_weight_b))
        # sampler = WeightedRandomSampler(samples_weight_both, len(samples_weight_both))
        
        # dataset = torch.utils.data.ConcatDataset((datasets[(ds_c)], datasets[(ds_b)]))
        # train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], drop_last=True, pin_memory=True, num_workers=len(config['device_ids'])*config['n_workers'], sampler=sampler)
        #######################

        config['continual'] = True
        for param in model.parameters():
            param.requires_grad = False
       
        if len(config['device_ids']) > 1:
            for param in model.unet.decoder.module.decoding_blocks[-2].parameters():
                print('BEFORE model.unet.decoder.module.decoding_blocks[-2]', param.requires_grad)
                param.requires_grad = True
                print('AFTER model.unet.decoder.module.decoding_blocks[-2]', param.requires_grad)
            for param in model.unet.decoder.module.decoding_blocks[-1].parameters():
                print('BEFORE model.unet.decoder.module.decoding_blocks[-1]', param.requires_grad)
                param.requires_grad = True
                print('AFTER model.unet.decoder.module.decoding_blocks[-1]', param.requires_grad)
            for param in model.unet.classifier.parameters():
                print('BEFORE model.unet.decoder.module.classifier', param.requires_grad)
                param.requires_grad = True
                print('AFTER model.unet.decoder.module.classifier', param.requires_grad)
        else:
            for param in model.unet.decoder.decoding_blocks[-2].parameters():
                param.requires_grad = True
            for param in model.unet.decoder.decoding_blocks[-1].parameters():
                param.requires_grad = True
            for param in model.unet.classifier.parameters():
                param.requires_grad = True
        
        # model.unet_optim = optim.Adam(filter(lambda p: p.requires_grad, model.unet.parameters()),lr=config['lr'])
        
        model.unet_optim = optim.Adam(model.unet.parameters(),lr=config['lr'] / 3)
        model.unet_scheduler = torch.optim.lr_scheduler.StepLR(model.unet_optim, (nr_epochs-init_epoch), gamma=0.1, last_epoch=-1)

        print('Freezing everything but last 2 layers of segmentor')
        agent.train(results, loss_f, test_dataloader, train_dataloader, config,
            init_epoch=init_epoch, nr_epochs=nr_epochs, run_loss_print_interval=1,
            eval_datasets=datasets, eval_interval=config['eval_interval'],
            save_path=exp_run.paths['states'], save_interval=config['save_interval'],
            display_interval=config['display_interval'],
            resume_epoch=config['resume_epoch'], device_ids=[0]) # device_ids=config['device_ids'])

        print('Finished training on C')

    # Save and print results for this experiment run
    exp_run.finish(results=results, plot_metrics=['Mean_LossBCEWithLogits', 'Mean_LossDice[smooth=1.0]', 'Mean_LossCombined[1.0xLossDice[smooth=1.0]+1.0xLossBCEWithLogits]'])
    # test_ds_key = '_'.join(ds_c)
    # metric = 'Mean_LossDice[smooth=1.0]'
    
    # last_dice = results.get_epoch_metric(
    #     results.get_max_epoch(metric, data=test_ds_key), metric, data=test_ds_key)
    # print('Last Dice score for hippocampus class: {}'.format(last_dice))

    # import mp.visualization.visualize_imgs as vis
    # Visualize result for the first subject in the test dataset
    # subject_ix = 0
    # subject = datasets[test_ds].instances[subject_ix].get_subject()
    # pred = datasets[test_ds].predictor.get_subject_prediction(agent, subject_ix)
    # vis.plot_3d_subject_pred(subject, pred)
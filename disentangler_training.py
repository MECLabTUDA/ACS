
# ------------------------------------------------------------------------------
# The same code as in example_training.ipynp but as a python module instead of 
# jupyter notebook. See that file for more detailed explainations.
# ------------------------------------------------------------------------------

# 1. Imports

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.experiments.experiment import Experiment
from mp.data.data import Data
from mp.data.datasets.ds_mr_hippocampus_decathlon import DecathlonHippocampus
import mp.visualization.visualize_imgs as vis
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset, PytorchSeg2DDatasetDomain
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.eval.losses.losses_segmentation import LossClassWeighted, LossDiceBCE
from mp.agents.disentangler_agent import DisentanglerAgent
from mp.eval.result import Result
from mp.utils.load_restore import nifty_dump

from mp.models.disentangler.cmfd import CMFD

torch.autograd.set_detect_anomaly(True)

# 2. Define configuration

config = {'experiment_name':'test_exp', 'device':'cuda:6',
    'nr_runs': 1, 'cross_validation': False, 'val_ratio': 0.0, 'test_ratio': 0.3,
    'input_shape': (1, 256, 256), 'resize': False, 'augmentation': 'none', 
    'class_weights': (0.,1.), 'lr': 2e-4, 'batch_size': 1, 'domain_code_size':10
    }
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(device)
print('Device name: {}'.format(device_name))
input_shape = config['input_shape']  

# 3. Create experiment directories
exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)

# 4. Define data
# data = Data()
# data.add_dataset(DecathlonHippocampus(merge_labels=True))
# nr_labels = data.nr_labels
# label_names = data.label_names
# train_ds = ('DecathlonHippocampus', 'train')
# test_ds = ('DecathlonHippocampus', 'test')

data = Data()
dataset_domain_0 = DecathlonHippocampus(merge_labels=True)
dataset_domain_1 = DecathlonHippocampus(merge_labels=True)
dataset_domain_0.name = 'DecathlonHippocampus0'
dataset_domain_1.name = 'DecathlonHippocampus1'
data.add_dataset(dataset_domain_0)
data.add_dataset(dataset_domain_1)

# print(data.datasets)

nr_labels = data.nr_labels
label_names = data.label_names
train_ds_0 = ('DecathlonHippocampus0', 'train')
train_ds_1 = ('DecathlonHippocampus1', 'train')
test_ds_0 = ('DecathlonHippocampus0', 'test')
test_ds_1 = ('DecathlonHippocampus1', 'test')

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
    dl_0 = DataLoader(datasets[(train_ds_0)], batch_size=config['batch_size'], shuffle=True)
    dl_1 = DataLoader(datasets[(train_ds_1)], batch_size=config['batch_size'], shuffle=True)

    # 8. Initialize model
    model = CMFD(input_shape, latent_channels=256, domain_code_size=config['domain_code_size'], latent_scaler_sample_size=250)
    model.to_device(device)

    # 9. Define loss and optimizer
    loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=device)
    loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'], 
        device=device)
    # TODO: fix CELoss
    loss_c = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 10. Train model
    results = Result(name='training_trajectory')   
    agent = DisentanglerAgent(model=model, label_names=label_names, device=device)
    agent.train(results, optimizer, loss_g, dl_0, dl_1,
        init_epoch=0, nr_epochs=10, run_loss_print_interval=5,
        eval_datasets=datasets, eval_interval=5, 
        save_path=exp_run.paths['states'], save_interval=5)

    # 11. Save and print results for this experiment run
    exp_run.finish(results=results, plot_metrics=['Mean_ScoreDice', 'Mean_ScoreDice[prostate]'])
    test_ds_key = '_'.join(test_ds)
    metric = 'Mean_ScoreDice[prostate]'
    last_dice = results.get_epoch_metric(
        results.get_max_epoch(metric, data=test_ds_key), metric, data=test_ds_key)
    print('Last Dice score for prostate class: {}'.format(last_dice))


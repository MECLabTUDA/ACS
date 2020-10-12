# ------------------------------------------------------------------------------
# An example of training and evaluating a model for the prostate dataset in the
# Medical Segmentation Decathlon. To execute, download the data for task 5 from
# http://medicaldecathlon.com/ and specify the path in mp.paths.py. 
# ------------------------------------------------------------------------------
#%% 1. Imports

from IPython import get_ipython
get_ipython().magic('load_ext autoreload') 
get_ipython().magic('autoreload 2')

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.experiments.experiment import Experiment
from mp.data.data import Data
from mp.data.datasets.ds_mr_prostate_decathlon import DecathlonProstateT2
import mp.visualization.visualize_imgs as vis
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.eval.losses.losses_segmentation import LossClassWeighted, LossDiceBCE
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.result import Result
from mp.utils.load_restore import nifty_dump

#%% 2. Define configuration
# The configuration dictionary boundles training parameters and anything 
# that can be manually defined. The config file is stored within the directory 
# created for a given experiment. By specifying arguments in config files, 
# higher-level modules don't need to be adapted for hyperparameter searches
# and experiments are more easily reproducible.
config = {'experiment_name':'test_exp', 'device':'cuda:0',
    'nr_runs': 1, 'cross_validation': False, 'val_ratio': 0.0, 'test_ratio': 0.3,
    'input_shape': (1, 256, 256), 'resize': False, 'augmentation': 'none', 
    'class_weights': (0.,1.), 'lr': 0.0001, 'batch_size': 8
    }
device = config['device']
device_name = torch.cuda.get_device_name(device)
print('Device name: {}'.format(device_name))
input_shape = config['input_shape']  

#%% 3. Create expeirment directories
# Initializing an experiment creates a directory exp/<exp name> in the
# storage directory defined in mp.paths.py. All files created during that 
# experiment are stored there.
exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)

#%% 4. Define data
# A data object can be initialized with multiple datasets. The idea behind this
# is to more easily test with o.o.d data, or simulate continual learning.
# In this example, we only add one dataset.
data = Data()
data.add_dataset(DecathlonProstateT2(merge_labels=True))
nr_labels = data.nr_labels
label_names = data.label_names
train_ds = ('DecathlonProstateT2', 'train')

#%% 5. Create data splits for each repetition
# For each dataset, the instance indexes are divided into train, validation
# and test sets. The values specified in the config are used to determine
# the number of runs and how indexes are divided. For more details, look at the
# method definition for 'set_data_splits'. The index splitting is also stored 
# within the experiment directory. For each repetition, a subdirectory is 
# created named after the repetition index.
exp.set_data_splits(data)

# Get the experiment run respective to the first data split. This would usually 
# be repeated for all 'nr_runs' runs
exp_run = exp.get_run(run_ix=0)

#%% 6. Bring data to Pytorch format
# Transform data to PyTorch format
datasets = dict()
for ds_name, ds in data.datasets.items():
    for split, data_ixs in exp.splits[ds_name][exp_run.run_ix].items():
        if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
            aug = config['augmentation'] if not('test' in split) else 'none'
            datasets[(ds_name, split)] = PytorchSeg2DDataset(ds, 
                ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                resize=config['resize'])
# Visualize the first instance from the training data
subject_ix = 0
subject = datasets[train_ds].instances[subject_ix].get_subject()
vis.plot_3d_subject_gt(subject)

#%% 7. Build train dataloader, and visualize
dl = DataLoader(datasets[(train_ds)], 
    batch_size=config['batch_size'], shuffle=True)
#vis_file = join_path([exp_run.paths['results'], config['train_dataset']+'_DL.png'])
vis.visualize_dataloader_with_masks(dl, img_size=(128, 128))

#%% 8. Initialize model
model = UNet2D(input_shape, nr_labels)
model.to(device)

#%% 8. Define loss and optimizer
# Define loss and optimizer. In this case, the loss is a combination of Dice
# and binary cross-entropy, weighted so that only the dice on the 'prostate'
# class is considered and not that for 'background'
loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=device)
loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'], 
    device=device)
optimizer = optim.Adam(model.parameters(), lr=config['lr'])

#%% 9. Train model
results = Result(name='training_trajectory')   
agent = SegmentationAgent(model=model, label_names=label_names, device=device)
agent.train(results, optimizer, loss_f, train_dataloader=dl,
    init_epoch=0, nr_epochs=3, run_loss_print_interval=1,
    eval_datasets=datasets, eval_interval=1, 
    save_path=exp_run.paths['states'], save_interval=1)

#%% 10. Save and print results for this experiment run
exp_run.finish(results=results)
test_ds = ('DecathlonProstateT2', 'test')
metric = 'ScoreDice'
last_dice = results.get_epoch_metric(results.get_max_epoch(metric, data=test_ds), metric, data=test_ds)
print('Last Dice score for prostate class: {}'.format(last_dice))

#%% 11. Visualize result for the first subject in the test dataset
subject_ix = 0
subject = datasets[test_ds].instances[subject_ix].get_subject()
pred = datasets[test_ds].predictor.get_subject_prediction(agent, subject_ix)
vis.plot_3d_subject_pred(subject, pred)
# Save the predicted mask to compare the 3D rendering to the ground truth in 
# a program such as SimpleITK
##nifty_dump(pred, name='pred_epoch_300', path='storage/agent_states_prostate_2D_5')

# %%

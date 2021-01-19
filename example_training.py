
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
from mp.data.datasets.ds_mr_hippocampus_harp import HarP
import mp.visualization.visualize_imgs as vis
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.models.segmentation.unet_milesial import UNet
from mp.eval.losses.losses_segmentation import LossClassWeighted, LossDiceBCE
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.result import Result
from mp.utils.load_restore import nifty_dump

# 2. Define configuration

config = {'experiment_name':'', 'device':'cuda:4',
    'nr_runs': 1, 'cross_validation': False, 'val_ratio': 0.0, 'test_ratio': 0.0,
    'input_shape': (1, 256, 256), 'resize': True, 'augmentation': 'none', 
    'class_weights': (0.,1.), 'lr': 0.001, 'batch_size': 8, 'n_samples': 1
    }
device = config['device']
device_name = torch.cuda.get_device_name(device)
print('Device name: {}'.format(device_name))
input_shape = config['input_shape']  

# 3. Create experiment directories
exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=False)

# 4. Define data
data = Data()
# data.add_dataset(HarP(merge_labels=True))
# nr_labels = data.nr_labels
# label_names = data.label_names
# train_ds = ('HarP[Part:All]', 'train')
# test_ds = ('HarP[Part:All]', 'test')

data.add_dataset(DecathlonHippocampus(merge_labels=True))
nr_labels = data.nr_labels
label_names = data.label_names
train_ds = ('DecathlonHippocampus', 'train')
test_ds = ('DecathlonHippocampus', 'test')

# 5. Create data splits for each repetition
exp.set_data_splits(data)

# Now repeat for each repetition
for run_ix in range(config['nr_runs']):
    exp_run = exp.get_run(run_ix=0)

    # 6. Bring data to Pytorch format
    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in exp.splits[ds_name][exp_run.run_ix].items():
            data_ixs =data_ixs[:config['n_samples']]
            if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                aug = config['augmentation'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = PytorchSeg2DDataset(ds, 
                    ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                    resize=config['resize'])


    # 7. Build train dataloader, and visualize
    dl = DataLoader(datasets[(train_ds)], 
        batch_size=config['batch_size'], shuffle=True, num_workers=2)

    # 8. Initialize model
    # model = UNet(input_shape, nr_labels)
    model = UNet2D(input_shape, nr_labels)
    model.to(device)

    # 9. Define loss and optimizer
    loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=device)
    loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'], 
        device=device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # for e in range(25):
    #     l = 0
    #     for data in dl:
    #         x, y = data
    #         pred = model(x.to(config['device']))
    #         loss = loss_f(pred, y.to(config['device']))
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         l += loss
    #     print(e, l/len(dl))

    # 10. Train model
    results = Result(name='training_trajectory')   
    agent = SegmentationAgent(model=model, label_names=label_names, device=device)
    agent.train(results, optimizer, loss_f, train_dataloader=dl,
        init_epoch=0, nr_epochs=25, run_loss_print_interval=1,
        eval_datasets=datasets, eval_interval=5,
        save_path=exp_run.paths['states'], save_interval=5)

    # 11. Save and print results for this experiment run
    exp_run.finish(results=results, plot_metrics=['Mean_ScoreDice', 'Mean_ScoreDice[hippocampus]', 'Mean_ScoreIoU', 'Mean_ScoreIoU[hippocampus]'])
    print(results.results.keys())
    test_ds_key = '_'.join(test_ds)
    metric = 'Mean_ScoreDice[hippocampus]'
    last_dice = results.get_epoch_metric(
        results.get_max_epoch(metric, data=test_ds_key), metric, data=test_ds_key)
    print('Last Dice score for hippocampus class: {}'.format(last_dice))

    
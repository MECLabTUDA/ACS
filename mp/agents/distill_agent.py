import time
import os
import torch
import torch.nn as nn
from mp.agents.agent import Agent
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.evaluate import ds_losses_metrics
from mp.eval.accumulator import Accumulator
from tqdm import tqdm

from mp.utils.pytorch.pytorch_load_restore import load_model_state, save_optimizer_state, load_optimizer_state, save_model_state_dataparallel

from torch.nn import functional as F
from mp.utils.tensorboard import create_writer

from mp.visualization.visualize_imgs import plot_3d_segmentation
from PIL import Image
import torchvision.transforms.functional as TF
from mp.utils.load_restore import pkl_dump, pkl_load

from mp.eval.inference.predict import softmax

class DistillAgent(SegmentationAgent):
    r"""An Agent for segmentation models."""
    def __init__(self, *args, **kwargs):
        if 'metrics' not in kwargs:
            kwargs['metrics'] = ['ScoreDice', 'ScoreIoU']
        super().__init__(*args, **kwargs)

    def train(self, results, loss_f, train_dataloader, test_dataloader, config,
        init_epoch=0, nr_epochs=100, run_loss_print_interval=1,
        eval_datasets=dict(), eval_interval=2, 
        save_path=None, save_interval=10, 
        display_interval=1, 
        resume_epoch=None,
        device_ids=[0]):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states.
        """

        self.agent_state_dict['epoch'] = init_epoch

        # Resume training at resume_epoch
        # if resume_epoch is not None:
        #     self.restore_state(save_path, resume_epoch)
        #     init_epoch = self.agent_state_dict['epoch'] + 1
        #     nr_epochs -= init_epoch
        
        # Create tensorboard summary writer 
        self.summary_writer = create_writer(os.path.join(save_path, '..'), init_epoch)

        # Move model to GPUs
        if len(device_ids) > 1:
            self.model.set_data_parallel(device_ids)
        print('Using GPUs:', device_ids)

        # TODO for better eval
        # if init_epoch == 0:
        #     self.track_metrics(init_epoch, results, loss_f, eval_datasets)
        
        from rtpt import RTPT
        rtpt = RTPT(name_initials='MM', experiment_name='PrototypeCAE', max_iterations=nr_epochs)
        rtpt.start()

        for epoch in range(init_epoch, nr_epochs):
            self.agent_state_dict['epoch'] = epoch
            
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose
            acc = self.perform_training_epoch(loss_f, train_dataloader, config,
                print_run_loss=print_run_loss)

            rtpt.step(subtitle=f"loss={acc.mean('loss'):2.2f}")
            

            # Write losses to tensorboard
            if (epoch+1) % display_interval == 0:
                self.track_loss(acc, epoch+1, config)

            # Create visualizations and write them to tensorboard
            if (epoch+1) % display_interval == 0:
                self.track_visualization(train_dataloader, save_path, epoch+1, config, 'train')
                self.track_visualization(test_dataloader, save_path, epoch+1, config, 'test')

            # Save agent and optimizer state
            if (epoch+1) % save_interval == 0 and save_path is not None:
                self.save_state(save_path, epoch+1)

            # Track statistics in results
            if (epoch+1) % eval_interval == 0:
                self.track_metrics(epoch+1, results, loss_f, eval_datasets)

        self.model.finish()

    def perform_training_epoch(self, loss_f, train_dataloader, config,
        print_run_loss=False):
        r"""Perform a training epoch
        
        Args:
            print_run_loss (bool): whether a runing loss should be tracked and
                printed.
        """
        acc = Accumulator('loss')
        start_time = time.time()

        for data in tqdm(train_dataloader):
            # Get data
            inputs, targets = self.get_inputs_targets(data)

            # Forward pass
            outputs = self.get_outputs(inputs)

            # Optimization step
            self.model.unet_optim.zero_grad()

            loss_seg = loss_f(outputs, targets)

            if self.model.unet_old != None:
                outputs_old = self.get_outputs_old(inputs)
                loss_distill = self.multi_class_cross_entropy_no_softmax(outputs, outputs_old)
            else:
                if loss_seg.is_cuda:
                    loss_distill = torch.zeros(1).to(loss_seg.get_device())
                else:
                    loss_distill = torch.zeros(1)

            loss = loss_seg + config['lambda_d'] * loss_distill
            loss.backward()

            self.model.unet_optim.step()
            
            acc.add('loss', float(loss.detach().cpu()), count=len(inputs))
            acc.add('loss_seg', float(loss_seg.detach().cpu()), count=len(inputs))
            acc.add('loss_distill', float(loss_distill.detach().cpu()), count=len(inputs))

        # self.model.unet_scheduler.step()

        if print_run_loss:
            print('\nrunning loss: {} - time/epoch {}'.format(acc.mean('loss'), round(time.time()-start_time, 4)))

        return acc

    def get_inputs_targets(self, data, eval=True):
        r"""Prepares a data batch.

        Args:
            data (tuple): a dataloader item, possibly in cpu

        Returns (tuple): preprocessed data in the selected device.
        """
        inputs, targets = data
        inputs, targets = inputs.to(self.device), targets.to(self.device)    
        return inputs, targets.float()

    def track_metrics(self, epoch, results, loss_f, datasets):
        r"""Tracks metrics. Losses and scores are calculated for each 3D subject, 
        and averaged over the dataset.
        """
        for ds_name, ds in datasets.items():
            eval_dict = ds_losses_metrics(ds, self, loss_f, self.metrics)
            for metric_key in eval_dict.keys():
                results.add(epoch=epoch, metric='Mean_'+metric_key, data=ds_name, 
                    value=eval_dict[metric_key]['mean'])
                results.add(epoch=epoch, metric='Std_'+metric_key, data=ds_name, 
                    value=eval_dict[metric_key]['std'])
            if self.verbose:
                print('Epoch {} dataset {}'.format(epoch, ds_name))
                for metric_key in eval_dict.keys():
                    self.writer_add_scalar(f'metric/{metric_key}/{ds_name}', eval_dict[metric_key]['mean'], epoch)
                    print('{}: {}'.format(metric_key, eval_dict[metric_key]['mean']))

    def track_loss(self, acc, epoch, config, phase='train'):
        r'''Tracks loss in tensorboard.

        Args:
            acc (Accumulator): accumulator containing the tracked losses
            phase (string): either "test" or "train"
        '''
    
        self.writer_add_scalar(f'loss_{phase}/loss_seg', acc.mean('loss_seg'), epoch)
        self.writer_add_scalar(f'loss_{phase}/loss_distill', acc.mean('loss_distill'), epoch)
        self.writer_add_scalar(f'loss_{phase}/loss_comb', acc.mean('loss'), epoch)
    
    def track_visualization(self, dataloader, save_path, epoch, config, phase='train'):
        r'''Creates visualizations and tracks them in tensorboard.

        Args:
            dataloader (Dataloader): dataloader to draw sample from
            save_path (string): path for the images to be saved (one folder up)
            phase (string): either "test" or "train"
        '''
        for imgs in dataloader:
            x_i, y_i = self.get_inputs_targets(imgs, eval=False)
            x_i_seg = self.get_outputs(x_i)
            break
        
        # select sample with guaranteed segmentation label
        sample_idx = 0
        for i, y_ in enumerate(y_i):
            # if torch.count_nonzero(y_[1]) > 0:
            if len(torch.nonzero(y_[1])) > 0:
                sample_idx = i
                break
        x_i_img = x_i[sample_idx].unsqueeze(0)

        # segmentation
        x_i_seg = x_i_seg[sample_idx][1].unsqueeze(0).unsqueeze(0)
        threshold = 0.5
        x_i_seg_mask = (x_i_seg > threshold).int()
        y_i_seg_mask = y_i[sample_idx][1].unsqueeze(0).unsqueeze(0).int()

        save_path = os.path.join(save_path, '..', 'imgs')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path_pred = os.path.join(save_path, f'e_{epoch:06d}_{phase}_pred.png')
        save_path_label = os.path.join(save_path, f'e_{epoch:06d}_{phase}_label.png')
        
        plot_3d_segmentation(x_i_img, x_i_seg_mask, save_path=save_path_pred, img_size=(256, 256), alpha=0.5)
        plot_3d_segmentation(x_i_img, y_i_seg_mask, save_path=save_path_label, img_size=(256, 256), alpha=0.5)
        
        image = Image.open(save_path_pred)
        image = TF.to_tensor(image)
        self.writer_add_image(f'imgs_{phase}/pred', image, epoch)

        image = Image.open(save_path_label)
        image = TF.to_tensor(image)
        self.writer_add_image(f'imgs_{phase}/label', image, epoch)

    def save_state(self, states_path, epoch, optimizer=None, overwrite=False):
        r"""Saves an agent state. Raises an error if the directory exists and 
        overwrite=False.
        """
        if states_path is not None:
            state_name = f'epoch_{epoch:04d}'
            state_full_path = os.path.join(states_path, state_name)
            if os.path.exists(state_full_path):
                if not overwrite:
                    raise FileExistsError
                shutil.rmtree(state_full_path)
            os.makedirs(state_full_path)
            save_model_state_dataparallel(self.model, 'model', state_full_path)
            pkl_dump(self.agent_state_dict, 'agent_state_dict', state_full_path)
            
            # if no optimizer is set, try to save _optim attributes of model
            # if optimizer is None:
            #     attrs = dir(self.model)
            #     for att in attrs:
            #         if '_optim' in att:
            #             optim = getattr(self.model, att)
            #             # only save if attribute is optimizer
            #             try:
            #                 save_optimizer_state(optim, att, state_full_path)
            #             except:
            #                 pass

    def restore_state(self, states_path, epoch, optimizer=None):
        r"""Tries to restore a previous agent state, consisting of a model 
        state and the content of agent_state_dict. Returns whether the restore 
        operation  was successful.
        """
        
        if epoch == -1:
            state_name = os.listdir(states_path)[-1]
        else:
            state_name = f'epoch_{epoch:04d}'
        
        state_full_path = os.path.join(states_path, state_name)
        try:
            correct_load = load_model_state(self.model, 'model', state_full_path, device=self.device)
            assert correct_load
            agent_state_dict = pkl_load('agent_state_dict', state_full_path)
            assert agent_state_dict is not None
            self.agent_state_dict = agent_state_dict
            # if optimizer is not None: 
            #     load_optimizer_state(optimizer, 'optimizer', state_full_path, device=self.device)
            if self.verbose:
                print('State {} was restored'.format(state_name))
            return True
        except:
            print('State {} could not be restored'.format(state_name))
            return False

    def multi_class_cross_entropy_no_softmax(self, prediction, target):
        return (-(target * torch.log(prediction)).sum(dim=-1)).mean()

    def get_outputs_old(self, inputs):
        r"""Applies a softmax transformation to the model outputs"""
        outputs = self.model.forward_old(inputs)
        # softmax = nn.Softmax(dim=1)
        outputs = softmax(outputs)
        return outputs
# ------------------------------------------------------------------------------
# A disentangler agent.
# ------------------------------------------------------------------------------
import time
import os
import torch
import torch.nn as nn
from mp.agents.agent import Agent
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.evaluate import ds_losses_metrics, ds_losses_metrics_domain
from mp.eval.accumulator import Accumulator
from tqdm import tqdm

from mp.utils.pytorch.pytorch_load_restore import load_model_state, save_optimizer_state, load_optimizer_state, save_model_state_dataparallel

from torch.nn import functional as F
from mp.utils.tensorboard import create_writer

from mp.visualization.visualize_imgs import plot_3d_segmentation
from PIL import Image
import torchvision.transforms.functional as TF
from mp.utils.load_restore import pkl_dump, pkl_load


class DisentanglerAgent(SegmentationAgent):
    r"""An Agent for autoencoder models."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def perform_training_epoch(self, loss_f, train_dataloader, print_run_loss=False):
        r"""Perform a training epoch
        
        Args:
            print_run_loss (bool): whether a runing loss should be tracked and
                printed.
        """
        acc = Accumulator('loss')
        start_time = time.time()

        for data in tqdm(train_dataloader):
            
            self.model.zero_grad_optimizers()

            half_batch = train_dataloader.batch_size // 2
            x, y, domain_code = self.get_inputs_targets(data)
            
            x_i, y_i, domain_code_i = x[:half_batch], y[:half_batch], domain_code[:half_batch]
            x_j, y_j, domain_code_j = x[half_batch:], y[half_batch:], domain_code[half_batch:]

            skip_connections_x_i, content_x_i, style_sample_x_i = self.model.forward_enc(x_i)
            latent_scale_x_i = self.model.latent_scaler(style_sample_x_i)

            x_i_hat = self.model.forward_gen(content_x_i, latent_scale_x_i, domain_code_i)
            
            # VAE loss
            # KL_div = nn.KLDivLoss(reduction='batchmean')
            # z = self.model.sample_z(style_sample_x_i.shape)
            # if style_sample_x_i.is_cuda:
            #     z = z.to(style_sample_x_i.get_device())
            # loss_vae =  KL_div(style_sample_x_i, z) + torch.linalg.norm((x_i_hat-x_i).view(-1,1), ord=1)
            mu, log_var = self.model.forward_style_enc(x_i) 
            loss_dict = self.vae_loss(x_i_hat, x_i, mu, log_var)
            loss_vae = loss_dict['loss']
            acc.add('loss_vae', float(loss_vae.detach().cpu()), count=len(x_i))
            acc.add('loss_vae_Reconstruction_Loss', float(loss_dict['Reconstruction_Loss'].detach().cpu()), count=len(x_i))
            acc.add('loss_vae_KLD', float(loss_dict['KLD'].detach().cpu()), count=len(x_i))

            self.debug_print('loss_vae', loss_vae)

            # content adversarial loss
            domain_x_i = self.model.forward_con_dis(content_x_i)

            skip_connections_x_j, content_x_j, style_sample_x_j = self.model.forward_enc(x_j)
            domain_x_j =  self.model.forward_con_dis(content_x_j)

            loss_c_adv = torch.sum(torch.log(domain_x_i) + (1 - torch.log(domain_x_j)))
            acc.add('loss_c_adv', float(loss_c_adv.detach().cpu()), count=len(x_i))
            self.debug_print('loss_c_adv', loss_c_adv)
            
            # content reconstruction loss
            # content_x_j, style_sample_x_j = self.model.forward_encoder(x_j)
            latent_scale_x_j = self.model.latent_scaler(style_sample_x_j)
            x_j_hat = self.model.forward_gen(content_x_i, latent_scale_x_j, domain_code_j)
            skip_connections_x_j_hat, content_x_j_hat, style_sample_x_j_hat = self.model.forward_enc(x_j_hat)
            loss_c_recon = torch.sum(torch.linalg.norm((content_x_i-content_x_j_hat).view(-1,1), ord=1))
            acc.add('loss_c_recon', float(loss_c_recon.detach().cpu()), count=len(x_i))
            self.debug_print('loss_c_recon', loss_c_recon)

            # latent code regression loss
            z = self.model.sample_z(style_sample_x_i.shape)
            if style_sample_x_i.is_cuda:
                z = z.to(style_sample_x_i.get_device())
            latent_scale_z = self.model.latent_scaler(z)
            z_hat = self.model.forward_gen(content_x_i, latent_scale_z, domain_code_j)
            skip_connections_z_hat, z_hat_content, z_hat_sample = self.model.forward_enc(z_hat)
            z = self.model.sample_z(z_hat_sample.shape) # as big as generator output
            if z_hat_sample.is_cuda:
                z = z.to(z_hat_sample.get_device())
            loss_lcr = torch.sum(torch.linalg.norm((z-z_hat_sample).view(-1,1), ord=1))
            acc.add('loss_lcr', float(loss_lcr.detach().cpu()), count=len(x_i))
            self.debug_print('loss_lcr', loss_lcr)
            
            # GAN loss
            domain_x_j = self.model.forward_dom_dis(x_j, domain_code_j)
            domain_x_j_hat = self.model.forward_dom_dis(x_j_hat, domain_code_j)
            domain_z_hat = self.model.forward_dom_dis(z_hat, domain_code_j)
            # loss_gan = torch.sum(torch.log(domain_x_j)) + torch.sum(0.5*torch.log(1-domain_x_j_hat)) + torch.sum(0.5*torch.log(1-domain_z_hat))
            loss_gan_g = torch.sum(0.5*torch.log(1-domain_x_j_hat)) + torch.sum(0.5*torch.log(1-domain_z_hat))
            loss_gan_d = - (torch.sum(torch.log(domain_x_j)) + loss_gan_g)
            # TODO: check whether tripple update of discriminator is still required
            loss_gan_d = 3 * loss_gan_d
            
            loss_gan_d.backward(retain_graph=True)
            loss_gan_g.backward(retain_graph=True)

            acc.add('loss_gan_d', float(loss_gan_d.detach().cpu()), count=len(x_i))
            # self.debug_print('loss_gan_d', loss_gan_d, True)

            acc.add('loss_gan_g', float(loss_gan_g.detach().cpu()), count=len(x_i))
            # self.debug_print('loss_gan_g', loss_gan_g, True)

            # TODO: mode seeking loss

            # segmentation loss
            x_i_seg = self.model.forward_dec(skip_connections_x_i, content_x_i)
            loss_seg = loss_f(x_i_seg, y_i)
            acc.add('loss_seg', float(loss_seg.detach().cpu()), count=len(x_i))
            self.debug_print('loss_seg', loss_seg)
            
            # TODO: joint distribution structure discriminator loss
            
            lambda_vae = 1
            lambda_c_adv = 1
            lambda_lcr = 1e-3
            lambda_seg = 5
            lambda_c_recon = 1e-3
            
            loss_comb_no_gan = lambda_vae * loss_vae + lambda_c_adv * loss_c_adv + lambda_lcr * loss_lcr + lambda_seg * loss_seg + lambda_c_recon * loss_c_recon # + lambda_ms * loss_ms 
            loss_comb_no_gan.backward()

            self.model.step_optimizers()

            # TODO: fix -> loss_gan currently excluded
            loss_comb = loss_gan_d + loss_gan_g + lambda_vae * loss_vae + lambda_c_adv * loss_c_adv + lambda_lcr * loss_lcr + lambda_seg * loss_seg + lambda_c_recon * loss_c_recon # + lambda_ms * loss_ms 
            acc.add('loss_comb', float(loss_comb.detach().cpu()), count=len(x_i))
            self.debug_print('loss_comb', loss_comb)

        if print_run_loss:
            print('\nrunning loss: {} - time/epoch {}'.format(acc.mean('loss_comb'), round(time.time()-start_time, 4)))

        return acc

    

    # def iou_pytorch(self, outputs: torch.Tensor, labels: torch.Tensor):
    #     '''IoU implementation from https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    #     '''
    #     SMOOTH = 1e-6

    #     # You can comment out this line if you are passing tensors of equal shape
    #     # But if you are passing output from UNet or something it will most probably
    #     # be with the BATCH x 1 x H x W shape
    #     outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
        
    #     intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    #     union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
        
    #     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
        
    #     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
        
    #     return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

    def vae_loss(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        Source: https://github.com/AntixK/PyTorch-VAE/blob/20c4dfa73dfc36f42970ccc334a42f37ffe08dcc/models/vanilla_vae.py
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # Source: https://github.com/AntixK/PyTorch-VAE/blob/20c4dfa73dfc36f42970ccc334a42f37ffe08dcc/tests/test_vae.py
        kld_weight = 0.005 # kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(- 0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def train(self, results, loss_f, train_dataloader, test_dataloader,
        init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
        eval_datasets=dict(), eval_interval=10,
        save_path=None, save_interval=10,
        display_interval=1, resume_epoch=None, device_ids=[0]):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states.
        """
        
        self.agent_state_dict['epoch'] = init_epoch

        # Resume training at resume_epoch
        if resume_epoch is not None:
            self.restore_state(save_path, resume_epoch)
            init_epoch = self.agent_state_dict['epoch'] + 1
            
        # Create tensorboard summary writer 
        self.summary_writer = create_writer(os.path.join(save_path, '..'), init_epoch)

        # Move model to GPUs
        if len(device_ids) > 1:
            self.model.set_data_parallel(device_ids)
        print('Using GPUs:', device_ids)

        for epoch in range(init_epoch, init_epoch+nr_epochs):
            
            self.agent_state_dict['epoch'] = epoch
            
            print_run_loss = (epoch+1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose
            acc = self.perform_training_epoch(loss_f, train_dataloader, print_run_loss=print_run_loss)

            # Write losses to tensorboard
            if (epoch+1) % display_interval == 0:
                self.track_loss(acc, epoch+1)

            # Create visualizations and write them to tensorboard
            if (epoch+1) % display_interval == 0:
                self.track_visualization(train_dataloader, save_path, epoch+1)
                self.track_visualization(test_dataloader, save_path, epoch+1, 'test')

            # Save agent and optimizer state
            if (epoch+1) % save_interval == 0 and save_path is not None:
                self.save_state(save_path, epoch+1)

            # Track statistics in results
            if (epoch+1) % eval_interval == 0:
                self.track_metrics(epoch+1, results, loss_f, eval_datasets)

    def track_metrics(self, epoch, results, loss_f, datasets):
        r"""Tracks metrics. Losses and scores are calculated for each 3D subject, 
        and averaged over the dataset.
        """
        for ds_name, ds in datasets.items():
            eval_dict = ds_losses_metrics_domain(ds, self, loss_f, self.metrics)
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

    def track_loss(self, acc, epoch, phase='train'):
        r'''Tracks loss in tensorboard.

        Args:
            acc (Accumulator): accumulator containing the tracked losses
            phase (string): either "test" or "train"
        '''
        self.writer_add_scalar(f'loss_{phase}/loss_vae', acc.mean('loss_vae'), epoch)
        self.writer_add_scalar(f'loss_{phase}/loss_c_adv', acc.mean('loss_c_adv'), epoch)
        self.writer_add_scalar(f'loss_{phase}/loss_c_recon', acc.mean('loss_c_recon'), epoch)
        self.writer_add_scalar(f'loss_{phase}/loss_lcr', acc.mean('loss_lcr'), epoch)
        self.writer_add_scalar(f'loss_{phase}/loss_gan_d', acc.mean('loss_gan_d'), epoch)
        self.writer_add_scalar(f'loss_{phase}/loss_gan_g', acc.mean('loss_gan_g'), epoch)
        self.writer_add_scalar(f'loss_{phase}/loss_seg', acc.mean('loss_seg'), epoch)
        self.writer_add_scalar(f'loss_{phase}/loss_comb', acc.mean('loss_comb'), epoch)
        
        self.writer_add_scalar(f'loss_{phase}/loss_vae_Reconstruction_Loss', acc.mean('loss_vae_Reconstruction_Loss'), epoch)
        self.writer_add_scalar(f'loss_{phase}/loss_vae_KLD', acc.mean('loss_vae_KLD'), epoch)
        
    def track_visualization(self, dataloader, save_path, epoch, phase='train'):
        r'''Creates visualizations and tracks them in tensorboard.

        Args:
            dataloader (Dataloader): dataloader to draw sample from
            save_path (string): path for the images to be saved (one folder up)
            phase (string): either "test" or "train"
        '''
        for imgs in dataloader:
            x_i, y_i, domain_code_i = self.get_inputs_targets(imgs)
            x_i_seg = self.get_outputs(x_i)
            break

        x_i_img = x_i[0].unsqueeze(0)
        x_i_seg = x_i_seg[0][1].unsqueeze(0).unsqueeze(0)
        threshold = 0.5
        x_i_seg_mask = (x_i_seg > threshold).int()
        y_i_seg_mask = y_i[0][1].unsqueeze(0).unsqueeze(0).int()

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
        
    def get_inputs_targets(self, data):
        r"""Prepares a data batch.

        Args:
            data (tuple): a dataloader item, possibly in cpu

        Returns (tuple): preprocessed data in the selected device.
        """
        inputs, targets, domain_code = data
        inputs, targets, domain_code = inputs.to(self.device), targets.to(self.device), domain_code.to(self.device)
        inputs = self.model.preprocess_input(inputs)       
        return inputs, targets.float(), domain_code

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

# ------------------------------------------------------------------------------
# A disentangler agent.
# ------------------------------------------------------------------------------
import time
import torch
import torch.nn as nn
from mp.agents.agent import Agent
from mp.eval.evaluate import ds_losses_metrics, ds_losses_metrics_domain
from mp.eval.accumulator import Accumulator
from tqdm import tqdm

class DisentanglerAgent(Agent):
    r"""An Agent for autoencoder models."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def perform_training_epoch(self, optimizer, loss_f, train_dataloader, print_run_loss=False):
        r"""Perform a training epoch
        
        Args:
            print_run_loss (bool): whether a runing loss should be tracked and
                printed.
        """
        acc = Accumulator('loss')
        start_time = time.time()

        for data in tqdm(train_dataloader):
            
            half_batch = train_dataloader.batch_size // 2
            x, y, domain_code = self.get_inputs_targets(data)
            
            x_i, y_i, domain_code_i = x[:half_batch], y[:half_batch], domain_code[:half_batch]
            x_j, y_j, domain_code_j = x[half_batch:], y[half_batch:], domain_code[half_batch:]

            content_x_i, style_sample_x_i = self.model.forward_encoder(x_i)
            latent_scale_x_i = self.model.latent_scaler(style_sample_x_i)

            x_i_hat = self.model.forward_generator(content_x_i, latent_scale_x_i, domain_code_i)
            
            self.model.zero_grad_optimizers()

            # VAE loss
            KL_div = nn.KLDivLoss(reduction='batchmean')
            z = self.model.sample_z(style_sample_x_i.shape)
            loss_vae =  KL_div(style_sample_x_i, z) + torch.linalg.norm((x_i_hat-x_i).view(-1,1), ord=1)
            acc.add('loss_vae', float(loss_vae.detach().cpu()), count=len(x_i))
            self.debug_print('loss_vae', loss_vae)

            # content adversarial loss
            domain_x_i = self.model.forward_content_discriminator(content_x_i)

            content_x_j, style_sample_x_j = self.model.forward_encoder(x_j)
            domain_x_j =  self.model.forward_content_discriminator(content_x_j)

            loss_c_adv = torch.sum(torch.log(domain_x_i)) + torch.sum(1 - torch.log(domain_x_j))
            acc.add('loss_c_adv', float(loss_c_adv.detach().cpu()), count=len(x_i))
            self.debug_print('loss_c_adv', loss_c_adv)
            
            # content reconstruction loss
            # content_x_j, style_sample_x_j = self.model.forward_encoder(x_j)
            latent_scale_x_j = self.model.latent_scaler(style_sample_x_j)
            x_j_hat = self.model.forward_generator(content_x_i, latent_scale_x_j, domain_code_j)
            content_x_j_hat, style_sample_x_j_hat = self.model.forward_encoder(x_j_hat)
            loss_c_recon = torch.sum(torch.linalg.norm((content_x_i-content_x_j_hat).view(-1,1), ord=1))
            acc.add('loss_c_recon', float(loss_c_recon.detach().cpu()), count=len(x_i))
            self.debug_print('loss_c_recon', loss_c_recon)

            # latent code regression loss
            latent_scale_z = self.model.latent_scaler(z)
            z_hat = self.model.forward_generator(content_x_i, latent_scale_z, domain_code_j)
            z_hat_content, z_hat_sample = self.model.forward_encoder(z_hat)
            z = self.model.sample_z(z_hat_sample.shape) # as big as generator output
            loss_lcr = torch.sum(torch.linalg.norm((z-z_hat_sample).view(-1,1), ord=1))
            acc.add('loss_lcr', float(loss_lcr.detach().cpu()), count=len(x_i))
            self.debug_print('loss_lcr', loss_lcr)
            
            # GAN loss
            domain_x_j = self.model.forward_multi_discriminator(x_j, domain_code_j)
            domain_x_j_hat = self.model.forward_multi_discriminator(x_j_hat, domain_code_j)
            domain_z_hat = self.model.forward_multi_discriminator(z_hat, domain_code_j)
            # loss_gan = torch.sum(torch.log(domain_x_j)) + torch.sum(0.5*torch.log(1-domain_x_j_hat)) + torch.sum(0.5*torch.log(1-domain_z_hat))
            loss_gan_d = - torch.sum(torch.log(domain_x_j)) - torch.sum(0.5*torch.log(1-domain_x_j_hat)) - torch.sum(0.5*torch.log(1-domain_z_hat))
            loss_gan_d = 3*loss_gan_d
            loss_gan_g = torch.sum(0.5*torch.log(1-domain_x_j_hat)) + torch.sum(0.5*torch.log(1-domain_z_hat))
            
            loss_gan_d.backward(retain_graph=True)
            loss_gan_g.backward(retain_graph=True)

            acc.add('loss_gan_d', float(loss_gan_d.detach().cpu()), count=len(x_i))
            # self.debug_print('loss_gan_d', loss_gan_d, True)

            acc.add('loss_gan_g', float(loss_gan_g.detach().cpu()), count=len(x_i))
            # self.debug_print('loss_gan_g', loss_gan_g, True)

            # import numpy as np
            # if loss_gan.item() == -np.inf:
            #     print(domain_x_j)
            #     print(domain_x_j_hat)
            #     print(domain_z_hat)
            #     exit(42)

            # TODO: mode seeking loss

            # segmentation loss
            x_i_seg_in = self.model.forward_generator(content_x_i, latent_scale_x_i, torch.zeros(domain_code_i.shape).to(self.model.device))
            x_i_seg = self.model.forward_segmentation(x_i_seg_in)
            loss_seg = loss_f(x_i_seg, y_i)
            acc.add('loss_seg', float(loss_seg.detach().cpu()), count=len(x_i))
            self.debug_print('loss_seg', loss_seg)


            # from mp.visualization.visualize_imgs import plot_3d_segmentation
            # plot_3d_segmentation(x_i[0].unsqueeze_(0), x_i_seg[0][1].unsqueeze_(0).unsqueeze_(0), save_path='pred.png', img_size=(256, 256), alpha=0.5)
            # plot_3d_segmentation(y_i[0][0].unsqueeze_(0).unsqueeze_(0), y_i[0][1].unsqueeze_(0).unsqueeze_(0), save_path='label.png', img_size=(256, 256), alpha=0.5)


            # TODO: joint distribution structure discriminator loss
            
            lambda_vae = 1
            lambda_c_adv = 1
            lambda_lcr = 10
            lambda_seg = 5
            lambda_c_recon = 1
            lambda_ms = 1
            
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

    def train(self, results, optimizer, loss_f, train_dataloader,
        init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
        eval_datasets=dict(), eval_interval=10, 
        save_path=None, save_interval=10,
        display_interval=1):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states.
        """
        for epoch in range(init_epoch, init_epoch+nr_epochs):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose
            acc = self.perform_training_epoch(optimizer, loss_f, train_dataloader,
                print_run_loss=print_run_loss)

            # Write losses to tensorboard
            if (epoch + 1) % display_interval == 0:
                self.track_loss(acc, epoch + 1)

            # Save agent and optimizer state
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1), optimizer)

            # Track statistics in results
            if (epoch + 1) % eval_interval == 0:
                self.track_metrics(epoch + 1, results, loss_f, eval_datasets)
    
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

    def track_loss(self, acc, epoch):
        r'''Tracks loss in tensorboard.

        Args:
            acc (Accumulator): accumulator containing the tracked losses
        '''
        self.writer_add_scalar('loss/loss_vae', acc.mean('loss_vae'), epoch)
        self.writer_add_scalar('loss/loss_c_adv', acc.mean('loss_c_adv'), epoch)
        self.writer_add_scalar('loss/loss_c_recon', acc.mean('loss_c_recon'), epoch)
        self.writer_add_scalar('loss/loss_lcr', acc.mean('loss_lcr'), epoch)
        self.writer_add_scalar('loss/loss_gan_d', acc.mean('loss_gan_d'), epoch)
        self.writer_add_scalar('loss/loss_gan_g', acc.mean('loss_gan_g'), epoch)
        self.writer_add_scalar('loss/loss_seg', acc.mean('loss_seg'), epoch)
        self.writer_add_scalar('loss/loss_comb', acc.mean('loss_comb'), epoch)

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

    def get_outputs(self, inputs, domain_code):
        r"""Returns model outputs.
        Args:
            inputs (torch.tensor): inputs
            domain_code (torch.tensor): domain codes
        Returns (torch.tensor): model outputs, with one channel dimension per 
            label.
        """
        return self.model.forward(inputs, domain_code)
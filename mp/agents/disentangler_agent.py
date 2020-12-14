# ------------------------------------------------------------------------------
# A disentangler agent.
# ------------------------------------------------------------------------------
import time
import torch
import torch.nn as nn
from mp.agents.agent import Agent
from mp.eval.evaluate import ds_losses_metrics
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

            # TODO: maybe introduce randomness to swap x_i and x_j to train on both diretions?

            optimizer.zero_grad()
            
            content_x_i, style_sample_x_i = self.model.forward_encoder(x_i)
            latent_scale_x_i = self.model.latent_scaler(style_sample_x_i)
            x_i_hat = self.model.forward_generator(content_x_i, latent_scale_x_i, domain_code_i)
            
            # VAE loss
            KL_div = nn.KLDivLoss()
            z = self.model.sample_z(style_sample_x_i.shape)
            loss_vae =  KL_div(style_sample_x_i, z) + torch.linalg.norm((x_i_hat-x_i).view(-1,1), ord=1)
            self.writer_add_scalar('loss/loss_vae', loss_vae.detach().cpu())
            acc.add('loss_vae', float(loss_vae.detach().cpu()), count=len(x_i))
            self.debug_print('loss_vae', loss_vae)

            # content adversarial loss
            domain_x_i = self.model.forward_content_discriminator(content_x_i)

            content_x_j, style_sample_x_j = self.model.forward_encoder(x_j)
            domain_x_j =  self.model.forward_content_discriminator(content_x_j)

            loss_c_adv = torch.sum(torch.log(domain_x_i)) + torch.sum(1 - torch.log(domain_x_j))
            self.writer_add_scalar('loss/loss_c_adv', loss_c_adv.detach().cpu())
            acc.add('loss_c_adv', float(loss_c_adv.detach().cpu()), count=len(x_i))
            self.debug_print('loss_c_adv', loss_c_adv)
            
            # content reconstruction loss
            # content_x_j, style_sample_x_j = self.model.forward_encoder(x_j)
            latent_scale_x_j = self.model.latent_scaler(style_sample_x_j)
            x_j_hat = self.model.forward_generator(content_x_i, latent_scale_x_j, domain_code_j)
            content_x_j_hat, style_sample_x_j_hat = self.model.forward_encoder(x_j_hat)
            loss_c_recon = torch.sum(torch.linalg.norm((content_x_i-content_x_j_hat).view(-1,1), ord=1))
            self.writer_add_scalar('loss/loss_c_recon', loss_c_recon.detach().cpu())
            acc.add('loss_c_recon', float(loss_c_recon.detach().cpu()), count=len(x_i))
            self.debug_print('loss_c_recon', loss_c_recon)

            # latent code regression loss
            latent_scale_z = self.model.latent_scaler(z)
            z_hat = self.model.forward_generator(content_x_i, latent_scale_z, domain_code_j)
            z_hat_content, z_hat_sample = self.model.forward_encoder(z_hat)
            z = self.model.sample_z(z_hat_sample.shape) # as big as generator output
            loss_lcr = torch.sum(torch.linalg.norm((z-z_hat_sample).view(-1,1), ord=1))
            self.writer_add_scalar('loss/loss_lcr', loss_lcr.detach().cpu())
            acc.add('loss_lcr', float(loss_lcr.detach().cpu()), count=len(x_i))
            self.debug_print('loss_lcr', loss_lcr)
            
            # GAN loss
            domain_x_j = self.model.forward_multi_discriminator(x_j, domain_code_j)
            domain_x_j_hat = self.model.forward_multi_discriminator(x_j_hat, domain_code_j)
            domain_z_hat = self.model.forward_multi_discriminator(z_hat, domain_code_j)
            loss_gan = torch.sum(torch.log(domain_x_j)) + torch.sum(0.5*torch.log(1-domain_x_j_hat)) + torch.sum(0.5*torch.log(1-domain_z_hat))
            self.writer_add_scalar('loss/loss_gan', loss_gan.detach().cpu())
            acc.add('loss_gan', float(loss_gan.detach().cpu()), count=len(x_i))
            self.debug_print('loss_gan', loss_gan, True)

            import numpy as np
            if loss_gan.item() == -np.inf:
                print(domain_x_j)
                print(domain_x_j_hat)
                print(domain_z_hat)
                exit(42)

            # TODO: mode seeking loss

            # segmentation loss
            x_i_seg_in = self.model.forward_generator(content_x_i, latent_scale_x_i, torch.zeros(domain_code_i.shape).to(self.model.device))
            x_i_seg = self.model.forward_segmentation(x_i_seg_in)
            loss_seg = loss_f(x_i_seg, y_i)
            self.writer_add_scalar('loss/loss_seg', loss_seg.detach().cpu())
            acc.add('loss_seg', float(loss_seg.detach().cpu()), count=len(x_i))
            self.debug_print('loss_seg', loss_seg)

            # TODO: joint distribution structure discriminator loss
            
            lambda_vae = 1
            lambda_c_adv = 1
            lambda_lcr = 10
            lambda_seg = 5
            lambda_c_recon = 1
            lambda_ms = 1
            
            # TODO: fix -> loss_gan currently excluded
            loss_comb = loss_gan + lambda_vae * loss_vae + lambda_c_adv * loss_c_adv + lambda_lcr * loss_lcr + lambda_seg * loss_seg + lambda_c_recon * loss_c_recon # + lambda_ms * loss_ms 
            self.writer_add_scalar('loss/loss_comb', loss_comb.detach().cpu())
            acc.add('loss_comb', float(loss_comb.detach().cpu()), count=len(x_i))
            self.debug_print('loss_comb', loss_comb)

            loss_gan.backward() # retain_graph=True)
            # loss_gan.backward()
            # Optimization step
            optimizer.step()

        if print_run_loss:
            print('\nrunning loss: {} - time/epoch {}'.format(acc.mean('loss'), round(time.time()-start_time, 4)))

    def train(self, results, optimizer, loss_f, train_dataloader,
        init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
        eval_datasets=dict(), eval_interval=10, 
        save_path=None, save_interval=10):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states.
        """
        # TODO: pass domain_code with eval_datasets
        # if init_epoch == 0:
            # self.track_metrics(init_epoch, results, loss_f, eval_datasets)

        for epoch in range(init_epoch, init_epoch+nr_epochs):
            self.current_epoch = epoch
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose
            self.perform_training_epoch(optimizer, loss_f, train_dataloader,
                print_run_loss=print_run_loss)
        
            # Track statistics in results
            # TODO: pass domain_code with datasets
            # if (epoch + 1) % eval_interval == 0:
            #     self.track_metrics(epoch + 1, results, loss_f, eval_datasets)

            # Save agent and optimizer state
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1), optimizer)

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
                    print('{}: {}'.format(metric_key, eval_dict[metric_key]['mean']))

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

    def get_outputs(self, inputs):
        r"""Returns model outputs.
        Args:
            data (torch.tensor): inputs, domain codes

        Returns (torch.tensor): model outputs, with one channel dimension per 
            label.
        """
        pass
# ------------------------------------------------------------------------------
# A disentangler agent.
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from mp.agents.agent import Agent
from mp.eval.evaluate import ds_losses_metrics
from mp.eval.accumulator import Accumulator

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
        for _, data in enumerate(train_dataloader):
            x_i, y_i = self.get_inputs_targets(data)
            
            # TODO: remove!!!
            x_j = x_i
            optimizer.zero_grad()
            
            domain_code_i = torch.zeros(10).to('cuda:0')
            domain_code_i[3] = 1
            domain_code_j = torch.zeros(10).to('cuda:0')
            domain_code_j[3] = 1

            content_x_i, style_sample_x_i = self.model.forward_encoder(x_i)
            latent_scale_x_i = self.model.latent_scaler(style_sample_x_i)
            x_i_hat = self.model.forward_generator(content_x_i, latent_scale_x_i, domain_code_i)
            
            # VAE loss
            KL_div = nn.KLDivLoss()
            z = self.model.sample_z(style_sample_x_i.shape)
            loss_vae =  KL_div(style_sample_x_i, z) + torch.linalg.norm((x_i_hat-x_i).view(-1,1), ord=1)
            print('loss_vae', loss_vae)

            # content adversarial loss
            domain_x_i = self.model.forward_content_discriminator(content_x_i)

            content_x_j, style_sample_x_j = self.model.forward_encoder(x_j)
            domain_x_j =  self.model.forward_content_discriminator(content_x_j)

            loss_c_adv = torch.sum(torch.log(domain_x_i)) + torch.sum(1 - torch.log(domain_x_j))
            print('loss_c_adv', loss_c_adv)
            
            # content reconstruction loss
            # content_x_j, style_sample_x_j = self.model.forward_encoder(x_j)
            latent_scale_x_j = self.model.latent_scaler(style_sample_x_j)
            x_j_hat = self.model.forward_generator(content_x_i, latent_scale_x_j, domain_code_j)
            content_x_j_hat, style_sample_x_j_hat = self.model.forward_encoder(x_j_hat)
            loss_c_recon = torch.sum(torch.linalg.norm((content_x_i-content_x_j_hat).view(-1,1), ord=1))
            print('loss_c_recon', loss_c_recon)

            # latent code regression loss
            latent_scale_z = self.model.latent_scaler(z)
            z_hat = self.model.forward_generator(content_x_i, latent_scale_z, domain_code_j)
            z_hat_content, z_hat_sample = self.model.forward_encoder(z_hat)
            z = self.model.sample_z(z_hat_sample.shape) # as big as generator output
            loss_lcr = torch.sum(torch.linalg.norm((z-z_hat_sample).view(-1,1), ord=1))
            print('loss_lcr', loss_lcr)
            
            # GAN loss
            domain_x_j = self.model.forward_multi_discriminator(x_j, domain_code_j)
            domain_x_j_hat = self.model.forward_multi_discriminator(x_j_hat, domain_code_j)
            domain_z_hat = self.model.forward_multi_discriminator(z_hat, domain_code_j)
            loss_gan = torch.sum(torch.log(domain_x_j)) + torch.sum(0.5*torch.log(1-domain_x_j_hat)) + torch.sum(0.5*torch.log(1-domain_z_hat))
            print('loss_gan', loss_gan)

            # TODO: mode seeking loss

            # segmentation loss
            x_i_seg_in = self.model.forward_generator(content_x_i, latent_scale_x_i, torch.zeros(domain_code_i.shape).to('cuda:0'))
            x_i_seg = self.model.forward_segmentation(x_i_seg_in)
            loss_seg = loss_f(x_i_seg, y_i)
            print('loss_seg', loss_seg)

            # TODO: joint distribution structure discriminator loss
            
            lambda_vae = 1
            lambda_c_adv = 1
            lambda_lcr = 10
            lambda_seg = 5
            lambda_c_recon = 1
            lambda_ms = 1

            loss = loss_gan + lambda_vae * loss_vae + lambda_c_adv * loss_c_adv + lambda_lcr * loss_lcr + lambda_seg * loss_seg + lambda_c_recon * loss_c_recon # + lambda_ms * loss_ms 

            # for l in [loss_vae, loss_c_adv, loss_lcr, loss_seg, loss_c_recon]:
            #     print(l)
                
            # exit(42)
            # Optimization step
            loss.backward()
            optimizer.step()

            acc.add('loss', float(loss.detach().cpu()), count=len(x_i))

        if print_run_loss:
            print('\nRunning loss: {}'.format(acc.mean('loss')))

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
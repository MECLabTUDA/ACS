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


class WhateverAgent(SegmentationAgent):
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
        
        for epoch in range(init_epoch, nr_epochs):
            self.agent_state_dict['epoch'] = epoch
            
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose
            acc = self.perform_training_epoch(loss_f, train_dataloader, config,
                print_run_loss=print_run_loss)
        
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

    def perform_training_epoch(self, loss_f, train_dataloader, config,
        print_run_loss=False):
        r"""Perform a training epoch
        
        Args:
            print_run_loss (bool): whether a runing loss should be tracked and
                printed.
        """
        acc = Accumulator('loss')
        start_time = time.time()

        if config['unet_only']:
            for data in tqdm(train_dataloader):
                # Get data
                inputs, targets, _ = self.get_inputs_targets(data, eval=False)

                # Forward pass
                outputs = self.get_outputs(inputs)

                # Optimization step
                self.model.unet_optim.zero_grad()

                loss = loss_f(outputs, targets)
                loss.backward()
                
                self.model.unet_optim.step()

                acc.add('loss', float(loss.detach().cpu()), count=len(inputs))
                acc.add('loss_seg', float(loss.detach().cpu()), count=len(inputs))

        elif config['continual']:
            for data in tqdm(train_dataloader):
            # Get data
                inputs, targets, _ = self.get_inputs_targets(data, eval=False)

                # Forward pass
                outputs = self.get_outputs(inputs)

                # Optimization step
                self.model.unet_decoder_optim.zero_grad()
                self.model.unet_classifier_optim.zero_grad()

                loss = loss_f(outputs, targets)
                loss.backward()
                
                self.model.unet_decoder_optim.step()
                self.model.unet_classifier_optim.step()

                acc.add('loss', float(loss.detach().cpu()), count=len(inputs))
                acc.add('loss_seg', float(loss.detach().cpu()), count=len(inputs))

        else:
            for data in tqdm(train_dataloader):

                x, y, domain_code = self.get_inputs_targets(data, eval=False)
                
                loss_vae_gen, acc = self.update_vae_gen(x, y, domain_code, acc, config, loss_f)

                # TODO: try update discriminator/segmentor twice
                for _ in range(config['d_iter']):
                    loss_dis_seg, acc = self.update_dis_seg(x, y, domain_code, acc, config, loss_f)

                loss_comb = loss_vae_gen + loss_dis_seg
                acc.add('loss', float(loss_comb.detach().cpu()), count=len(x))

        if print_run_loss:
            print('\nrunning loss: {} - time/epoch {}'.format(acc.mean('loss'), round(time.time()-start_time, 4)))

        return acc

    def get_inputs_targets(self, data, eval=True):
        r"""Prepares a data batch.

        Args:
            data (tuple): a dataloader item, possibly in cpu

        Returns (tuple): preprocessed data in the selected device.
        """
        if eval:
            inputs, targets, _ = data
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # inputs = self.model.preprocess_input(inputs)       
            return inputs, targets.float()
        else:
            inputs, targets, domain_code = data
            inputs, targets, domain_code = inputs.to(self.device), targets.to(self.device), domain_code.to(self.device)
            # inputs = self.model.preprocess_input(inputs)       
            return inputs, targets.float(), domain_code

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
        if not config['unet_only'] and not config['continual']:
            self.writer_add_scalar(f'loss_{phase}/loss_vae', acc.mean('loss_vae'), epoch)
            self.writer_add_scalar(f'loss_{phase}/loss_vae_Reconstruction_Loss', acc.mean('loss_vae_Reconstruction_Loss'), epoch)
            self.writer_add_scalar(f'loss_{phase}/loss_vae_KLD', acc.mean('loss_vae_KLD'), epoch)
            # self.writer_add_scalar(f'loss_{phase}/loss_c_adv', acc.mean('loss_c_adv'), epoch)
            self.writer_add_scalar(f'loss_{phase}/loss_c_adv_d', acc.mean('loss_c_adv_d'), epoch)
            self.writer_add_scalar(f'loss_{phase}/loss_c_adv_e', acc.mean('loss_c_adv_e'), epoch)
            self.writer_add_scalar(f'loss_{phase}/loss_c_recon', acc.mean('loss_c_recon'), epoch)
            self.writer_add_scalar(f'loss_{phase}/loss_lcr', acc.mean('loss_lcr'), epoch)
            self.writer_add_scalar(f'loss_{phase}/loss_gan_d', acc.mean('loss_gan_d'), epoch)
            self.writer_add_scalar(f'loss_{phase}/loss_gan_g', acc.mean('loss_gan_g'), epoch)

        self.writer_add_scalar(f'loss_{phase}/loss_seg', acc.mean('loss_seg'), epoch)
        self.writer_add_scalar(f'loss_{phase}/loss_comb', acc.mean('loss'), epoch)
    
    def track_visualization(self, dataloader, save_path, epoch, config, phase='train'):
        r'''Creates visualizations and tracks them in tensorboard.

        Args:
            dataloader (Dataloader): dataloader to draw sample from
            save_path (string): path for the images to be saved (one folder up)
            phase (string): either "test" or "train"
        '''
        for imgs in dataloader:
            x_i, y_i, domain_code_i = self.get_inputs_targets(imgs, eval=False)
            x_i_seg = self.get_outputs(x_i)
            break
        
        # select sample with guaranteed segmentation label
        sample_idx = 0
        for i, y_ in enumerate(y_i):
            if torch.count_nonzero(y_[1]) > 0:
                sample_idx = i
                break
        x_i_img = x_i[sample_idx].unsqueeze(0)
        x_i_domain = domain_code_i[sample_idx].unsqueeze(0)

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

        if not config['unet_only']:
            # gan
            skip_connections_x, content_x, style_sample_x = self.model.forward_enc(x_i_img)
            latent_scale_x = self.model.latent_scaler(style_sample_x)
            x_hat = self.model.forward_gen(content_x, latent_scale_x, x_i_domain)

            save_path_gan = os.path.join(save_path, f'e_{epoch:06d}_{phase}_gan.png')

            plot_3d_segmentation(x_hat, torch.zeros(x_hat.shape), save_path=save_path_gan, img_size=(256, 256), alpha=0.5)

            image = Image.open(save_path_gan)
            image = TF.to_tensor(image)
            self.writer_add_image(f'imgs_{phase}/gan', image, epoch)

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

    def update_vae_gen(self, x, y, domain_code, acc, config, loss_f):
     
        self.model.zero_grad_optim_vae_gen()
        
        # vae loss
        skip_connections_x, content_x, style_sample_x = self.model.forward_enc(x)

        latent_scale_x = self.model.latent_scaler(style_sample_x)
        x_hat = self.model.forward_gen(content_x, latent_scale_x, domain_code)
        mu, log_var = self.model.forward_style_enc(x) 

        loss_dict = self.vae_loss(x_hat, x, mu, log_var)
        loss_vae = loss_dict['loss']
        acc.add('loss_vae', float(loss_vae.detach().cpu()), count=len(x))
        acc.add('loss_vae_Reconstruction_Loss', float(loss_dict['Reconstruction_Loss'].detach().cpu()), count=len(x))
        acc.add('loss_vae_KLD', float(loss_dict['KLD'].detach().cpu()), count=len(x))

        # content reconstruction loss
        # skip_connections_x_i, content_x_i, style_sample_x_i = self.model.forward_enc(x_i)
        # skip_connections_x_j, content_x_j, style_sample_x_j = self.model.forward_enc(x_j)

        # latent_scale_x_j = self.model.latent_scaler(style_sample_x_j)
        # x_j_hat = self.model.forward_gen(content_x_i, latent_scale_x_j, domain_code_j)
        # skip_connections_x_j_hat, content_x_j_hat, style_sample_x_j_hat = self.model.forward_enc(x_j_hat)

        # latent_scale_x_i = self.model.latent_scaler(style_sample_x_i)
        # x_i_hat = self.model.forward_gen(content_x_j, latent_scale_x_i, domain_code_i)
        # skip_connections_x_i_hat, content_x_i_hat, style_sample_x_i_hat = self.model.forward_enc(x_i_hat)

        # loss_c_recon_i = torch.mean(torch.linalg.norm((content_x_i-content_x_j_hat).view(-1,1), ord=1))
        # loss_c_recon_j = torch.mean(torch.linalg.norm((content_x_j-content_x_i_hat).view(-1,1), ord=1))

        skip_connections_x_hat, content_x_hat, style_sample_x_hat = self.model.forward_enc(x_hat)
        loss_c_recon = torch.mean(torch.linalg.norm((content_x-content_x_hat).view(-1,1), ord=1))

        # loss_c_recon = (loss_c_recon_i + loss_c_recon_j)/2
        acc.add('loss_c_recon', float(loss_c_recon.detach().cpu()), count=len(x))

        # gan
        z = self.model.sample_z(style_sample_x.shape)
        if style_sample_x.is_cuda:
            z = z.to(style_sample_x.get_device())
        latent_scale_z = self.model.latent_scaler(z)
        z_hat = self.model.forward_gen(content_x, latent_scale_z, domain_code)
        domain_z_hat = self.model.forward_dom_dis(z_hat, domain_code)
        # domain_x = self.model.forward_dom_dis(x, domain_code)
        # loss_gan_g = torch.mean(torch.log(domain_x)) + torch.mean(torch.log(1-domain_z_hat))

        all_ones = torch.ones_like(domain_z_hat)
        if domain_z_hat.is_cuda:
            all_ones = all_ones.to(domain_z_hat.get_device())
        fake_loss = nn.functional.binary_cross_entropy_with_logits(domain_z_hat, all_ones)
        loss_gan_g = fake_loss
        # print('loss_gan_g', loss_gan_g)
        # print('dom_dis loss_gan_g domain_z_hat', domain_z_hat.shape, 'domain_code', all_ones.shape)

        # loss_gan_g = torch.mean(torch.log(1-domain_z_hat))
        acc.add('loss_gan_g', float(loss_gan_g.detach().cpu()), count=len(x))

        # lcr loss
        skip_connections_z_hat, z_hat_content, z_hat_sample = self.model.forward_enc(z_hat)
        # TODO: we have to take z from above, cause lcr loss is kind of a style reconstruction loss
        # z = self.model.sample_z(z_hat_sample.shape) # as big as generator output
        # if z_hat_sample.is_cuda:
        #     z = z.to(z_hat_sample.get_device())
            
        loss_lcr = torch.mean(torch.linalg.norm((z-z_hat_sample).view(-1,1), ord=1))
        acc.add('loss_lcr', float(loss_lcr.detach().cpu()), count=len(x))
       
        # skip_connections_x_i, content_x_i, style_sample_x_i = self.model.forward_enc(x_i)
        # domain_x_i = self.model.forward_con_dis(content_x_i)
        # skip_connections_x_j, content_x_j, style_sample_x_j = self.model.forward_enc(x_j)
        # domain_x_j =  self.model.forward_con_dis(content_x_j)
        
        # bce_loss = nn.BCEWithLogitsLoss()
        # bce_i = bce_loss(domain_x_i, domain_code_i)
        # bce_j = bce_loss(domain_x_j, domain_code_j)
        # loss_c_adv = -(bce_i + bce_j)
        # loss_c_adv = config['lambda_c_adv']*loss_c_adv

        # segmentation loss
        # x_i_seg = self.model.forward_dec(skip_connections_x_i, content_x_i)
        # x_seg = self.model.unet(x)
        x_seg = self.get_outputs(x)
        loss_seg = loss_f(x_seg, y)
        acc.add('loss_seg', float(loss_seg.detach().cpu()), count=len(x))

        # TODO try
        # domain_x = self.model.forward_con_dis(content_x)
        domain_x = self.model.forward_con_dis(skip_connections_x, content_x)
        # make discriminator believe all domains are equally likely i.e. domain can't be reconstructed
        # TODO TODO
        # domain_code_fake = torch.zeros_like(domain_code) # * (1 / domain_code.shape[-1])
        # print('lcon_dis c_adv_e domain_x', domain_x.shape, 'domain_code_fake', domain_code_fake.shape)
        # _, domain_index = torch.max(domain_code, dim=1)
        # loss_c_adv_e = - torch.nn.functional.cross_entropy(domain_x, domain_index)
        domain_dummy = torch.zeros_like(domain_x)
        domain_dummy[-1] = 1
        loss_c_adv_e = self.multi_class_cross_entropy_with_softmax(domain_x, domain_dummy)

        # loss_c_adv_e = -1 * nn.functional.binary_cross_entropy_with_logits(domain_x, domain_code)
        # loss_c_adv = torch.mean(torch.log(domain_x_i) + torch.mean(1 - torch.log(domain_x_j)))
        acc.add('loss_c_adv_e', float(loss_c_adv_e.detach().cpu()), count=len(x))

        loss_g_vae = config['lambda_c_adv']*loss_c_adv_e*3 + config['lambda_vae']*loss_vae + config['lambda_c_recon']*loss_c_recon + config['lambda_gan']*loss_gan_g + config['lambda_lcr']*loss_lcr + config['lambda_seg']*loss_seg # + loss_c_adv
        loss_g_vae.backward()
        
        self.model.step_optim_vae_gen()

        return loss_g_vae, acc

    def update_dis_seg(self, x, y, domain_code, acc, config, loss_f):
        
        self.model.zero_grad_optim_dis_seg()

        # content adversarial loss
        skip_connections_x, content_x, style_sample_x = self.model.forward_enc(x)
        # domain_x = self.model.forward_con_dis(content_x)
        domain_x = self.model.forward_con_dis(skip_connections_x, content_x)
        # print('con_dis c_adv_d domain_x', domain_x.shape, 'domain_code', domain_code.shape)
        _, domain_index = torch.max(domain_code, dim=1)

        # content_z = self.model.sample_z(content_x.shape)
        # skip_connections_z = self.model.sample_z(skip_connections_x.shape)
        # if content_x.is_cuda:
        #     content_z = content_z.to(content_x.get_device())
        #     skip_connections_z = skip_connections_z.to(content_x.get_device())
        # # domain_z = self.model.forward_con_dis(content_z)
        # domain_z = self.model.forward_con_dis(skip_connections_z, content_z)
        # domain_dummy = torch.zeros_like(domain_x)
        # domain_dummy[-1] = 1
        # loss_c_adv_d_dummy = self.multi_class_cross_entropy_with_softmax(domain_z, domain_dummy)

        loss_c_adv_d_real = self.multi_class_cross_entropy_with_softmax(domain_x, domain_code)

        loss_c_adv_d = loss_c_adv_d_real # + 1/2 * loss_c_adv_d_dummy
        # loss_c_adv_d_ = torch.nn.functional.cross_entropy(domain_x, domain_index)

        # print(loss_c_adv_d, loss_c_adv_d_)

        # loss_c_adv = torch.mean(torch.log(domain_x_i) + torch.mean(1 - torch.log(domain_x_j)))
        acc.add('loss_c_adv_d', float(loss_c_adv_d.detach().cpu()), count=len(x))
        
        # gan
        z = self.model.sample_z(style_sample_x.shape)
        if style_sample_x.is_cuda:
            z = z.to(style_sample_x.get_device())
        latent_scale_z = self.model.latent_scaler(z)
        z_hat = self.model.forward_gen(content_x, latent_scale_z, domain_code)
        domain_z_hat = self.model.forward_dom_dis(z_hat, domain_code)
        domain_x = self.model.forward_dom_dis(x, domain_code)
        # loss_gan_d = -1 * (torch.mean(torch.log(domain_x)) + torch.mean(torch.log(1-domain_z_hat)))

        all_ones = torch.ones_like(domain_x)
        all_zeros = torch.zeros_like(domain_z_hat)
        if domain_x.is_cuda:
            all_ones = all_ones.to(domain_x.get_device())
        if domain_z_hat.is_cuda:
            all_zeros = all_zeros.to(domain_z_hat.get_device())

        # print('dom_dis loss_gan_d domain_x', domain_x.shape, 'all_ones', all_ones.shape)
        # print('dom_dis loss_gan_d domain_z_hat', domain_z_hat.shape, 'domain_code', all_zeros.shape)
        
        real_loss = nn.functional.binary_cross_entropy_with_logits(domain_x, all_ones)
        fake_loss = nn.functional.binary_cross_entropy_with_logits(domain_z_hat, all_zeros)

        loss_gan_d = real_loss + fake_loss
        # print('loss_gan_d', loss_gan_d)
        acc.add('loss_gan_d', float(loss_gan_d.detach().cpu()), count=len(x))

        # # segmentation loss
        # # x_i_seg = self.model.forward_dec(skip_connections_x_i, content_x_i)
        # x_i_seg = self.model.unet(x_i)
        # x_j_seg = self.model.unet(x_j)
        # loss_seg = loss_f(x_i_seg, y_i) + loss_f(x_j_seg, y_j)
        # loss_seg /= 2
        # loss_seg = config['lambda_seg']*loss_seg
        # acc.add('loss_seg', float(loss_seg.detach().cpu()), count=len(x_i))
        
        loss_dis_seg = config['lambda_c_adv']*loss_c_adv_d + config['lambda_gan']*loss_gan_d # + loss_seg
        loss_dis_seg.backward()
        
        self.model.step_optim_dis_seg()

        return loss_dis_seg, acc
 
    def vae_loss(self, recons, input, mu, log_var, kld_weight=5e-3):
        """
        Computes the VAE loss function.
        Source: https://github.com/AntixK/PyTorch-VAE/blob/20c4dfa73dfc36f42970ccc334a42f37ffe08dcc/models/vanilla_vae.py
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        # Source: https://github.com/AntixK/PyTorch-VAE/blob/20c4dfa73dfc36f42970ccc334a42f37ffe08dcc/tests/test_vae.py

        # kld_weight = 0.005 # kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(- 0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def multi_class_cross_entropy_with_softmax(self, prediction, target):
        softmax = nn.Softmax(dim=1)
        return (-(target * torch.log(softmax(prediction).clamp(min=1e-08, max=1. - 1e-08))).sum(dim=-1)).mean()

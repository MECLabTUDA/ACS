import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from src.continual_learning.oracles.oracle import Oracle
from src.utils.load_restore import join_path, pkl_load, pkl_dump
from src.utils.introspection import get_class

import sys

def log_prob(x, mu, sigma_inv, eigvals):
    # Evaluate a multivariate gaussian PDF with mean mu and covariance matrix sigma
    # i.e. P(x | domain)
    # = N(x, mu, sigma)
    # Calculate density in log space because the eigenvalues are too small and the determinant would be 0 otherwise.
    # @ is for matrix multiplication
    
    # this is also a degenerate case where some of the eigenvalues of sigma are 0
    # so we take the pseudo-inverse and the product of the eigenvals greater than 0
    log_inv_det=-np.sum(np.log(2*np.pi*eigvals[eigvals>1e-5]))
    log_exponent=-(1/2)*(x-mu).T@sigma_inv@(x-mu)
    return log_inv_det+log_exponent
import sys
class DensityOracle(Oracle):
    def __init__(self, train_datasets, exp_paths, batch_size, feature_model_name='AlexNet'):
        super().__init__(train_datasets=train_datasets, exp_paths=exp_paths, batch_size=1, lowest_score=False, name='DensityOracle_'+feature_model_name)
        self.feature_model_name = feature_model_name
        self.feature_model = self.get_feature_extractor(feature_model_name)
        saved_params = pkl_load(path=self.exp_paths['obj'], name='densities_'+feature_model_name)
        if saved_params is None:
            print('SAVED PARAMS IS NONE')
            sys.exit()

            for ds in self.train_datasets:
                ds.set_tranform(transform=self.feature_model_name)
            dataloaders = [torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False) for ds in self.train_datasets]
            print('Getting initial features')
            train_features = [self.get_features(dataloader) for dataloader in dataloaders]
            print('Calculating initial values for estimation')
            self.mus, self.sigmas, self.sigma_eigenvals, self.sigma_invs, self.biases = self.get_density_information(train_features)
            pkl_dump((self.mus, self.sigmas, self.sigma_eigenvals, self.sigma_invs, self.biases), path=self.exp_paths['obj'], name='densities_'+feature_model_name)
        else:
            self.mus, self.sigmas, self.sigma_eigenvals, self.sigma_invs, self.biases = saved_params

    def get_dataloader(self, dataset):
        dataset.set_tranform(transform=self.feature_model_name)
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    def get_scores(self, dataloader):
        features = self.get_features(dataloader)
        scores = [[] for task_ix in range(self.nr_tasks)]
        for i in range(len(features)):
            x = features[i]
            domains_score = self.predict_domain(x) # Returns one score per domain
            for model_task_ix in range(self.nr_tasks):
                scores[model_task_ix].append(float(domains_score[model_task_ix]))
        return scores

    def get_feature_extractor(self, feature_model_name='AlexNet'):
        # Fetch pretrained model
        if feature_model_name == 'AlexNet':  # input_size = 224
            feature_extractor = models.alexnet(pretrained=True).cuda()
        elif feature_model_name == 'AutoEncoder':
            autoencoder_path = 'src.models.autoencoding.pretrained_autoencoder.PretrainedAutoencoder'
            autoencoder = get_class(autoencoder_path)(config={'feature_model_name': 'AlexNet'})
            autoencoder.cuda()
            agent_name = 'AutoencoderOracle_PretrainedAutoencoder_AlexNet_Task_0'
            agent = get_class('src.agents.autoencoder_agent.AutoencoderAgent')(model=autoencoder, config={'tracking_interval': 5}, exp_paths=self.exp_paths, agent_name=agent_name) 
            restored = agent.restore_state(epoch=49, eval=True)
            assert restored
            feature_extractor = agent.model
        return feature_extractor

    def get_features(self, dataloader):
        #print('Getting {} features'.format(len(dataloader)))
        task_x_features = []
        for x, y in dataloader:
            if self.feature_model_name == 'AlexNet':
                x = self.feature_model.features(x.cuda())
                x = self.feature_model.avgpool(x)
                x = torch.flatten(x, 1)
            elif self.feature_model_name == 'AutoEncoder':
                x = self.feature_model.preprocess_input(x.cuda())
                x = self.feature_model.encode(x)
            features = x.detach().cpu().numpy().reshape((1, -1))
            #print('Features shape: {}'.format(features.shape))
            #print(features.shape) torch.Size([1, 16, 291, 291])
            task_x_features.append(features)
        #print('All features shape: {}'.format(np.concatenate(task_x_features).shape))
        #print(np.concatenate(task_x_features).shape) == (len(dataloader), feature_size)
        return np.concatenate(task_x_features)

    def get_density_information(self, x_features):
        mus=[]
        for i in range(self.nr_tasks):
            mus.append(np.mean(x_features[i],axis=0))
        #assert mus[0].shape == (1024,)

        sigmas=[]
        for i in range(self.nr_tasks):
            sigmas.append(np.cov(x_features[i],rowvar=0))
        #assert sigmas[0].shape == (1024, 1024)

        print('sigma_eigenvals')
        sigma_eigenvals=[]
        for i in range(self.nr_tasks):
            sigma_eigenvals.append(np.linalg.eigvals(sigmas[i]).real)

        print('sigma_invs')
        sigma_invs=[]
        for i in range(self.nr_tasks):
            sigma_invs.append(np.linalg.pinv(sigmas[i]))

        print('biases')
        biases=[]
        for k in range(self.nr_tasks):
            biases.append(log_prob(mus[k], mus[k], sigma_invs[k], sigma_eigenvals[k]))

        return mus, sigmas, sigma_eigenvals, sigma_invs, biases

    def predict_domain(self, x):
        # predict domain by taking P(x | domain), 
        # i.e. N(x | mu_domain, sigma_domain) / N(mu_domain | mu_domain, sigma_domain)
        # = log(N(x | mu_domain, sigma_domain)) - log(N(mu_domain | mu_domain, sigma_domain))
        
        # the bias term is strictly speaking not correct from a probability standpoint, 
        # but helps empirically, because the densities are degenerate.
        
        densities=[]
        for k in range(self.nr_tasks):
            densities.append(log_prob(x, self.mus[k], self.sigma_invs[k], self.sigma_eigenvals[k])) #- self.biases[k])
            #densities.append(-np.sqrt(np.mean(np.abs(x-mus[k]))))
        densities=np.array(densities)
        return densities
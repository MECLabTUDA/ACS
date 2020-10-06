import os
import torch
import torch.nn as nn
 
from src.continual_learning.oracles.oracle import Oracle
from src.utils.introspection import get_class
from torch.optim import lr_scheduler

class AutoencoderOracle(Oracle):

    def __init__(self, train_datasets, exp_paths, batch_size, feature_model_name='AlexNet', autoencoder_path='src.models.autoencoding.pretrained_autoencoder.PretrainedAutoencoder', nr_training_epochs = 50):
        autoencoder_name = autoencoder_path.split('.')[-1]
        super().__init__(train_datasets=train_datasets, exp_paths=exp_paths, batch_size=1, lowest_score=True, name='AutoencoderOracle_'+autoencoder_name+'_'+feature_model_name)
        
        self.autoencoder_path = autoencoder_path
        self.feature_model_name = feature_model_name
        self.nr_training_epochs = nr_training_epochs

        for task_ix, task_train_dataset in enumerate(train_datasets):
            task_train_dataset.set_tranform(transform=self.feature_model_name)
            # Here, the actual batch size can be used
            dataloader = torch.utils.data.DataLoader(task_train_dataset, batch_size=batch_size, shuffle=False)
            dataloaders = {'train': dataloader, 'val': dataloader}
            #self.train_or_load_agent(dataloaders, task_ix)

    def train_or_load_agent(self, dataloaders, task_ix):
        restored, agent = self.load_agent(task_ix)
        if not restored:
            print('Training autoencoder for task {}'.format(task_ix))
            agent.train_model(dataloaders=dataloaders, nr_epochs=self.nr_training_epochs)

    def load_agent(self, task_ix):
        autoencoder = get_class(self.autoencoder_path)(config={'feature_model_name': self.feature_model_name})
        autoencoder.cuda()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)
        criterion = nn.MSELoss()
        agent_name = self.name+'_Task_'+str(task_ix)
        agent = get_class('src.agents.autoencoder_agent.AutoencoderAgent')(model=autoencoder, config={'tracking_interval': self.nr_training_epochs//5}, exp_paths=self.exp_paths, scheduler=scheduler, optimizer=optimizer, criterion=criterion, agent_name=agent_name) 
        restored = agent.restore_state(epoch=self.nr_training_epochs, eval=True)
        return restored, agent
    
    def get_dataloader(self, dataset):
        dataset.set_tranform(transform=self.feature_model_name)
        return torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)

    def get_scores(self, dataloader):
        scores = [[] for task_ix in range(self.nr_tasks)]
        for model_task_ix in range(self.nr_tasks):
            restored, agent = self.load_agent(task_ix=model_task_ix)
            assert restored
            for data_batch in dataloader:
                _, targets, outputs = agent.get_input_target_output(data_batch)
                loss = agent.calculate_criterion(outputs, targets)
                scores[model_task_ix].append(float(loss.cpu().detach().numpy()))
        return scores
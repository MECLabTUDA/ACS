# ------------------------------------------------------------------------------
# An Agent is an extension of a model. It also includes the logic to train the 
# model. This superclass diverges slightly from mp.agents.agent.Agent.
# ------------------------------------------------------------------------------

import os
import shutil
from mp.eval.accumulator import Accumulator
from mp.utils.load_restore import pkl_dump, pkl_load
from mp.utils.pytorch.pytorch_load_restore import save_model_state, load_model_state, save_optimizer_state, load_optimizer_state
from mp.eval.inference.predict import arg_max
from mp.eval.evaluate import ds_losses_metrics
import sys

class Agent:

    def __init__(self, model, label_names=None, metrics=[], device='cuda:0', 
        scores_label_weights=None, verbose=True):
        self.model = model
        self.device = device
        self.metrics = metrics
        self.label_names = label_names
        self.nr_labels = len(label_names) if self.label_names else 0
        self.scores_label_weights = scores_label_weights
        self.verbose = verbose
        self.agent_state_dict = dict()

    def get_inputs_targets(self, data):
        inputs, targets = data
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs = self.model.preprocess_input(inputs)       
        return inputs, targets.float()

    def get_outputs(self, inputs):
        return self.model(inputs)

    def predict_from_outputs(self, outputs):
        return arg_max(outputs, channel_dim=1)

    def predict(self, inputs):
        outputs = self.get_outputs(inputs)
        return self.predict_from_outputs(outputs)

    def perform_training_epoch(self, optimizer, loss_f, train_dataloader, 
        print_run_loss=False):
        acc = Accumulator('loss')
        for i, data in enumerate(train_dataloader):
            # Get data
            inputs, targets = self.get_inputs_targets(data)

            # Forward pass
            outputs = self.get_outputs(inputs)

            # Optimization step
            optimizer.zero_grad()
            loss = loss_f(outputs, targets)
            loss.backward()
            acc.add('loss', float(loss.detach().cpu()), count=len(inputs))
            optimizer.step()

        if print_run_loss:
            print('\nRunning loss: {}'.format(acc.mean('loss')))

    def train(self, results, optimizer, loss_f, train_dataloader,
        init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
        eval_datasets=dict(), eval_interval=10, 
        save_path=None, save_interval=10):

        if init_epoch == 0:
            self.track_metrics(init_epoch, results, loss_f, eval_datasets)
        for epoch in range(init_epoch, init_epoch+nr_epochs):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose
            self.perform_training_epoch(optimizer, loss_f, train_dataloader, 
                print_run_loss=print_run_loss)
        
            # Track statistics in results
            if (epoch + 1) % eval_interval == 0:
                self.track_metrics(epoch + 1, results, loss_f, eval_datasets)

            # Save agent and optimizer state
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1), optimizer)

    def track_metrics(self, epoch, results, loss_f, datasets):
        """Losses and scores are calculated for each 3D subject, and averaged 
        over the dataset.
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

    def save_state(self, states_path, state_name, optimizer=None, overwrite=False):
        """
        Saves an agent state. Raises an error if the directory exists and 
        overwrite=False.
        """
        if states_path is not None:
            state_full_path = os.path.join(states_path, state_name)
            if os.path.exists(state_full_path):
                if not overwrite:
                    raise FileExistsError
                shutil.rmtree(state_full_path)
            os.makedirs(state_full_path)
            save_model_state(self.model, 'model', state_full_path)
            pkl_dump(self.agent_state_dict, 'agent_state_dict', state_full_path)
            if optimizer is not None:
                save_optimizer_state(optimizer, 'optimizer', state_full_path)


    def restore_state(self, states_path, state_name, optimizer=None):
        """
        Tries to restore a previous agent state, consisting of a model 
        state and the content of agent_state_dict. Returns whether the restore 
        operation  was successful.
        """
        state_full_path = os.path.join(states_path, state_name)
        try:
            correct_load = load_model_state(self.model, 'model', state_full_path)
            assert correct_load
            agent_state_dict = pkl_load('agent_state_dict', state_full_path)
            assert agent_state_dict is not None
            self.agent_state_dict = agent_state_dict
            if optimizer is not None: 
                load_optimizer_state(optimizer, 'optimizer', state_full_path)
            if self.verbose:
                print('State {} was restored'.format(state_name))
            return True
        except:
            print('State {} could not be restored'.format(state_name))
            return False
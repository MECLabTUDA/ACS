import torch
import os
from mp.utils.load_restore import pkl_load, join_path, pkl_dump
from mp.visualization.confusion_matrix import ConfusionMatrix

class Oracle():
    def __init__(self, train_datasets, exp_paths, batch_size=1, lowest_score=False, name=None):
        """
        param batch_size: batch size of dataloaders
        param name: name of the oracle (name of subclass)
        param lowest_score: select the task id for which the score is lowest?
            Otherwise that with highest score is selected.
        """
        ''' 
        experiment_path: a path to some folder where intermediate values can be stored
        '''
        self.name = name
        self.lowest_score = lowest_score
        self.exp_paths = exp_paths

        self.batch_size = batch_size # OLD
        self.train_datasets = train_datasets # OLD
        self.nr_tasks = len(train_datasets) # OLD
        
        # Initialize scores dictionary. The keys are instance names and the 
        # values are model_name -> score dictionaries.
        self.scores = dict()
        self.load_scores() # Load scores which have been set

    def get_dataloader(self, dataset): #OLD
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)
        
    def accuracy(self, prediction, target): #OLD
        assert len(prediction) == len(target)
        correct = sum([1 if prediction[i]==target[i] else 0 for i in range(len(prediction))])
        return float(correct)/len(prediction)

    def class_tp_tn_fp_fn(self, prediction, target, label=1): #OLD
        targets_label = [1 if x==label else 0 for x in target]
        predictions_label = [1 if x==label else 0 for x in prediction]
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(targets_label)):
            if targets_label[i] == 1:
                if predictions_label[i] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if predictions_label[i] == 1:
                    fp += 1
                else:
                    tn += 1
        return tp, tn, fp, fn

    def select_model(self, dataset=None, dataset_id=None, per_dataset=False): #OLD
        """
        :param per_dataset: one model is selected for the entire dataset
        :returns: one model selection per 
        """
        if not dataset_id or dataset_id not in self.scores:
            dl = self.get_dataloader(dataset)
            scores = self.get_scores(dl)
            if dataset_id:
                self.scores[dataset_id] = scores
                self.save_scores()
        else:
            scores = self.scores[dataset_id]
        selected_models = []
        for example_ix in range(len(scores[0])):
            models_score = [scores[model_task][example_ix] for model_task in range(self.nr_tasks)]
            if self.lowest_score:
                selected_model = models_score.index(min(models_score))
            else:
                selected_model = models_score.index(max(models_score))
            selected_models.append(selected_model)
        if per_dataset:
            selected_model = max(selected_models, key=lambda x: selected_models.count(x))
            return selected_model
        return selected_models

    def get_scores(self, dataloader):
        """Template method for getting scores. Returns one score value per 
        model per dataloader item."""
        raise NotImplementedError

    def save_scores(self):
        full_path = join_path([self.exp_paths['obj'], 'oracles'])
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        pkl_dump(self.scores, 'scores_'+self.name, path=full_path)

    def load_scores(self):
        try:
            full_path = join_path([self.exp_paths['obj'], 'oracles'])
            scores = pkl_load('scores_'+self.name, path=full_path)
            if scores is not None:
                self.scores = scores
        except:
            pass

    def get_domain_confusion(self, lst_dataset_id_gt): #OLD
        """
        A list of (id, correct model) or (id, lst(correct model)) tuples
        """
        cm = ConfusionMatrix(self.nr_tasks)
        for dataset_id, gt in lst_dataset_id_gt:
            if isinstance(gt, list):
                selected_models = self.select_model(dataset=None, dataset_id=dataset_id, per_dataset=False)
                assert len(selected_models) == len(gt)
                for ix in range(len(gt)):
                    cm.add(predicted=selected_models[ix], actual=gt[ix])
            else:
                selected_model = self.select_model(dataset=None, dataset_id=dataset_id, per_dataset=True)
                cm.add(predicted=selected_model, actual=gt)
        return cm



        

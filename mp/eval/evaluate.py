# ------------------------------------------------------------------------------
# Functions to calculate metrics and losses for subject dataloaders and datasets. 
# The differences lie in that dataloaders may transform (e.g. resize) the 
# targets in a way that affects the result.
# ------------------------------------------------------------------------------

from mp.eval.accumulator import Accumulator
from mp.eval.metrics.metrics_segmentation import get_mean_scores
from mp.data.pytorch.transformation import per_label_channel

def dl_losses(dl, agent, loss_f):
    acc = Accumulator()
    for data in dl:
        inputs, targets = agent.get_inputs_targets(data)
        outputs = agent.get_outputs(inputs)
        # Calculate losses
        loss_dict = loss_f.get_evaluation_dict(outputs, targets)
        # Add to the accumulator   
        for key, value in loss_dict.items():         
            acc.add('Loss_'+key, value, count=len(inputs))
    return acc

def dl_metrics(dl, agent, metrics):
    acc = Accumulator()
    for data in dl:
        inputs, targets = agent.get_inputs_targets(data)
        one_channeled_target = agent.predict_from_outputs(targets)
        outputs = agent.get_outputs(inputs)
        pred = agent.predict_from_outputs(outputs)
        # Calculate metrics
        scores_dict = get_mean_scores(one_channeled_target, pred, metrics=metrics, 
                    label_names=agent.label_names, 
                    label_weights=agent.scores_label_weights)
        # Add to the accumulator      
        for key, value in scores_dict.items():         
            acc.add(key, value, count=len(inputs))
    return acc

import numpy as np
def ds_losses(ds, agent, loss_f):
    """
    returns: {loss -> {subject_name -> value}}}, with 2 additional entries per 
    loss for 'mean' and 'std'. Note that the metric is calculated per dataloader
    pero dataset. So, for instance, the scores for slices in a 2D dataloader are
    averaged.
    """
    eval_dict = dict()
    acc = Accumulator()
    for instance_ix, instance in enumerate(ds.instances):
        subject_name = instance.name
        dl = ds.get_subject_dataloader(instance_ix)
        subject_acc = dl_losses(dl, agent, loss_f)
        # Add to the accumulator and eval_dict
        for loss_key in subject_acc.get_keys():
            value = subject_acc.mean(loss_key)
            acc.add(loss_key, value, count=1)
            if loss_key not in eval_dict:
                eval_dict[loss_key] = dict()
            eval_dict[loss_key][subject_name] = value
    # Add mean and std values to the eval_dict
    for loss_key in acc.get_keys():
        eval_dict[loss_key]['mean'] = acc.mean(loss_key)
        eval_dict[loss_key]['std'] = acc.std(loss_key)
    return eval_dict

def ds_metrics(ds, agent, metrics):
    """
    returns: {metric -> {subject_name -> value}}}, with 2 additional entries per 
    metric for 'mean' and 'std'.
    """
    eval_dict = dict()
    acc = Accumulator()
    for instance_ix, instance in enumerate(ds.instances):
        subject_name = instance.name
        target = instance.y.tensor.to(agent.device)
        pred = ds.predictor.get_subject_prediction(agent, instance_ix)
        # Calculate metrics
        scores_dict = get_mean_scores(target, pred, metrics=metrics, 
                    label_names=agent.label_names, 
                    label_weights=agent.scores_label_weights)
        # Add to the accumulator and eval_dict   
        for metric_key, value in scores_dict.items():         
            acc.add(metric_key, value, count=1)
            if metric_key not in eval_dict:
                eval_dict[metric_key] = dict()
            eval_dict[metric_key][subject_name] = value
    # Add mean and std values to the eval_dict
    for metric_key in acc.get_keys():
        eval_dict[metric_key]['mean'] = acc.mean(metric_key)
        eval_dict[metric_key]['std'] = acc.std(metric_key)
    return eval_dict

def ds_losses_metrics(ds, agent, loss_f, metrics):
    eval_dict = ds_losses(ds, agent, loss_f)
    if metrics:
        eval_dict.update(ds_metrics(ds, agent, metrics))
    return eval_dict
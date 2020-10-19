# ------------------------------------------------------------------------------
# Collection of metrics to compare whole 1-channel segmentation masks.
# Metrics receive two 1-channel integer arrays.
# ------------------------------------------------------------------------------

import torch
import mp.eval.metrics.scores as score_defs

def get_tp_tn_fn_fp_segmentation(target, pred, class_ix=1):
    r"""Get TP, TN, FN and FP pixel values for segmentation."""
    assert target.shape + pred.shape
    device, shape = target.device, target.shape
    zeros = torch.zeros(shape).to(device)
    ones = torch.ones(shape).to(device)
    target_class = torch.where(target==class_ix,ones,zeros)
    pred_class = torch.where(pred==class_ix,ones,zeros)
    tp = torch.where(target_class==1,pred_class,zeros).sum()
    tn = torch.where(target_class==0,1-pred_class,zeros).sum()
    fn = torch.where(target_class==1,1-pred_class,zeros).sum()
    fp = torch.where(pred_class==1,1-target_class,zeros).sum()
    tp, tn, fn, fp = int(tp), int(tn), int(fn), int(fp)
    #assert int(ones.sum()) == tp+tn+fn+fp
    return tp, tn, fn, fp

def get_mean_scores(target, pred, metrics=['ScoreDice', 'ScoreIoU'], 
    label_names=['background', 'class 1'], label_weights=None,
    segmentation=True):
    r"""Returns the scores per label, as well as the (weighted) mean, such as
    to avoid considering "don't care" classes. The weights don't have to be 
    normalized.
    """
    scores = {metric: dict() for metric in metrics}
    # Calculate metric values per each class
    metrics = {metric: getattr(score_defs, metric)() for metric in metrics}
    for label_nr, label_name in enumerate(label_names):
        # TODO: enable also for classification
        tp, tn, fn, fp = get_tp_tn_fn_fp_segmentation(target, pred, class_ix=label_nr)
        for metric_key, metric_f in metrics.items():
            score = metric_f.eval(tp, tn, fn, fp)
            scores[metric_key+'['+label_name+']'] = score
            scores[metric_key][label_name] = score
    # Calculate metric means
    if label_weights is None:
        label_weights = {label_name: 1 for label_name in label_names}
    for metric_key in metrics.keys():
        # Replace the dictionary by the mean
        mean = sum([
            label_score*label_weights[label_name] for label_name, label_score 
            in scores[metric_key].items()]) /sum(list(label_weights.values()))
        scores[metric_key] = mean
    return scores
# ------------------------------------------------------------------------------
# Collection of loss metrics to compare whole 1-channel segmentation masks.
# Metrics receive two 1-channel integer arrays.
# ------------------------------------------------------------------------------

import torch

def get_tp_tn_fn_fp(target, pred, class_ix=1):
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
    assert int(ones.sum()) == tp+tn+fn+fp
    return tp, tn, fn, fp

class ScoreAbstract:
    def __init__(self):
        self.name = self.__class__.__name__

    def eval(self, tp, tn, fn, fp):
        raise NotImplementedError

class ScoreDice(ScoreAbstract):
    def eval(tp, tn, fn, fp):
        if tp == 0:
            if fn+fp > 0:
                return 0.
            else:
                return 1.
        return (2*tp)/(2*tp+fp+fn)

class ScoreIoU(ScoreAbstract):
    def eval(tp, tn, fn, fp):
        if tp == 0:
            if fn+fp > 0:
                return 0.
            else:
                return 1.
        return tp/(tp+fp+fn)

class ScorePrecision(ScoreAbstract):
    def eval(tp, tn, fn, fp):
        if tp == 0:
            if fp > 0:
                return 0.
            else:
                return 1.
        return tp/(tp+fp)

class ScorePPV(ScorePrecision):
    pass

class ScoreRecall(ScoreAbstract):
    def eval(tp, tn, fn, fp):
        if tp == 0:
            if fp > 0:
                return 0.
            else:
                return 1.
        return tp/(tp+fn)

class ScoreSensitivity(ScoreRecall):
    pass

class ScoreTPR(ScoreRecall):
    pass

class ScoreSpecificity(ScoreAbstract):
    def eval(tp, tn, fn, fp):
        if tn == 0:
            if fp > 0:
                return 0.
            else:
                return 1.
        return tn/(tn+fp)

class ScoreTNR(ScoreSpecificity):
    pass

def get_mean_scores(target, pred, metrics=['ScoreDice', 'ScoreIoU'], 
    label_names=['background', 'class 1'], label_weights=None):
    """
    Returns the scores per label, as well as the (weighted) mean, for instance
    to avoid considering "don't care" classes. The weights don't have to be 
    normalized.
    """
    scores = {metric: dict() for metric in metrics}
    # Calculate metric values per each class
    metrics = {metric: globals()[metric] for metric in metrics}
    for label_nr, label_name in enumerate(label_names):
        tp, tn, fn, fp = get_tp_tn_fn_fp(target, pred, class_ix=label_nr)
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
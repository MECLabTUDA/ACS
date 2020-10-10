import torch
from mp.eval.metrics.mean_scores import get_tp_tn_fn_fp_segmentation, get_mean_scores

A_1=[[0,0,0,0,0,0,0],
    [0,1,3,3,0,1,0],
    [0,0,3,1,1,2,2],
    [0,0,0,1,1,2,2]]
A_2=[[0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,3,3,3],
    [2,2,1,1,1,1,0]]
A_3=[[0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0],
    [2,0,1,1,0,0,0],
    [2,0,0,0,0,0,0]]
A_4=[[1,1,1,0,0,0,0],
    [0,0,0,0,2,2,0],
    [0,2,0,0,2,2,0],
    [0,2,0,0,3,3,3]]

B_1=[[0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,3,1,1,2,2],
    [0,0,0,1,1,2,2]]
B_2=[[0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [2,2,2,2,1,1,0]]
B_3=[[0,0,0,0,0,1,0],
    [0,0,2,0,1,1,0],
    [0,0,2,0,0,3,0],
    [0,0,0,0,3,3,0]]
B_4=[[0,0,1,0,0,0,0],
    [1,1,0,0,0,0,0],
    [0,2,0,0,0,0,0],
    [0,2,0,0,0,0,3]]

a = torch.tensor([A_1, A_2, A_3, A_4])
b = torch.tensor([B_1, B_2, B_3, B_4])

# Batched-inputs
a_batch = torch.stack([a, a, a])
b_batch = torch.stack([b, b, b])

# Single-instance inputs
a = a.unsqueeze(0)
b = b.unsqueeze(0)

def test_tp_tn_fn_fp():
    assert get_tp_tn_fn_fp_segmentation(a, b, 0) == (64, 20, 9, 19)
    assert get_tp_tn_fn_fp_segmentation(a, b, 1) == (7, 91, 9, 5)
    assert get_tp_tn_fn_fp_segmentation(a, b, 2) == (8, 94, 6, 4)
    assert get_tp_tn_fn_fp_segmentation(a, b, 3) == (2, 100, 7, 3)

def test_dice_iou():
    scores = get_mean_scores(a, b, metrics=['ScoreDice', 'ScoreIoU'], label_names=['0', '1', '2', '3'])
    target_scores = {'ScoreDice': 0.555, 'ScoreDice[0]': 0.820, 'ScoreDice[1]': 0.5, 'ScoreDice[2]': 0.615, 'ScoreDice[3]': 0.286, 'ScoreIoU': 0.410, 'ScoreIoU[0]': 0.696, 'ScoreIoU[1]': 0.333, 'ScoreIoU[2]': 0.444, 'ScoreIoU[3]': 0.167}
    for key, value in target_scores.items():
        assert abs(value - scores[key]) <= 0.01

def test_weighted_metrics():
    scores = get_mean_scores(a, b, metrics=['ScoreDice', 'ScoreIoU'], label_names=['0', '1', '2', '3'], label_weights={'0':2, '1':1, '2':0, '3':1})
    target_scores = {'ScoreDice': 0.607, 'ScoreDice[0]': 0.820, 'ScoreDice[1]': 0.5, 'ScoreDice[2]': 0.615, 'ScoreDice[3]': 0.286, 'ScoreIoU': 0.473, 'ScoreIoU[0]': 0.696, 'ScoreIoU[1]': 0.333, 'ScoreIoU[2]': 0.444, 'ScoreIoU[3]': 0.167}
    for key, value in target_scores.items():
        assert abs(value - scores[key]) <= 0.01
    
    scores = get_mean_scores(a, b, metrics=['ScoreDice', 'ScoreIoU'], label_names=['0', '1', '2', '3'], label_weights={'0':0.2, '1':0.1, '2':0, '3':0.1})
    target_scores = {'ScoreDice': 0.607, 'ScoreDice[0]': 0.820, 'ScoreDice[1]': 0.5, 'ScoreDice[2]': 0.615, 'ScoreDice[3]': 0.286, 'ScoreIoU': 0.473, 'ScoreIoU[0]': 0.696, 'ScoreIoU[1]': 0.333, 'ScoreIoU[2]': 0.444, 'ScoreIoU[3]': 0.167}
    for key, value in target_scores.items():
        assert abs(value - scores[key]) <= 0.01

def test_batched_tp_tn_fn_fp():
    assert get_tp_tn_fn_fp_segmentation(a_batch, b_batch, 0) == (64*3, 20*3, 9*3, 19*3)
    assert get_tp_tn_fn_fp_segmentation(a_batch, b_batch, 1) == (7*3, 91*3, 9*3, 5*3)
    assert get_tp_tn_fn_fp_segmentation(a_batch, b_batch, 2) == (8*3, 94*3, 6*3, 4*3)
    assert get_tp_tn_fn_fp_segmentation(a_batch, b_batch, 3) == (2*3, 100*3, 7*3, 3*3)

def test_batched_metrics():
    scores = get_mean_scores(a_batch, b_batch, metrics=['ScoreDice', 'ScoreIoU'], label_names=['0', '1', '2', '3'], label_weights={'0':2, '1':1, '2':0, '3':1})
    target_scores = {'ScoreDice': 0.607, 'ScoreDice[0]': 0.820, 'ScoreDice[1]': 0.5, 'ScoreDice[2]': 0.615, 'ScoreDice[3]': 0.286, 'ScoreIoU': 0.473, 'ScoreIoU[0]': 0.696, 'ScoreIoU[1]': 0.333, 'ScoreIoU[2]': 0.444, 'ScoreIoU[3]': 0.167}
    for key, value in target_scores.items():
        assert abs(value - scores[key]) <= 0.01

    scores = get_mean_scores(a_batch, b_batch, metrics=['ScoreDice', 'ScoreIoU'], label_names=['0', '1', '2', '3'], label_weights={'0':2, '1':1, '2':0, '3':1})
    target_scores = {'ScoreDice': 0.607, 'ScoreDice[0]': 0.820, 'ScoreDice[1]': 0.5, 'ScoreDice[2]': 0.615, 'ScoreDice[3]': 0.286, 'ScoreIoU': 0.473, 'ScoreIoU[0]': 0.696, 'ScoreIoU[1]': 0.333, 'ScoreIoU[2]': 0.444, 'ScoreIoU[3]': 0.167}
    for key, value in target_scores.items():
        assert abs(value - scores[key]) <= 0.01
    
    scores = get_mean_scores(a_batch, b_batch, metrics=['ScoreDice', 'ScoreIoU'], label_names=['0', '1', '2', '3'], label_weights={'0':0.2, '1':0.1, '2':0, '3':0.1})
    target_scores = {'ScoreDice': 0.607, 'ScoreDice[0]': 0.820, 'ScoreDice[1]': 0.5, 'ScoreDice[2]': 0.615, 'ScoreDice[3]': 0.286, 'ScoreIoU': 0.473, 'ScoreIoU[0]': 0.696, 'ScoreIoU[1]': 0.333, 'ScoreIoU[2]': 0.444, 'ScoreIoU[3]': 0.167}
    for key, value in target_scores.items():
        assert abs(value - scores[key]) <= 0.01
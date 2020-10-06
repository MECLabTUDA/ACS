import torch
from mp.eval.losses.losses_segmentation import LossBCE, LossDice, LossClassWeighted, LossDiceBCE
from mp.data.pytorch.transformation import per_label_channel

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

# Build test inputs
a = torch.tensor([A_1, A_2, A_3, A_4])
b = torch.tensor([B_1, B_2, B_3, B_4])
c = torch.tensor([A_4, B_3, A_1, B_2])
a = per_label_channel(a.unsqueeze(0), nr_labels=4, channel_dim=0)
b = per_label_channel(b.unsqueeze(0), nr_labels=4, channel_dim=0)
c = per_label_channel(c.unsqueeze(0), nr_labels=4, channel_dim=0)

# Batched-inputs
a_batch = torch.stack([a, a, a])
b_batch = torch.stack([b, b, b])
c_batch = torch.stack([c, c, c])

# Single-instance inputs
a = a.unsqueeze(0)
b = b.unsqueeze(0)
c = c.unsqueeze(0)

# Zero- and one- filled arrays
d = torch.zeros(a.shape, dtype=torch.float64)
e = torch.ones(a.shape, dtype=torch.float64)
d_batch = torch.zeros(a_batch.shape, dtype=torch.float64)
e_batch = torch.ones(a_batch.shape, dtype=torch.float64)

def test_bce():
    loss = LossBCE()
    assert float(loss(a, a)) == float(loss(b, b)) == 0
    assert float(loss(a_batch, a_batch)) == float(loss(b_batch, b_batch)) == 0
    assert float(loss(a, b)) == float(loss(a, b))
    assert abs(float(loss(a, b)) - 13.839) < 0.01
    assert abs(float(loss(a_batch, b_batch)) - 13.839) < 0.01
    assert float(loss(d, e)) == float(loss(e, d)) == 100
    assert float(loss(d_batch, e_batch)) == float(loss(e_batch, d_batch)) == 100
    assert loss(a, b) < loss(b, c) < loss(a, c) < loss(e, d)

def test_batched_dice():
    # For higher batches, the smoothed Dice loss is higher (it pproaches the 
    # actual loss better).
    loss = LossDice(smooth=.001)
    assert abs(float(loss(a, b)) - 0.2768) < 0.001
    assert abs(float(loss(a_batch, b_batch)) - 0.2768) < 0.001
    loss = LossDice(smooth=1.)
    assert abs(float(loss(a, b)) - 0.2756) < 0.001
    assert abs(float(loss(a_batch, b_batch)) - 0.2764) < 0.001

def test_weighted_dice():
    # The loss is lower the higher the smoothing factor. The smaller the 
    # smoothing factor, the more similar the result to the inverse Dice score.
    dice_loss = LossDice(smooth=1.)
    loss = LossClassWeighted(loss=dice_loss, nr_labels=4)
    assert abs(float(loss(a, b)) - 0.4245) < 0.001
    assert abs(float(loss(a, a)) - 0.0) < 0.001
    assert abs(float(loss(d, e)) - 0.9911) < 0.001
    dice_loss = LossDice(smooth=.001)
    loss = LossClassWeighted(loss=dice_loss, nr_labels=4)
    assert abs(float(loss(a, b)) - 0.4446) < 0.001
    assert abs(float(loss(a, a)) - 0.0) < 0.001
    assert abs(float(loss(d, e)) - 0.9999) < 0.001
    loss = LossClassWeighted(loss=dice_loss, weights=[2,1,0,1])
    assert abs(float(loss(a, b)) - 0.3933) < 0.001
    loss = LossClassWeighted(loss=dice_loss, weights=[0.2,0.1,.0,.1])
    assert abs(float(loss(a, b)) - 0.3933) < 0.001

def test_combined_losses():
    loss = LossDiceBCE(bce_weight=1., smooth=1.)
    assert abs(float(loss(a, b)) - 14.115) < 0.001
    loss = LossDiceBCE(bce_weight=.5, smooth=1.)
    assert abs(float(loss(a, b)) - 7.195) < 0.001
    loss = LossDiceBCE(bce_weight=.5, smooth=.001)
    assert abs(float(loss(a, b)) - 7.1964) < 0.001
    assert abs(float(loss(a_batch, b_batch)) - 7.1964) < 0.001

def test_get_evaluation_dict():
    loss = LossBCE()
    evaluation_dict = loss.get_evaluation_dict(a, b)
    assert abs(evaluation_dict['LossBCE'] - 13.839) < 0.01

    dice_loss = LossDice(smooth=.001)
    loss = LossClassWeighted(loss=dice_loss, nr_labels=4)
    evaluation_dict = loss.get_evaluation_dict(a, b)
    assert abs(evaluation_dict['LossDice[smooth=0.001][0]'] - 0.1795) < 0.01
    assert abs(evaluation_dict['LossDice[smooth=0.001][1]'] - 0.5) < 0.01
    assert abs(evaluation_dict['LossDice[smooth=0.001][2]'] - 0.3846) < 0.01
    assert abs(evaluation_dict['LossDice[smooth=0.001][3]'] - 0.7142) < 0.01
    assert abs(evaluation_dict['LossClassWeighted[loss=LossDice[smooth=0.001]; weights=(1, 1, 1, 1)]'] - 0.4446) < 0.01    
    
    dice_loss = LossDice(smooth=.001)
    loss = LossClassWeighted(loss=dice_loss, weights=[0.2,0.1,.0,.1])
    evaluation_dict = loss.get_evaluation_dict(a, b)
    assert abs(evaluation_dict['LossDice[smooth=0.001][0]'] - 0.1795) < 0.01
    assert abs(evaluation_dict['LossDice[smooth=0.001][1]'] - 0.5) < 0.01
    assert abs(evaluation_dict['LossDice[smooth=0.001][2]'] - 0.3846) < 0.01
    assert abs(evaluation_dict['LossDice[smooth=0.001][3]'] - 0.7142) < 0.01
    assert abs(evaluation_dict['LossClassWeighted[loss=LossDice[smooth=0.001]; weights=(0.2, 0.1, 0.0, 0.1)]'] - 0.3933) < 0.01

    loss = LossDiceBCE(bce_weight=.5, smooth=1.)
    evaluation_dict = loss.get_evaluation_dict(a, b)
    assert abs(evaluation_dict['LossCombined[1.0xLossDice[smooth=1.0]+0.5xLossBCE]'] - 7.195) < 0.01
    assert abs(evaluation_dict['LossBCE'] - 13.839) < 0.01
    assert abs(evaluation_dict['LossDice[smooth=1.0]'] - 0.275) < 0.01

def test_batched_weighted_dice():
    dice_loss = LossDice(smooth=.001)
    loss = LossClassWeighted(loss=dice_loss, nr_labels=4)
    assert abs(float(loss(a_batch, b_batch)) - 0.4446) < 0.001
    assert abs(float(loss(a_batch, a_batch)) - 0.0) < 0.001
    assert abs(float(loss(d_batch, e_batch)) - 0.9999) < 0.001
    loss = LossClassWeighted(loss=dice_loss, weights=[2,1,0,1])
    assert abs(float(loss(a_batch, b_batch)) - 0.3933) < 0.001
    loss = LossClassWeighted(loss=dice_loss, weights=[0.2,0.1,.0,.1])
    assert abs(float(loss(a_batch, b_batch)) - 0.3933) < 0.001
    evaluation_dict = loss.get_evaluation_dict(a_batch, b_batch)
    assert abs(evaluation_dict['LossDice[smooth=0.001][0]'] - 0.1795) < 0.01
    assert abs(evaluation_dict['LossDice[smooth=0.001][1]'] - 0.5) < 0.01
    assert abs(evaluation_dict['LossDice[smooth=0.001][2]'] - 0.3846) < 0.01
    assert abs(evaluation_dict['LossDice[smooth=0.001][3]'] - 0.7142) < 0.01
    assert abs(evaluation_dict['LossClassWeighted[loss=LossDice[smooth=0.001]; weights=(0.2, 0.1, 0.0, 0.1)]'] - 0.3933) < 0.01

def test_batched_weighted_dice_two():
    dice_loss = LossDice(smooth=.001)
    loss = LossClassWeighted(loss=dice_loss, weights=[20.,10.,.0,10.])
    assert abs(float(loss(a_batch, b_batch)) - 0.3933) < 0.001
    evaluation_dict = loss.get_evaluation_dict(a_batch, b_batch)
    assert abs(evaluation_dict['LossDice[smooth=0.001][0]'] - 0.1795) < 0.01
    assert abs(evaluation_dict['LossDice[smooth=0.001][1]'] - 0.5) < 0.01
    assert abs(evaluation_dict['LossDice[smooth=0.001][2]'] - 0.3846) < 0.01
    assert abs(evaluation_dict['LossDice[smooth=0.001][3]'] - 0.7142) < 0.01
    assert abs(evaluation_dict['LossClassWeighted[loss=LossDice[smooth=0.001]; weights=(20.0, 10.0, 0.0, 10.0)]'] - 0.3933) < 0.01
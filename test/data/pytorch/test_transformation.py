import torch
from mp.data.pytorch.transformation import per_label_channel, one_output_channel

def test_per_label_channel():
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
    a = torch.tensor([A_1, A_2, A_3, A_4])
    a = a.unsqueeze(0)
    per_label_channel_a = per_label_channel(a, nr_labels=4, channel_dim=0)
    one_output_channel_a = one_output_channel(per_label_channel_a, channel_dim=0).numpy()
    assert (a.numpy() == one_output_channel_a).all()



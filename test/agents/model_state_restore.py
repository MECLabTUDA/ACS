import torch
from mp.data.datasets.ds_mr_prostate_decathlon import DecathlonProstateT2
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.eval.losses.losses_segmentation import LossDice, LossClassWeighted
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.evaluate import ds_losses_metrics

def test_restore_model_state_and_eval():
    device = 'cpu'

    # Fetch data
    data = DecathlonProstateT2()
    label_names = data.label_names
    nr_labels = data.nr_labels

    # Transform data to PyTorch format and build train dataloader
    input_shape = (1, 256, 256)
    datasets = dict()
    datasets['train'] = PytorchSeg2DDataset(data, ix_lst=[0], size=input_shape, aug_key='none', resize=False)
    datasets['mixed'] = PytorchSeg2DDataset(data, ix_lst=[0, 1], size=input_shape, aug_key='none', resize=False)
    datasets['test'] = PytorchSeg2DDataset(data, ix_lst=[1], size=input_shape, aug_key='none', resize=False)
    dl = torch.utils.data.DataLoader(datasets['train'], batch_size=8, shuffle=False)

    # Build model
    model = UNet2D(input_shape, nr_labels)

    # Define agent
    agent = SegmentationAgent(model=model, label_names=label_names, device=device,
            metrics=['ScoreDice'])

    # Restore model state
    agent.restore_state(states_path='test/storage/agent_states_prostate_2D', state_name="epoch_300")

    # Calculate metrics and compare
    loss_g = LossDice(1e-05)
    loss_f = LossClassWeighted(loss=loss_g, weights=(1.,1.), device=device)
    eval_dict = ds_losses_metrics(datasets['mixed'], agent, loss_f, metrics=['ScoreDice'])

    test_target_dict = {'ScoreDice': {'prostate00': 0.9280484305139076, 'prostate01': 0.5375613582619043, 'mean': 0.732804894387906, 'std': 0.19524353612600165}, 
    'ScoreDice[background]': {'prostate00': 0.996721191337123, 'prostate01': 0.9785040545630738, 'mean': 0.9876126229500983, 'std': 0.009108568387024618}, 
    'ScoreDice[prostate]': {'prostate00': 0.8593756696906922, 'prostate01': 0.09661866196073488, 'mean': 0.47799716582571355, 'std': 0.3813785038649787}, 
    'Loss_LossClassWeighted[loss=LossDice[smooth=1e-05]; weights=(1.0, 1.0)]': {'prostate00': 0.10226414799690246, 'prostate01': 0.4694981321692467, 'mean': 0.2858811400830746, 'std': 0.1836169920861721}, 
    'Loss_LossDice[smooth=1e-05][0]': {'prostate00': 0.005160685380299886, 'prostate01': 0.03430714905261993, 'mean': 0.01973391721645991, 'std': 0.014573231836160022}, 
    'Loss_LossDice[smooth=1e-05][1]': {'prostate00': 0.19936761061350505, 'prostate01': 0.9046891242265701, 'mean': 0.5520283674200376, 'std': 0.3526607568065325}}

    for metric_key, metric_dict in test_target_dict.items():
        for key, value in metric_dict.items():
            assert abs(value - eval_dict[metric_key][key]) < 0.01

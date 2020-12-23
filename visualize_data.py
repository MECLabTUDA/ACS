import os
from torch.utils.data import DataLoader
import mp.visualization.visualize_imgs as vis
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset
from mp.paths import storage_path
from mp.utils.load_restore import save_json
from mp.data.datasets.ds_mr_hippocampus_decathlon import DecathlonHippocampus
from mp.data.datasets.ds_mr_hippocampus_dryad import DryadHippocampus
from mp.data.datasets.ds_mr_hippocampus_harp import HarP

for ds in [DecathlonHippocampus(merge_labels=True), DryadHippocampus(merge_labels=True), HarP(merge_labels=True)]:
    save_path = os.path.join(storage_path, 'dataset_visualizations')
    save_path = os.path.join(save_path, ds.name).replace(':', '-')
    os.makedirs(save_path)
    save_json({'name': ds.name, 'size': ds.size, 'mean_shape': ds.mean_shape}, save_path, 'ds_summary')

    # Visualize all subjects
    ds_len = len(ds.instances)
    for subject_ix, instance in enumerate(ds.instances):
        print('Subject {} of {}'.format(subject_ix+1, ds_len))
        vis.plot_3d_subject_gt(instance.get_subject(), save_path=os.path.join(save_path, 'subject_'+str(subject_ix)+'_'+instance.name+'.png'))

    input_shape = (1, 64, 64)
    resize = False
    batch_size = 10

    aug_key = 'none'
    norm_key = 'rescaling'
    py_ds = PytorchSeg2DDataset(ds, ix_lst=None, size=input_shape, norm_key=norm_key, aug_key=aug_key, resize=resize)
    dl = DataLoader(py_ds, batch_size=batch_size, shuffle=False)
    vis.visualize_dataloader_with_masks(dl, img_size=(128, 128), max_nr_imgs=100, save_path=os.path.join(save_path, 'dataloader_unshuffled.png'))
    dl = DataLoader(py_ds, batch_size=batch_size, shuffle=True)
    vis.visualize_dataloader_with_masks(dl, img_size=(128, 128), max_nr_imgs=100, save_path=os.path.join(save_path, 'dataloader_shuffled.png'))

import os
import SimpleITK as sitk
import mp.visualization.visualize_imgs as vi
import torch

def test_3d_images():
    images_path = os.path.join('test', 'test_obj')
    x = sitk.ReadImage(os.path.join(images_path, 'img_00.nii'))
    x = sitk.GetArrayFromImage(x)[0] # Take only T2-weighted
    y = sitk.ReadImage(os.path.join(images_path, 'mask_00.nii'))
    y = sitk.GetArrayFromImage(y)
    save_path = os.path.join('test', 'test_obj')
    #vi.plot_3d_img(x, save_path=os.path.join(save_path, '3dimg.png'))
    #assert os.path.isfile(os.path.join(save_path, '3dimg.png'))
    vi.plot_3d_segmentation(x, y, save_path=os.path.join(save_path, '3dsegm.png'), img_size=(128, 128))
    assert os.path.isfile(os.path.join(save_path, '3dsegm.png'))
'''
def test_2d_overlay():
    images_path = os.path.join('test', 'test_obj')
    x = sitk.ReadImage(os.path.join(images_path, 'img_00.nii'))
    x = sitk.GetArrayFromImage(x)[0] # Take only T2-weighted
    y = sitk.ReadImage(os.path.join(images_path, 'mask_00.nii'))
    y = sitk.GetArrayFromImage(y)
    save_path = os.path.join('test', 'test_obj')
    vi.plot_overlay_mask(y[9], y[10], save_path=os.path.join(save_path, 'masks.png'))
    assert os.path.isfile(os.path.join(save_path, 'masks.png'))
    torch_a = torch.from_numpy(y[9]).float()
    torch_b = torch.from_numpy(y[10]).float()
    vi.plot_overlay_mask(torch_a, torch_b, save_path=os.path.join(save_path, 'torch_masks.png'))
    assert os.path.isfile(os.path.join(save_path, 'torch_masks.png'))
'''
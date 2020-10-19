# ------------------------------------------------------------------------------
# A Predictor makes a prediction for a subject index that has the same size
# as the subject's target. It reverses the trandormation operations performed
# so that inputs can be passed through the model. It, for instance, merges
# patches and 2D slices into 3D volumes of the original size.
# ------------------------------------------------------------------------------

import copy
import torch
import torchio
import mp.data.pytorch.transformation as trans

class Predictor():
    r"""A predictor recreates a prediction with the correct dimensions from 
    model outputs. There are different predictors for different PytorchDatasets,
    and these are setted internally with the creation of a PytorchDataset.
    Args:
        instances (list[Instance]): a list of instances, as for a Dataset
        size (tuple[int]): size as (channels, width, height, Opt(depth))
        norm (torchio.transforms): a normaliztion strategy
    """
    def __init__(self, instances, size=(1, 56, 56, 10), norm=None):
        self.instances = instances
        assert len(size) > 2
        self.size = size
        self.norm = norm

    def transform_subject(self, subject):
        r"""Apply normalization strategy to subject."""
        if self.norm is not None:
            subject = self.norm(subject)
        return subject

    def get_subject(self, subject_ix):
        r"""Copy and load a TorchIO subject."""
        subject = copy.deepcopy(self.instances[subject_ix].get_subject())
        subject.load()
        subject = self.transform_subject(subject)
        return subject

    def get_subject_prediction(self, agent, subject_ix):
        r"""Get a prediction for a 3D subject."""
        raise NotImplementedError

class Predictor2D(Predictor):
    r"""The Predictor2D makes a forward pass for each 2D slice and merged these
    into a volume.
    """
    def __init__(self, *args, resize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize = resize

    def get_subject_prediction(self, agent, subject_ix):

        subject = self.get_subject(subject_ix)

        # Slides first
        x = subject.x.tensor.permute(3, 0, 1, 2)
        # Get original size
        original_size = subject['y'].data.shape
        original_size_2d = original_size[:3]

        pred = []
        with torch.no_grad():
            for slice_idx in range(len(x)):
                if self.resize:
                    inputs = trans.resize_2d(x[slice_idx], size=self.size).to(agent.device)
                    inputs = torch.unsqueeze(inputs, 0)
                    slice_pred = agent.predict(inputs).float()
                    pred.append(trans.resize_2d(slice_pred, size=original_size_2d, label=True))
                else:
                    inputs = trans.centre_crop_pad_2d(x[slice_idx], size=self.size).to(agent.device)
                    inputs = torch.unsqueeze(inputs, 0)
                    slice_pred = agent.predict(inputs).float()
                    pred.append(trans.centre_crop_pad_2d(slice_pred, size=original_size_2d))

        # Merge slices and rotate so depth last
        pred = torch.stack(pred, dim=0)
        pred = pred.permute(1, 2, 3, 0)
        assert original_size == pred.shape
        return pred

class Predictor3D(Predictor):
    r"""The Predictor3D Reconstructs an image into the original size after 
    performing a forward pass.
    """
    def __init__(self, *args, resize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize = resize

    def get_subject_prediction(self, agent, subject_ix):
        subject = self.get_subject(subject_ix)

        x = subject['x'].data
        # Get original label size
        original_size = subject['y'].data.shape

        
        if self.resize:
            # Resize to appropiate model size and make prediction
            x = trans.resize_3d(x, size=self.size).to(agent.device)
            x = torch.unsqueeze(x, 0)
            with torch.no_grad():
                pred = agent.predict(x).float()
            # Restore prediction to original size
            pred = trans.resize_3d(pred, size=original_size, label=True)

        else:
            # Crop or pad instead of interpolating
            x = trans.centre_crop_pad_3d(x, size=self.size).to(agent.device)
            x = torch.unsqueeze(x, 0)
            with torch.no_grad():
                pred = agent.predict(x).float()
            pred = trans.centre_crop_pad_3d(pred, size=original_size)
        assert original_size == pred.shape
        return pred

class GridPredictor(Predictor):
    r"""The GridPredictor deconstructs a 3D volume into patches, makes a forward 
    pass through the model and reconstructs a prediction of the output size.
    """
    def __init__(self, *args, patch_overlap = (0,0,0), **kwargs):
        super().__init__(*args, **kwargs)
        assert patch_overlap[2] == 0 # Otherwise, have gotten wrong overlap
        self.patch_overlap = patch_overlap
        self.patch_size = self.size[1:]

    def get_subject_prediction(self, agent, subject_ix):

        subject = self.get_subject(subject_ix)
        original_size = subject['y'].data.shape

        grid_sampler = torchio.inference.GridSampler(
            sample=subject,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap)

        # Make sure the correct transformations are performed before predicting
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=5)
        patch_aggregator = torchio.inference.GridAggregator(grid_sampler)
        with torch.no_grad():
            for patches_batch in patch_loader:            
                input_tensor = patches_batch['x'][torchio.DATA].to(agent.device)
                locations = patches_batch[torchio.LOCATION].to(agent.device)
                pred = agent.predict(input_tensor)
                # Add dimension for channel, which is not in final output
                pred = torch.unsqueeze(pred, 1)

                patch_aggregator.add_batch(pred, locations)
        output = patch_aggregator.get_output_tensor().to(agent.device)

        assert original_size == output.shape
        return output
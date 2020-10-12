# ------------------------------------------------------------------------------
# An autoencoding agent.
# ------------------------------------------------------------------------------

from mp.agents.agent import Agent

class AutoencodingAgent(Agent):
    r"""An Agent for autoencoder models."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_inputs_targets(self, data):
        r"""The usual dataloaders are used for autoencoders. However, these 
        ignore the target and instead treat he input as target
        """
        inputs, targets = data
        inputs = inputs.to(self.device)
        inputs = self.model.preprocess_input(inputs)
        targets = inputs.clone()
        return inputs, targets

    def predict_from_outputs(self, outputs):
        r"""No transformation is performed on the outputs, as the goal is to
        reconstruct the input. Therefore, the output should have the same dim
        as the input."""
        return outputs
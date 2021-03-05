# ------------------------------------------------------------------------------
# A standard segmentation agent, which performs softmax in the outputs.
# ------------------------------------------------------------------------------

from mp.agents.agent import Agent
from mp.eval.inference.predict import softmax

class SegmentationAgent(Agent):
    r"""An Agent for segmentation models."""
    def __init__(self, *args, **kwargs):
        if 'metrics' not in kwargs:
            kwargs['metrics'] = ['ScoreDice', 'ScoreIoU']
        super().__init__(*args, **kwargs)

    def get_outputs(self, inputs):
        r"""Applies a softmax transformation to the model outputs"""
        outputs = self.model(inputs)
        outputs = softmax(outputs).clamp(min=1e-08, max=1.-1e-08)
        return outputs
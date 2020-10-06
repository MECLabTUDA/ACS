# ------------------------------------------------------------------------------
# A standard segmentation agent.
# ------------------------------------------------------------------------------

from mp.agents.agent import Agent
from mp.eval.inference.predict import softmax

class SegmentationAgent(Agent):
    def __init__(self, *args, **kwargs):
        if 'metrics' not in kwargs:
            kwargs['metrics'] = ['ScoreDice', 'ScoreIoU']
        super().__init__(*args, **kwargs)

    def get_outputs(self, inputs):
        # Apply a softmax transformation to the model outputs
        outputs = self.model(inputs)
        outputs = softmax(outputs)
        return outputs
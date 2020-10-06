from src.continual_learning.oracles.oracle import Oracle

class LastOracle(Oracle):
    def __init__(self, train_datasets, exp_paths, batch_size):
        super().__init__(train_datasets=train_datasets, exp_paths=exp_paths, batch_size=batch_size, lowest_score=False, name='LastOracle') 

    def get_scores(self, dataloader):
        nr_examples = 0
        for x, y in dataloader:
            nr_examples += len(x)
        scores = [[1.0 if task_ix==self.nr_tasks-1 else 0.0 for x in range(nr_examples)] for task_ix in range(self.nr_tasks)]
        return scores
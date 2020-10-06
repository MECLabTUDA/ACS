from src.continual_learning.oracles.oracle import Oracle

class TaskOracle(Oracle):
    def __init__(self, train_datasets, exp_paths, batch_size):
        super().__init__(train_datasets=train_datasets, exp_paths=exp_paths, batch_size=batch_size, lowest_score=False, name='TaskOracle')

    def set_scores(self, dataset_id, gt_task_ixs):
        """
        For the task oracle, scores should be set manually.
        """
        scores = [[1.0 if gt_task_ix==model_ix else 0.0 for gt_task_ix in gt_task_ixs] for model_ix in range(self.nr_tasks)]
        self.scores[dataset_id] = scores
        self.save_scores()

    def get_scores(self, dataloader):
        raise Exception('Scores should be set manually.')
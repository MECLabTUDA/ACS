


# Dictionary of agent_name -> agent
agents = {'agent_name': None}


# For each agent and each dataset, the metrics are calculated
from mp.eval.evaluate import ds_metrics

metrics = ['ScoreDice', 'ScoreIoU']

scores = ds_metrics(ds, agent, metrics)


# For each agent and each patient, the metrics are calculated

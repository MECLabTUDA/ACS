import os

from mp.eval.result import Result
from mp.visualization.plot_results import plot_results

def test_plotting():
    res = Result(name='example_result')
    res.add(1, 'accuracy', 0.2, data='train')
    res.add(2, 'accuracy', 0.3, data='train')
    res.add(3, 'accuracy', 0.4, data='train')
    res.add(0, 'F1', 0.5, data='train')
    res.add(3, 'F1', 0.7, data='train')
    res.add(0, 'F1', 0.3, data='val')
    res.add(3, 'F1', 0.45, data='val')
    save_path = os.path.join('test', 'test_obj')
    plot_results(res, measures = ['accuracy', 'F1'], save_path=save_path, title='Test figure', ending='.png')
    assert os.path.isfile(os.path.join(save_path, 'example_result.png'))
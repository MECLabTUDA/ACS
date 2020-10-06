import os
import shutil

from mp.experiment.experiment import Experiment
from mp.eval.result import Result
from mp.utils.load_restore import load_json
from mp.paths import storage_path

def test_success():
    notes='A test experiment which is successful'
    exp = Experiment({'test_param': 2, 'nr_runs': 1}, name='TEST_SUCCESS', notes=notes)
    exp_run = exp.get_run(0)
    res = Result(name='some_result')
    res.add(1, 'A', 2.0)
    res.add(3, 'A', 10.0)
    exp_run.finish(results=res)
    path = os.path.join(os.path.join(storage_path, 'exp'), 'TEST_SUCCESS')
    exp_review = load_json(path, 'review')
    assert exp_review['notes'] == notes
    config = load_json(path, 'config')
    assert config['test_param'] == 2
    res_path = os.path.join(path, os.path.join('0', 'results'))
    assert os.path.isfile(os.path.join(res_path, 'some_result.png'))
    shutil.rmtree(path)

def test_failure():
    notes='A test experiment which fails'
    exp = Experiment({'test_param': 2, 'nr_runs': 1}, name='TEST_FAILURE', notes=notes)
    exp_run = exp.get_run(0)
    exp_run.finish(exception=Exception)
    path = os.path.join(os.path.join(storage_path, 'exp'), 'TEST_FAILURE')
    exp_review = load_json(path, 'review')
    assert exp_review['notes'] == notes
    exp_run_review = load_json(os.path.join(path, '0'), 'review')
    assert 'FAILURE' in exp_run_review['state']
    shutil.rmtree(path)

def test_reload():
    notes='A test experiment which is reloaded'
    # First experiment creation
    exp = Experiment({'test_param': 2, 'nr_runs': 1}, name='TEST_RELOAD', notes=notes)
    res = Result(name='some_result')
    # Experiment reload
    exp = Experiment(name='TEST_RELOAD', reload_exp=True)
    assert exp.review['notes'] == notes
    assert exp.config['test_param'] == 2
    path = os.path.join(os.path.join(storage_path, 'exp'), 'TEST_RELOAD')
    shutil.rmtree(path)

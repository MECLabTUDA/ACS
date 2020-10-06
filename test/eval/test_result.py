from mp.eval.result import Result

def test_results():
    res = Result(name='Example')
    res.add(1, 'accuracy', 0.2, data='example')
    res.add(2, 'accuracy', 0.3, data='example')
    res.add(3, 'accuracy', 0.4, data='example')
    res.add(0, 'F1', 0.5, data='example')
    res.add(3, 'F1', 0.7, data='example')
    assert res.get_min_epoch(metric='accuracy', data='example') == 1
    assert res.get_max_epoch(metric='F1', data='example') == 3
    assert res.get_epoch_metric(epoch=2, metric='accuracy', data='example') == 0.3
    assert len(res.to_pandas()) == 5

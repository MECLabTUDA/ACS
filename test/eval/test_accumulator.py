from mp.eval.accumulator import Accumulator

def test_acc():
    acc = Accumulator(keys=['A'])
    for i in range(5):
        acc.add('A', float(i))
    assert acc.mean('A') == 2.0
    assert 1.41 < acc.std('A') < 1.415
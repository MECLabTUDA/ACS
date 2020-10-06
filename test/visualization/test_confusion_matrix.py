import os
from mp.visualization.confusion_matrix import ConfusionMatrix

def test_confusion_matrix():
    # We have 3 classes
    cm = ConfusionMatrix(3)
    # 2 tp for each class
    cm.add(predicted=0, actual=0, count=2)
    cm.add(predicted=1, actual=1, count=2)
    cm.add(predicted=2, actual=2, count=2)
    # 3 exampels of class 0 were predicted as class 1
    cm.add(predicted=1, actual=0, count=3)
    # 1 example of class 1 was predicted as class 2
    cm.add(predicted=2, actual=1, count=1)
    save_path = os.path.join('test', 'test_obj')
    cm.plot(path=save_path, name='test_confusion_matrix' )
    assert os.path.isfile(os.path.join(save_path, 'test_confusion_matrix.png'))
    assert cm.cm == [[2, 3, 0], [0, 2, 1], [0, 0, 2]]
    assert cm.get_accuracy() == 0.6
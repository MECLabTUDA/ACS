from mp.utils.introspection import introspect
from mp.utils.load_restore import join_path

def test_introspection():
    class_path = 'mp.models.classification.small_cnn.SmallCNN'
    exp = introspect(class_path)()
    assert exp.__class__.__name__ == 'SmallCNN'
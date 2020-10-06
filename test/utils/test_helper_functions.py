import mp.utils.helper_functions as hf

def test_date_time():
    date = hf.get_time_string(True)
    assert len(date) == 21
    date = hf.get_time_string(False)
    assert len(date) == 19
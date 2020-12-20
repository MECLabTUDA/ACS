# ------------------------------------------------------------------------------
# Miscellaneous helper functions.
# ------------------------------------------------------------------------------

def f_optional_args(f, args, x):
    r"""If there are arguments, these are passed to the function."""
    if args:
        return f(x, **args)
    else:
        return f(x)

import datetime
def get_time_string(cover=False):
    r"""
    Returns the current time in the format YYYY-MM-DD_HH-MM, or
    [YYYY-MM-DD_HH-MM] if 'cover' is set to 'True'.
    """
    date = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]
    if cover:
        return '['+date+']'
    else:
        return date

import ntpath
def divide_path_fname(path):
    r"""Divide path and name from a full path."""
    path_to_file, file_name = ntpath.split(path)
    if not file_name:
        # Cease where the path ends with a slash
        file_name = ntpath.basename(path_to_file)
        path_to_file = path_to_file.split(file_name)[0]
    return path_to_file, file_name

import numpy as np
import torch
def seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
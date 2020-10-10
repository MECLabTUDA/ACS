# ------------------------------------------------------------------------------
# Functions to save and restore different data types.
# ------------------------------------------------------------------------------

import os

# PICKLE
import pickle
def pkl_dump(obj, name, path='obj'):
    """Saves an object in pickle format."""
    if '.p' not in name:
        name = name + '.pkl'
    path = os.path.join(path, name)
    pickle.dump(obj, open(path, 'wb'))

def pkl_load(name, path='obj'):
    """Restores an object from a pickle file."""
    if '.p' not in name:
        name = name + '.pkl'
    path = os.path.join(path, name)
    try:
        obj = pickle.load(open(path, 'rb'))
    except FileNotFoundError:
        obj = None
    return obj

# NUMPY
from numpy import save, load

def np_dump(obj, name, path='obj'):
    """Saves an object in npy format."""
    if '.npy' not in name:
        name = name + '.npy'
    path = os.path.join(path, name)
    save(path, obj)

def np_load(name, path='obj'):
    """Restores an object from a npy file."""
    if '.npy' not in name:
        name = name + '.npy'
    path = os.path.join(path, name)
    try:
        obj = load(path)
    except FileNotFoundError:
        obj = None
    return obj

# JSON
import json
def save_json(dict_obj, path, name):
    """Saves a dictionary in json format."""
    if '.json' not in name:
        name += '.json'
    with open(os.path.join(path, name), 'w') as json_file:
        json.dump(dict_obj, json_file)

def load_json(path, name):
    """Restores a dictionary from a json file."""
    if '.json' not in name:
        name += '.json'
    with open(os.path.join(path, name), 'r') as json_file:
        return json.load(json_file)

# NIFTY
def nifty_dump(x, name, path):
    if 'torch.Tensor' in str(type(x)):
        x = x.detach().cpu().numpy()
    if '.nii' not in name:
        name = name + '.nii.gz'
    # Remove channels dimension and rotate axis so depth first
    if len(x.shape) == 4:
        x = np.moveaxis(x[0], -1, 0)
    assert len(x.shape) == 3
    path = os.path.join(path, name)
    sitk.WriteImage(sitk.GetImageFromArray(x), path)

# OTHERS
import functools
def join_path(list):
    """From a list of chained directories, forms a path"""
    return functools.reduce(os.path.join, list)

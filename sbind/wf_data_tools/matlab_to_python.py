import h5py
import scipy.io as sio
import numpy as np
from pathlib import Path
import builtins

builtin_types = tuple(getattr(builtins, t) for t in dir(builtins) if isinstance(getattr(builtins, t), type) and getattr(builtins, t) is not object and getattr(builtins, t) is not type(None))

def make_sure_parent_dir_exists(file_path):
  path = Path(file_path)
  path.parent.mkdir(parents=True, exist_ok=True)

def loadmat(file_path, variable_names=None):
  try:
    mat_dict = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True, chars_as_strings=True, variable_names=variable_names)
  except NotImplementedError:
    py_data = {}
    with h5py.File(file_path, 'r') as f:
      for k, v in f.items():
        if not isinstance(v, h5py._hl.dataset.Dataset):
          continue

        if isinstance(v, h5py.h5r.Reference):
          py_data[k] = f[v]
          continue

        py_data[k] = np.array(v[()]).T
    return py_data
  return _check_keys(mat_dict)

def _check_keys(d):
  for key in d:
    if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
      d[key] = _todict(d[key])
    elif isinstance(d[key], np.ndarray) and len(d[key]) > 0 and isinstance(d[key].item(0), sio.matlab.mio5_params.mat_struct):
      for i in range( d[key].size ):
        if isinstance(d[key].item(i), sio.matlab.mio5_params.mat_struct):
          d[key].itemset( i, _todict( d[key].item(i) ) )
    else:
      pass
  return d

def _todict(matobj):
  d = {}
  for key in matobj._fieldnames:
    elem = matobj.__dict__[key]
    if isinstance(elem, sio.matlab.mio5_params.mat_struct):
      d[key] = _todict(elem)
    elif isinstance(elem, np.ndarray) and elem.size > 0 and isinstance(elem.item(0), sio.matlab.mio5_params.mat_struct):
      for i in range( elem.size ):
        if isinstance(elem.item(i), sio.matlab.mio5_params.mat_struct):
          elem.itemset(i, _todict( elem.item(i) ) )
      d[key] = elem
    else:
      d[key] = elem
  return d

def load_cd_dataset(file_path):
  return loadmat(file_path)
"""Miscellaneous utility functions."""
from copy import deepcopy
import numpy as np
import torch

def carry_to_device(data, device, dtype=torch.float32):
  if torch.is_tensor(data):
    return data.to(device)
  
  elif isinstance(data, np.ndarray):
    return torch.tensor(data, dtype=dtype).to(device)

  elif isinstance(data, dict):
    for key in data.keys():
      data[key] = carry_to_device(data[key], device)
    return data
  
  elif isinstance(data, list):
    for i, d in enumerate(data):
      data[i] = carry_to_device(d, device)
    return data

  return data

############################################################################
# These functions are taken from: https://github.com/pytorch/pytorch/issues/8741
def optimizer_to(optim, device):
  for param in optim.state.values():
    # Not sure there are any global tensors in the state dict
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
      for subparam in param.values():
        if isinstance(subparam, torch.Tensor):
          subparam.data = subparam.data.to(device)
          if subparam._grad is not None:
            subparam._grad.data = subparam._grad.data.to(device)

def lr_scheduler_to(sched, device):
  for param in sched.__dict__.values():
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)

############################################################################

def extract_dict_from_key(dict_, key_):
  if not isinstance(dict_[key_], dict):
    return dict_
  
  dict_inside = dict_[key_]
  for key, val in dict_inside.items():
    dict_[f'{key_}_{key}'] = val
  
  dict_.pop(key_, None)
  return dict_

def convert_to_tensor(x, dtype=torch.float32):
  if isinstance(x, torch.Tensor):
    return x
  elif isinstance(x, np.ndarray):
    # Use np.ndarray as middle step so that function works with tf tensors as well
    return torch.tensor(x, dtype=dtype)
  elif isinstance(x, list):
    return [convert_to_tensor(elem) for elem in x]
  elif isinstance(x, dict):
    x_as_tensor = {}
    for k, v in x.items():
      x_as_tensor[k] = convert_to_tensor(v)
    return x_as_tensor
  raise ValueError('Only Numpy array can be converted to tensor.')

def convert_to_numpy(x, dtype=np.float32):
  if isinstance(x, np.ndarray):
    return x
  elif isinstance(x, torch.Tensor):
    return x.detach().numpy()
  elif isinstance(x, list):
    return [convert_to_numpy(elem) for elem in x]
  elif isinstance(x, dict):
    x_as_numpy = {}
    for k, v in x.items():
      x_as_numpy[k] = convert_to_numpy(v)
    return x_as_numpy
  raise ValueError('Only tensor elements can be converted to numpy.')

def merge_dicts(dict2, dict1):
  if isinstance(dict2, dict):
    for key in dict2.keys():
      if key not in dict1: 
        if isinstance(dict2[key], dict):
          dict1[key] = {}
        else:
          dict1[key] = None
      dict1[key] = merge_dicts(dict1[key], dict2[key])
      return dict1

  # else not isinstance(dict2, dict)
  if isinstance(dict2, torch.Tensor):
    dict2_cp = dict2.clone().detach()
  else:
    dict2_cp = torch.tensor(dict2, dtype=torch.float32).reshape(1, -1)
  
  if dict1 is None:
    return dict2_cp.clone().detach()
  elif not isinstance(dict1, torch.Tensor):
    dict1 = torch.tensor(dict1, dtype=torch.float32).reshape(1, -1)
  return torch.cat((dict1, dict2_cp), dim=0)

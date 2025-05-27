"""Timeseries custom datasets."""


from ..utility import torch_utils as misc_torch_utils

import numpy as np
import random
from torch.utils.data import Dataset
import torch

_SEED = 24

class ImageDataset(Dataset):
    def __init__(self, your_data):  # Initialize with your data
        self.data = your_data

    def __len__(self):  # Return the total number of samples
        return len(self.data)

    def __getitem__(self, index):  # Retrieve a sample at the specified index
        sample = self.data[index]
        return sample



class TimeSeriesDataset(Dataset):
  def __init__(self, data, seq_len=1, step_ahead=0):
    """Input data should be 2D array (num_samples, num_features).

    Single dimensional (i.e., num_features = 1) should still be 2D with shape
    (num_samples, 1).
    """
    if len(data.shape) == 1:  # Just in case user doesn't read documentation.
      data = torch.atleast_2d(data).T  # (num_samples, 1)

    self.data = data
    self.seq_len = seq_len
    self.step_ahead = step_ahead

  def __len__(self):
    return max(0, (self.data.shape[0] - self.step_ahead) // self.seq_len) #  - self.seq_len check if it is needed
    # return int(self.data.shape[0] // self.seq_len)

  def __getitem__(self, idx):
    start_idx = idx * self.seq_len
    end_idx = start_idx + self.seq_len + self.step_ahead

    # Ensure we don't go beyond the end of the data
    end_idx = min(end_idx, self.data.shape[0])

    return self.data[start_idx:end_idx, ...]


class MultimodalTimeSeriesDataset(Dataset):
  def __init__(self, time_series_kwargs):
    self.__check_kwargs(time_series_kwargs)
    self.multimodal_timeseries = {}
    self.multimodal_timeseries.update(**time_series_kwargs)

  def __check_kwargs(self, input_kwargs):
    self.len, seq_len = -1, -1
    no_seq_len_warning = False
    for k, v in input_kwargs.items():
      if not isinstance(v, TimeSeriesDataset):
        raise ValueError('All provided datasets must be of TimeSeriesDataset type.')

      if self.len < 0:
        self.len = len(v)
      elif self.len != len(v):
        raise ValueError('All timeseries must have the same length.')

      if seq_len < 0:
        seq_len = v.seq_len
      elif seq_len != v.seq_len and no_seq_len_warning:
        no_seq_len_warning = True
        print('Warning: different timeseries use different sequence lengths.')

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    sample = {}
    for k, v in self.multimodal_timeseries.items():
      sample[k] = v[idx]
    return sample


from torch.utils.data import Sampler


class StatefulSampler(Sampler):
  def __init__(self, len_data, batch_size):
    original_order = np.arange(len_data)
    self.num_batch = np.ceil(len_data / batch_size).astype(int)
    self.indices = np.concatenate([original_order[i::self.num_batch] for i in range(self.num_batch)])

  def __iter__(self):
    return iter(self.indices)

  def __len__(self):
    return len(self.indices)


def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2 ** 32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


def create_reproducible_dataloader(dataset, **dataloader_kwargs):
  g = torch.Generator()
  g.manual_seed(_SEED)
  return torch.utils.data.DataLoader(dataset, worker_init_fn=seed_worker,
                                     generator=g, **dataloader_kwargs)


def create_dataloader_from_data(y, z, mask=None, z_mask=None, seq_len=128, step_ahead=0, batch_size=1, stateful=False, shuffle=False, num_workers=0):
  y_torch = misc_torch_utils.convert_to_tensor(y)
  z_torch = misc_torch_utils.convert_to_tensor(z)
  if mask is not None:
    if z_mask is None:
      z_mask = mask
    mask_torch = misc_torch_utils.convert_to_tensor(mask)
    z_mask = misc_torch_utils.convert_to_tensor(z_mask)
    y_z_ts = MultimodalTimeSeriesDataset(
      {'y': TimeSeriesDataset(y_torch, seq_len=seq_len, step_ahead=step_ahead),
       'z': TimeSeriesDataset(z_torch, seq_len=seq_len, step_ahead=step_ahead),
       'mask': TimeSeriesDataset(mask_torch, seq_len=seq_len, step_ahead=step_ahead),
       'z_mask': TimeSeriesDataset(z_mask, seq_len=seq_len, step_ahead=step_ahead),
       }
    )
  else:
    y_z_ts = MultimodalTimeSeriesDataset(
      {'y': TimeSeriesDataset(y_torch, seq_len=seq_len, step_ahead=step_ahead),
       'z': TimeSeriesDataset(z_torch, seq_len=seq_len, step_ahead=step_ahead),
       }
    )

  sampler = StatefulSampler(len(y_z_ts), batch_size) if stateful else None

  return create_reproducible_dataloader(y_z_ts, sampler=sampler, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)



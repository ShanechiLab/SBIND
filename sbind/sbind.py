import copy
import sbind.models.rnn_cells as custom_cells
import numpy as np

import time
import torch
import torch.nn as nn
from .models import models
from .models import model_helpers
_TIME_TESTS = False

class SBIND(nn.Module):
  def __init__(self, A_config, K_config, Cy_config, nx : int,
               Cz_config : model_helpers.ConvConfig = None,
               n1 : int = 0,
               A2_config:model_helpers.ConvConfig = None, K2_config:model_helpers.ConvConfig = None, Cy2_config:model_helpers.ConvConfig = None, Cz2_config: model_helpers.ConvConfig = None,
               fit_Cz2 : bool = True,
               use_lstm : bool = False,
               Cy_observation_model : str = 'gaussian', # gaussian | poisson
               Cz_observation_model : str = 'gaussian', # gaussian | poisson
               unified_K : bool = False,
               device : str = 'cuda', **rnncell_kwargs):
    super(SBIND, self).__init__()

    if nx <= 0:
      raise ValueError('Invalid nx input, must be greater than 0.')
    if n1 and not Cz_config:
      raise ValueError(f'n1={n1} > 0 but no Cz_config provided')

    self.device = torch.device(device)
    self.nx = nx
    self.n1 = min(n1, self.nx) # n1 can't be greater than nx.
    self.fit_Cz2 = fit_Cz2
    self.Cy_observation_model = Cy_observation_model
    self.Cz_observation_model = Cz_observation_model
    self.use_lstm = use_lstm

    self.A_config = copy.deepcopy(A_config)
    self.K_config = copy.deepcopy(K_config)
    self.Cy_config = copy.deepcopy(Cy_config)
    self.Cz_config = copy.deepcopy(Cz_config)


    self.stage_1, self.stage_2 = None, None
    self.unified_K = unified_K
    K_stg2_config = copy.deepcopy(K_config)

    if self.n1 > 0: # Stage 1 RNN+Deocder.
      self.stage_1 = SBIND_SingleStage(copy.deepcopy(n1), copy.deepcopy(A_config), copy.deepcopy(K_config), copy.deepcopy(Cz_config),
                                       copy.deepcopy(Cy_config), device=self.device, use_lstm=self.use_lstm, unified_K=self.unified_K, **rnncell_kwargs)

    self.n2 = self.nx - self.n1
    if self.n2 > 0: # Stage 2 RNN+Decoder.
      unified_K2 = self.unified_K

      if self.n1 > 0:
        K_stg2_input = K_config.in_channels
        # Don't update to self.n2 because unfied for now
        K_stg2_config = K_stg2_config.update_input_output_dims(K_stg2_input, K_stg2_config.out_channels, A_config.x_pixels, A_config.y_pixels)
        unified_K2 = True

      A2_config = A2_config if A2_config is not None else copy.deepcopy(self.A_config)
      K2_config = K2_config if K2_config is not None else copy.deepcopy(K_stg2_config)
      Cy2_config = Cy2_config if Cy2_config is not None else copy.deepcopy(self.Cy_config)
      Cz2_config = Cz2_config if (Cz2_config is not None and self.fit_Cz2) else Cz_config if (self.fit_Cz2 and Cz_config) else None

      self.stage_2 = SBIND_SingleStage(self.n2, A2_config, K2_config, Cy2_config,
                                       Cz2_config, self.n1, self.device, self.use_lstm, unified_K=unified_K2, **rnncell_kwargs)


  def _combine_stages(self, stage1_out, stage2_out, observation_model):
    if observation_model == 'gaussian':
      return torch.add(stage1_out, stage2_out)
    if observation_model == 'image_gaussian':
      return torch.add(self._high_pass_filter(stage1_out), stage2_out)
    if observation_model == 'poisson':
      return torch.multiply(stage1_out, stage2_out)
    else:
      raise ValueError(f'Unsupported observations type: {observation_model}')

  def forward(self, y, x_init=False):
    out = {} # Output will be different based on training mode vs eval mode.

    if self.n1 > 0: # Run stage 1.
      if _TIME_TESTS:
        stg1_start = time.time()

      z1_pred, x1_pred, y1_pred, z1f_pred = self.stage_1(y, x_init)
      if self.Cy_observation_model == 'poisson':
        y1_pred = torch.exp(y1_pred)

      if _TIME_TESTS:
        stg1_end = time.time()


    if self.n2 > 0: # Run stage 2.
      if _TIME_TESTS:
        stg2_start = time.time()
      y_stg2 = y
      if self.n1 > 0:
        stg2_out = self.stage_2(y_stg2, x_init, x1_pred.detach())
      else:
        stg2_out = self.stage_2(y_stg2, x_init)
      z2_pred = None
      if self.fit_Cz2:
        y2_pred, x2_pred, z2_pred, y2f_pred = stg2_out
      else:
        y2_pred, x2_pred, y2f_pred = stg2_out

      if self.Cy_observation_model == 'poisson':
        y2_pred = torch.exp(y2_pred)
        y2f_pred = torch.exp(y2f_pred)

      if _TIME_TESTS:
        stg2_end = time.time()

    if self.n1 > 0 and self.n2 > 0:
      if x1_pred.shape[-2:] != x2_pred.shape[-2:]:
        x1_pred = torch.nn.functional.pad(x1_pred, (0, x2_pred.size(-1) - x1_pred.size(-1), 0, x2_pred.size(-2) - x1_pred.size(-2)))

      out['x'] = torch.cat((x1_pred, x2_pred), -3) # Distinct states, concatenate.
      out['y'] = self._combine_stages(y1_pred, y2_pred, self.Cy_observation_model)
      if self.fit_Cz2:
        out['z'] = self._combine_stages(z1_pred, z2_pred, self.Cz_observation_model)
      else:
        out['z'] = z1_pred
      out['zf'] = z1f_pred
      out['yf'] = y2f_pred
      if _TIME_TESTS:
        print(f'stg1: {stg1_end - stg1_start}, stg2: {stg2_end - stg2_start}')

    elif self.n1 > 0:
      out['y'], out['x'], out['z'], out['zf'], out['yf'] = y1_pred, x1_pred, z1_pred, z1f_pred, {} # for now
      if _TIME_TESTS:
        print(f'stg1: {stg1_end - stg1_start}')


    else: # stage 2 only
      out['y'], out['x'], out['z'], out['zf'], out['yf'] = y2_pred, x2_pred, z2_pred, {}, y2f_pred # for now {}
      if _TIME_TESTS:
        print(f'stg2: {stg2_end - stg2_start}')

    if self.training: # Add the other outputs.
      if self.n1 > 0:
        out['y1'], out['x1'], out['z1'] = y1_pred, x1_pred, z1_pred
      if self.n2 > 0:
        out['y2'], out['x2'], out['z2'] = y2_pred, x2_pred, z2_pred
    return out


  def stg1_infer(self, y, x_init=False):

    out = {} # Output will be different based on training mode vs eval mode.

    if self.n1 > 0: # Run stage 1.
      if _TIME_TESTS:
        stg1_start = time.time()

      z1_pred, x1_pred, y1_pred, z1f_pred = self.stage_1(y, x_init)
      if self.Cy_observation_model == 'poisson':
        y1_pred = torch.exp(y1_pred)

      if _TIME_TESTS:
        stg1_end = time.time()


      out['y'], out['x'], out['z'], out['zf'], out['yf'] = y1_pred, x1_pred, z1_pred, z1f_pred, {} # for now
      if _TIME_TESTS:
        print(f'stg1: {stg1_end - stg1_start}')


    return out


  def _high_pass_filter(stage1_out, kernel_size=5):
    """
    Applies a high-pass filter to a 5D tensor (b, seq_len, ch, x, y) by converting it to 4D,
    applying a scaled high-pass filter kernel, and projecting back to 5D.

    Args:
        stage1_out (torch.Tensor): Input tensor of shape (b, seq_len, ch, x, y).
        kernel_size (int): Size of the high-pass filter kernel (default is 5).

    Returns:
        torch.Tensor: High-pass filtered output with the same shape as input.
    """
    # Ensure input is 5D
    if stage1_out.ndim != 5:
      raise ValueError("Input tensor must be 5D (b, seq_len, ch, x, y)")

    b, seq_len, ch, x, y = stage1_out.shape

    # Create a high-pass filter kernel
    kernel = torch.ones((kernel_size, kernel_size), dtype=stage1_out.dtype, device=stage1_out.device)
    kernel /= kernel.numel()  # Normalize to make a mean filter
    kernel = -kernel  # Invert to start high-pass behavior
    kernel[kernel_size // 2, kernel_size // 2] += 1  # Center value becomes 1 - (1/n^2)
    kernel *= 0.05
    kernel = (torch.ones((kernel_size, kernel_size), dtype=stage1_out.dtype, device=stage1_out.device) * (
              -kernel_size * 1e-2 / (kernel_size ** 2))).fill_diagonal_(
      kernel_size * 1e-2 - kernel_size * 1e-2 / (kernel_size ** 2))

    # Reshape the input to 4D for convolution: (b*seq_len, ch, x, y)
    stage1_out_4d = stage1_out.view(b * seq_len, ch, x, y)

    # Apply the high-pass filter using depthwise convolution
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape to (1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2  # Same padding to maintain output shape
    filtered_out = torch.nn.functional.conv2d(stage1_out_4d, kernel, padding=padding, groups=ch)

    # Reshape back to 5D: (b, seq_len, ch, x, y)
    return filtered_out.view(b, seq_len, ch, x, y)



class SBIND_SingleStage(nn.Module):
  def __init__(self, nx, A_config, K_config, C_config,
               decoder_config : model_helpers.ConvConfig = None, n1 = None,
               device : str = 'cuda', use_lstm : bool = False, unified_K: bool = False, step_ahead_prediction: np.ndarray = None, indep_msa: bool = False, **rnncell_kwargs):

    super(SBIND_SingleStage, self).__init__()

    self.device = torch.device(device)
    self.nx = nx
    self.step_ahead_prediction = step_ahead_prediction
    self.indep_multiple_steps_ahead = indep_msa
    self.use_lstm = use_lstm
    self.unified_K = unified_K
    if self.unified_K:
      self.K_config = K_config
    else:
      self.K_config = K_config.update_input_output_dims(K_config.in_channels, nx, A_config.x_pixels, A_config.y_pixels)

    self.K2 = None
    self.K = models.ConvEncoder(K_config)

    if self.use_lstm:
      self.A_config = A_config.update_input_output_dims(nx + K_config.out_channels, 4*nx, A_config.x_pixels, A_config.y_pixels)
      self.rnn = custom_cells.ConvLSTMCell(A_config, device=torch.device(device), **rnncell_kwargs)
    else:
      if n1 is not None:
        self.n1 = n1 if n1 == 0 else n1
      in_channels_A = nx + K_config.out_channels if self.unified_K else nx
      in_channels_A = in_channels_A + self.n1 if n1 else in_channels_A
      self.x1_norm = nn.BatchNorm2d(self.n1, momentum=0.1) if n1 else None


      self.A_config = A_config.update_input_output_dims(in_channels_A, nx, A_config.x_pixels, A_config.y_pixels)
      self.rnn = custom_cells.ConvRNNCell(A_config, device=torch.device(device), step_ahead_prediction=step_ahead_prediction is not None, **rnncell_kwargs)


    if C_config.encoder_layers == C_config.decoder_layers:  # stage 2 for Cz
      self.C_config = C_config.update_input_output_dims(C_config.out_channels, nx, A_config.x_pixels, A_config.y_pixels)
      self.C = models.ConvEncoder(C_config)
    else:  # stage 1 for Cy
      self.C_config = C_config.update_input_output_dims(C_config.in_channels, nx, A_config.x_pixels, A_config.y_pixels)
      self.C = models.ConvDecoder(C_config)

    self.decoder = None

    if decoder_config is not None:  # Secondary modality is optional.
      if decoder_config.encoder_layers == decoder_config.decoder_layers:  # stage 2 for Cz
        self.decoder_config = decoder_config.update_input_output_dims(decoder_config.out_channels, nx, A_config.x_pixels,
                                                                      A_config.y_pixels)
        self.decoder = models.ConvEncoder(decoder_config)
      else: # stage 1 for Cy
        self.decoder_config = decoder_config.update_input_output_dims(decoder_config.in_channels, nx, A_config.x_pixels,
                                                                      A_config.y_pixels)
        self.decoder = models.ConvDecoder(decoder_config)
    self._use_proj = False

  def use_proj(self):
    self._use_proj = True

  def forward(self, y, x_init=False, x1_states=None):

    recur_len = y.shape[1] if self.step_ahead_prediction is None else y.shape[1] - (self.step_ahead_prediction[-1]-1)
    if x_init or not self.rnn.stateful or self.rnn.state is None:  # Initialize to zero.
      self.rnn.set_state(nn.init.zeros_(torch.empty(y.shape[0], self.rnn.state_size, self.rnn.y_pixels, self.rnn.x_pixels).squeeze(-2).squeeze(-1)))
    elif self.use_lstm:
      self.rnn.set_state(self.rnn.state[:y.shape[0]])
    else:
      self.rnn.set_state(self.rnn.state[:y.shape[0]])

    x_pred = torch.empty((y.shape[0], recur_len) + (self.rnn.state_size, self.rnn.y_pixels, self.rnn.x_pixels)).squeeze(-2).squeeze(-1).to(self.device)

    if _TIME_TESTS:
      rnn_start = time.time()


    if self._use_proj:
      y, _, _ = self.projection_head(y)
    y, pool_idx, target_output_size = self.K(y)

    if x1_states is not None: # stage 1 -> stage 2
      if x1_states.shape[-2:] != y.shape[-2:]:
        x1_states = nn.functional.interpolate(x1_states.view(-1, *x1_states.shape[2:]), size=y.shape[-2:], mode='nearest').view(*x1_states.shape[:3], *y.shape[-2:])

      if self.n1 == 1:
        x1_states = x1_states.mean(axis=-3, keepdim=True)

      x1_states = self.x1_norm(
        x1_states.view(-1, *x1_states.shape[2:])
      ).view(*x1_states.shape[:3], *x1_states.shape[-2:])
      if self.K2 is not None:
        y, pool_idx2, target_output_size2 = self.K2(torch.cat((y, x1_states), -3))
      else:
        y = torch.cat((y, x1_states), -3)

    for i in range(recur_len):
      x_pred[:, i, :] = self.rnn.state
      self.rnn(y[:, i, :])

    C_pred = self.C(x_pred, pool_idx, target_output_size)

    Cf_pred = {}
    if self.step_ahead_prediction is not None:
      if self.indep_multiple_steps_ahead:
        xf_pred = x_pred.detach()
      else:
        xf_pred = x_pred

      for step_ahead_i in range(2, self.step_ahead_prediction[-1]+1):
        xf_pred = self.rnn.A_f(xf_pred)[0]

        if step_ahead_i in self.step_ahead_prediction:
          Cf_pred[step_ahead_i] = self.C(xf_pred)


    if _TIME_TESTS:
      rnn_end = time.time()
      print(f'RNN time: {rnn_end - rnn_start}')

    if self.decoder is None:
      return C_pred, x_pred, Cf_pred

    if _TIME_TESTS:
      decoder_start = time.time()

    decoder_pred = self.decoder(x_pred.detach(), pool_idx, target_output_size)

    if _TIME_TESTS:
      decoder_end = time.time()
      print(f'decoder time: {decoder_end - decoder_start}')

    return C_pred, x_pred, decoder_pred, Cf_pred


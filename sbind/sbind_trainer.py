import sbind.sbind as SBIND
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

_BASE_PATH = 'RESULTS_FOLDER'
_TIME_TESTS = False

_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def expand_zeros(mask, m):
  shape = mask.shape

  for i in range(shape[0]):
    zeros_indices = torch.where(mask[i, :, 0, 0] == 0)[0]

    for idx in zeros_indices:
      end_idx = min(idx + m + 1, shape[1])
      mask[i, idx:end_idx, 0, 0] = 0

  return mask

def get_focal_loss(pred, true, focal_kwargs=[1.0, 0.9]):
  gamma, alpha = focal_kwargs
  pred_sigmoid = pred.sigmoid()
  batch_loss_cls = F.binary_cross_entropy(pred_sigmoid, true.float(), reduction='none')
  pos_msk =1.0 *(true == 1)
  if pos_msk.numel() == 0:
    pos_msk = torch.tensor(1.0, requires_grad=True, device=pred.device)
  p_t_weight = (1.0 - true) * (1.0 - pred_sigmoid) + true * pred_sigmoid
  alpha_soft = (1.0 - true) * (1.0 - alpha) + true * alpha
  batch_loss_cls_focal = alpha_soft * torch.pow(p_t_weight.detach(), gamma) * batch_loss_cls

  rcnn_loss_focal = batch_loss_cls_focal.sum() / torch.clamp(pos_msk.sum(), min=1.0)
  if rcnn_loss_focal.item() == 0:
    rcnn_loss_focal = torch.tensor(0.0, requires_grad=True, device=pred.device)
  return rcnn_loss_focal

def gradient_loss(pred, true, mask=None, temporal_mask=None):
  def compute_gradient(x):
    gradient_x = torch.diff(x, dim=-1)
    gradient_y = torch.diff(x, dim=-2)
    return gradient_x, gradient_y

  pred_gradient_x, pred_gradient_y = compute_gradient(pred)
  true_gradient_x, true_gradient_y = compute_gradient(true)

  if mask is not None and temporal_mask is not None:
    loss_x = torch.mean(mask[...,:,:-1] * temporal_mask * torch.abs(pred_gradient_x - true_gradient_x))
    loss_y = torch.mean(mask[...,:-1,:] * temporal_mask * torch.abs(pred_gradient_y - true_gradient_y))
  elif mask is not None:
    loss_x = torch.mean(mask[...,:,:-1] * torch.abs(pred_gradient_x - true_gradient_x))
    loss_y = torch.mean(mask[...,:-1,:] * torch.abs(pred_gradient_y - true_gradient_y))
  elif temporal_mask is not None:
    loss_x = torch.mean(temporal_mask * torch.abs(pred_gradient_x - true_gradient_x))
    loss_y = torch.mean(temporal_mask * torch.abs(pred_gradient_y - true_gradient_y))
  else:
    loss_x = torch.mean(torch.abs(pred_gradient_x - true_gradient_x))
    loss_y = torch.mean(torch.abs(pred_gradient_y - true_gradient_y))

  return loss_x + loss_y

def l1_loss(pred, true, mask=None, temporal_mask=None):
  if mask is not None and temporal_mask is not None:
    return torch.mean(mask * temporal_mask * torch.abs(pred - true))
  elif mask is not None and temporal_mask is None:
    return torch.mean(mask * torch.abs(pred - true))
  elif mask is None and temporal_mask is not None:
    return torch.mean(temporal_mask * torch.abs(pred - true))
  else:
    return F.l1_loss(pred, true)

def mse_loss(pred, true, mask=None, temporal_mask=None):
  if mask is not None and temporal_mask is not None:
    return torch.mean(mask * temporal_mask *(pred - true) ** 2)
  elif mask is not None and temporal_mask is None:
    return torch.mean(mask * (pred - true) ** 2)
  elif mask is None and temporal_mask is not None:
    return torch.mean(temporal_mask * (pred - true) ** 2)
  else:
    return F.mse_loss(pred, true)

def combined_loss(pred, true, mask=None, temporal_mask=None, lambda_l1=0.01, lambda_grad=0.1):
  mse = mse_loss(pred, true, mask, temporal_mask)

  l1 = l1_loss(pred, true, mask, temporal_mask)

  grad = gradient_loss(pred, true, mask, temporal_mask)

  return mse + lambda_l1 * l1 + lambda_grad * grad

def save_checkpoint(model, A_Config, K_Config, Cy_Config, Cz_Config, optimizers, epoch, checkpoint_path, losses, val_losses, step_ahead_prediction=None, indep_msa=False, use_lstm=False, stateful=True):
  checkpoint = {
    'epoch': epoch,
    'nx': model.nx,
    'n1': model.n1,
    'A_Config': A_Config,
    'K_Config': K_Config,
    'Cz_Config': Cz_Config,
    'Cy_Config': Cy_Config,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
    'losses': losses,
    'val_losses': val_losses,
    'step_ahead_prediction': step_ahead_prediction,
    'indep_msa': indep_msa,
    'use_lstm': use_lstm,
    'stateful': stateful
  }
  torch.save(checkpoint, checkpoint_path)

class early_stopping_with_min_epochs():
  def __init__(self, bind_trainer, loss_name, monitor, patience,
               restore_best_weights=False, start_from_epoch=0):
    self.bind_trainer = bind_trainer
    self.loss_name = loss_name
    self.monitor = monitor
    self.patience = patience
    self.restore_best_weights = restore_best_weights
    self.start_from_epoch = start_from_epoch

    self.best_measure = float('inf')
    self.best_model_state_dict = None
    self.early_stopping_counter = 0

  def should_stop(self, measure, epoch):
    if epoch < self.start_from_epoch:
      return False

    if measure < self.best_measure:
      self.early_stopping_counter = 0
      self.best_measure = measure
      if self.restore_best_weights:
        if self.loss_name == 'stage_1_loss1':
          self.best_model_state_dict = self.bind_trainer.model.stage_1.state_dict()
        elif self.loss_name == 'stage_1_loss2':
          self.best_model_state_dict = self.bind_trainer.model.stage_1.decoder.state_dict()
        elif self.loss_name == 'stage_2_loss1':
          self.best_model_state_dict = self.bind_trainer.model.stage_2.state_dict()
        elif self.loss_name == 'stage_2_loss2':
          self.best_model_state_dict = self.bind_trainer.model.stage_2.decoder.state_dict()
    else:
      self.early_stopping_counter += 1

    if self.early_stopping_counter >= self.patience:
      if self.restore_best_weights and self.best_model_state_dict is not None:
        if self.loss_name == 'stage_1_loss1':
          self.bind_trainer.model.stage_1.load_state_dict(self.best_model_state_dict)
        elif self.loss_name == 'stage_1_loss2':
          self.bind_trainer.model.stage_1.decoder.load_state_dict(self.best_model_state_dict)
        elif self.loss_name == 'stage_2_loss1':
          self.bind_trainer.model.stage_2.load_state_dict(self.best_model_state_dict)
        elif self.loss_name == 'stage_2_loss2':
          self.bind_trainer.model.stage_2.decoder.load_state_dict(self.best_model_state_dict)

      return True

    return False

class CustomLRScheduler:
  def __init__(self, optimizer, lr_step_size, lr_gamma, wd_step_size=None, wd_gamma=None):
    self.optimizer = optimizer
    self.lr_step_size = lr_step_size
    self.lr_gamma = lr_gamma
    self.wd_step_size = wd_step_size
    self.wd_gamma = wd_gamma
    self.lr_epoch_count = 0
    self.wd_epoch_count = 0

  def step(self):
    self.lr_epoch_count += 1
    self.wd_epoch_count += 1

    if self.lr_epoch_count >= self.lr_step_size:
      self.lr_epoch_count = 0
      for param_group in self.optimizer.param_groups:
        param_group['lr'] *= self.lr_gamma
    if self.wd_step_size is not None:
      if self.wd_epoch_count >= self.wd_step_size:
        self.wd_epoch_count = 0
        for param_group in self.optimizer.param_groups:
          param_group['weight_decay'] *= self.wd_gamma
import itertools
import torch
import torch.nn.functional as F
import torch.optim as optim
import time

class SBINDTrainer():
  def __init__(self, model : 'SBIND.SBIND', stage_1 : bool, stage_2 : bool,
               end_to_end : bool = False,
               optimizer = optim.AdamW, optim_kwargs={'lr': 0.001, 'weight_decay':1e-8},
               verbose=1, sap_loss_ratio=1.0, lambda_l1 = 1.0, lambda_grad = 0.1,
               cross_entropy_weight=None, image_mask=None, clip_gradients=None):
    self.model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    self.end_to_end = end_to_end
    self.stage_1, self.stage_2 = stage_1, stage_2
    self.verbose = verbose
    self.cross_entropy_weight = cross_entropy_weight
    self.sap_loss_ratio = sap_loss_ratio
    self.image_mask = image_mask
    self.clip_gradients = clip_gradients
    self.lambda_l1 = lambda_l1
    self.lambda_grad = lambda_grad

    if self.stage_1:
      stg1_loss1_params = itertools.chain(self.model.stage_1.rnn.parameters(), self.model.stage_1.K.parameters(), self.model.stage_1.C.parameters())
      self.optimizer_stg1_loss1 = optimizer(stg1_loss1_params, **optim_kwargs)
      self.optimizer_stg1_loss2 = optimizer(self.model.stage_1.decoder.parameters(), **optim_kwargs)

    if self.stage_2:
      if self.model.stage_2.x1_norm is not None:
        stg2_loss1_params = itertools.chain(self.model.stage_2.rnn.parameters(), self.model.stage_2.K.parameters(), self.model.stage_2.C.parameters(), self.model.stage_2.x1_norm.parameters())
      else:
        stg2_loss1_params = itertools.chain(self.model.stage_2.rnn.parameters(), self.model.stage_2.K.parameters(), self.model.stage_2.C.parameters())
      self.optimizer_stg2_loss1 = optimizer(stg2_loss1_params, **optim_kwargs)

      if self.model.fit_Cz2:
        self.optimizer_stg2_loss2 = optimizer(self.model.stage_2.decoder.parameters(), **optim_kwargs)

    if self.stage_1:
      if hasattr(self.model.stage_1, 'projection_head') and self.model.stage_1.projection_head:
        self.optimizer_stg1_proj = optimizer(self.model.stage_1.projection_head.parameters(), **optim_kwargs)
    if self.stage_2:
      if hasattr(self.model.stage_2, 'projection_head') and self.model.stage_2.projection_head:
        self.optimizer_stg2_proj = optimizer(self.model.stage_2.projection_head.parameters(), **optim_kwargs)

  def _compute_loss(self, true, pred, observation_model, mask=None, temporal_mask=None):
    if observation_model == 'image_gaussian':
      return combined_loss(pred, true, mask, temporal_mask, lambda_l1=self.lambda_l1, lambda_grad=self.lambda_grad)

    if observation_model == 'gaussian':
      return mse_loss(pred, true, mask, temporal_mask)
    elif observation_model == 'poisson':
      return F.poisson_nll_loss(pred, true, log_input=False)
    elif observation_model in ['categorical', 'categorical_focal']:
      self.num_classes = int(pred.shape[-1] / true.shape[-1])
      nz = true.shape[-1]
      if temporal_mask is not None:
        if temporal_mask.dim() == 3 and temporal_mask.shape[-1] == 1:
          temporal_mask = temporal_mask.squeeze(-1).bool()
        else:
          temporal_mask = temporal_mask.bool()[:,:,0]
        true = true[temporal_mask]
        pred = pred[temporal_mask]

        pred = pred.view(-1, nz, self.num_classes).view(-1, self.num_classes)
        true = true.reshape(-1).long()
        if observation_model == 'categorical':
          return F.cross_entropy(pred, true, weight=self.cross_entropy_weight) if true.shape[0] > 0 else torch.tensor(0.0, requires_grad=True, device=pred.device)
        else:
          return get_focal_loss(pred.view(-1), true, focal_kwargs=self.cross_entropy_weight) if true.shape[0] > 0 else torch.tensor(0.0, requires_grad=True, device=pred.device)

      else:
        pred = pred.view(pred.shape[0], pred.shape[1], self.num_classes, -1).permute(0, 2, 1, 3)
    elif observation_model == 'categorical_seq':
      num_classes = int(pred.shape[-1] / true.shape[-1])
      reshaped_pred = pred.view(pred.shape[0], num_classes, -1)
      return F.cross_entropy(reshaped_pred, true[:,0].long())
    raise ValueError(f'Unsupported observation model {observation_model}')

  def _stage_1_loss1(self, data, model_out, temporal_mask=None, eval=False):
    if _TIME_TESTS:
      start = time.time()
    this_temporal_mask = temporal_mask[:, :model_out['z'].shape[1], :] if temporal_mask is not None else None
    loss_pred = self._compute_loss(data['z'][:, :model_out['z'].shape[1], :], model_out['z'], self.model.Cz_observation_model,
                                   temporal_mask=this_temporal_mask)
    loss_ms_ahead = torch.tensor(0, dtype=torch.float32, device=loss_pred.device)
    for step_ahead_i, model_out_i in model_out.items():
      if step_ahead_i != 'z':
        if temporal_mask is not None:
          this_temporal_mask = temporal_mask[:, step_ahead_i-1:step_ahead_i-1+model_out['z'].shape[1], :]
          this_temporal_mask = expand_zeros(this_temporal_mask,step_ahead_i-1)
        else:
          this_temporal_mask = None
        loss_ms_ahead += self._compute_loss(data['z'][:, step_ahead_i-1:step_ahead_i-1+model_out['z'].shape[1], :],
                                            model_out_i,
                                            self.model.Cz_observation_model,
                                            temporal_mask=this_temporal_mask)
    loss = loss_pred + self.sap_loss_ratio * loss_ms_ahead

    if eval:
      return loss

    if self.model.stage_1._use_proj:
      self.optimizer_stg1_proj.zero_grad(set_to_none=True)
      loss.backward()
      if self.clip_gradients is not None:
        torch.nn.utils.clip_grad_value_(self.model.stage_1.projection_head.parameters(), clip_value=self.clip_gradients)
      self.optimizer_stg1_proj.step()

    else:
      self.optimizer_stg1_loss1.zero_grad(set_to_none=True)
      loss.backward()
      if self.clip_gradients is not None:
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_gradients)
      self.optimizer_stg1_loss1.step()

    if _TIME_TESTS:
      end = time.time()
      print(f'loss1: {end - start}')
    return loss.item()

  def _stage_1_loss2(self, data, model_out, temporal_mask=None, eval=False):
    this_temporal_mask = temporal_mask[:, :model_out['y'].shape[1], :] if temporal_mask is not None else None
    loss = self._compute_loss(data['y'][:, :model_out['y'].shape[1], :], model_out['y'], self.model.Cy_observation_model, self.image_mask, temporal_mask=this_temporal_mask)
    if eval:
      return loss
    self.optimizer_stg1_loss2.zero_grad(set_to_none=True)
    loss.backward()
    if self.clip_gradients is not None:
      torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_gradients)
    self.optimizer_stg1_loss2.step()
    return loss.item()

  def _stage_2_loss1(self, data, model_out, temporal_mask=None, eval=False):
    this_temporal_mask = temporal_mask[:, :model_out['y'].shape[1], :1]  if temporal_mask is not None else None
    loss_pred = self._compute_loss(data['y'][:, :model_out['y'].shape[1], :1], model_out['y'], self.model.Cy_observation_model, mask=self.image_mask, temporal_mask=this_temporal_mask)
    loss_ms_ahead = torch.tensor(0, dtype=torch.float32, device=loss_pred.device)
    for step_ahead_i, model_out_i in model_out.items():
      if step_ahead_i != 'y':
        if temporal_mask is not None:
          this_temporal_mask = temporal_mask.clone()
          this_temporal_mask = expand_zeros(this_temporal_mask,step_ahead_i-1)[:, step_ahead_i-1:step_ahead_i-1+model_out['y'].shape[1], :]
        else:
            this_temporal_mask = None
        loss_ms_ahead += self._compute_loss(data['y'][:, step_ahead_i-1:step_ahead_i-1+model_out['y'].shape[1], :], model_out_i, self.model.Cy_observation_model, mask=self.image_mask, temporal_mask=this_temporal_mask)
        loss_ms_ahead += self._compute_loss(data['y'][:, step_ahead_i-1:step_ahead_i-1+model_out['y'].shape[1], :], model_out_i, self.model.Cy_observation_model, mask=self.image_mask, temporal_mask=this_temporal_mask)

    loss = loss_pred + self.sap_loss_ratio * loss_ms_ahead

    if eval:
      return loss
    if self.model.stage_2._use_proj:
      self.optimizer_stg2_proj.zero_grad(set_to_none=True)
      loss.backward()
      if self.clip_gradients is not None:
        torch.nn.utils.clip_grad_value_(self.model.stage_2.projection_head.parameters(), clip_value=self.clip_gradients)
      self.optimizer_stg2_proj.step()

    else:
      self.optimizer_stg2_loss1.zero_grad(set_to_none=True)
      loss.backward()
      if self.clip_gradients is not None:
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_gradients)
      self.optimizer_stg2_loss1.step()
    return loss.item()

  def _stage_2_loss2(self, data, model_out, temporal_mask=None, eval=False):
    this_temporal_mask = temporal_mask[:, :model_out['z'].shape[1], :]  if temporal_mask is not None else None
    loss = self._compute_loss(data['z'][:, :model_out['z'].shape[1], :], model_out['z'], self.model.Cz_observation_model, temporal_mask=this_temporal_mask)

    if eval:
      return loss
    self.optimizer_stg2_loss2.zero_grad(set_to_none=True)
    loss.backward()
    if self.clip_gradients is not None:
      torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_gradients)
    self.optimizer_stg2_loss2.step()
    return loss.item()

  def _all_losses(self):
    all_losses = []
    if self.stage_1:
      all_losses.append((self._stage_1_loss1, 'stage_1_loss1'))
      all_losses.append((self._stage_1_loss2, 'stage_1_loss2'))

    if self.stage_2:
      all_losses.append((self._stage_2_loss1, 'stage_2_loss1'))
      if self.model.fit_Cz2:
        all_losses.append((self._stage_2_loss2, 'stage_2_loss2'))
    return all_losses

  def _get_optimizer(self, loss_name):
    if loss_name == 'stage_1_loss1':
      return self.optimizer_stg1_loss1 if not self.model.stage_1._use_proj else self.optimizer_stg1_proj
    elif loss_name == 'stage_1_loss2':
      return self.optimizer_stg1_loss2
    elif loss_name == 'stage_2_loss1':
      return self.optimizer_stg2_loss1 if not self.model.stage_2._use_proj else self.optimizer_stg2_proj
    elif loss_name == 'stage_2_loss2':
      return self.optimizer_stg2_loss2

  def _get_proj_optimizer(self, loss_name):
    if loss_name == 'stage_1_loss1':
      return self.optimizer_stg1_proj
    elif loss_name == 'stage_1_loss2':
      return self.optimizer_stg1_loss2
    elif loss_name == 'stage_2_loss1':
      return self.optimizer_stg2_proj
    elif loss_name == 'stage_2_loss2':
      return self.optimizer_stg2_loss2

  def _get_list_of_optimizers(self):
    return [self._get_optimizer(loss_name) for _, loss_name in self._all_losses()]

  def _fit_sequential(self, num_epochs, train_dataloader,
                      validation_dataloader=None, device=_DEVICE,
                      check_point_args=None, val_every_epoch=10,
                      early_stopping_measure='loss',
                      early_stopping_patience=3, start_from_epoch=10, restore_best_weights=False, scheduler_type=None,
                      **scheduler_args):
    history = {}
    self.image_mask = self.image_mask.to(device) if self.image_mask is not None else None
    for loss_func, loss_name in self._all_losses():
      if early_stopping_measure is not None:
        loss = np.empty((num_epochs[loss_name],))
        val_loss = np.empty((num_epochs[loss_name],))
        early_stopping = early_stopping_with_min_epochs(self, monitor=early_stopping_measure, loss_name=loss_name,
                                                        patience=early_stopping_patience,
                                                        restore_best_weights=restore_best_weights,
                                                        start_from_epoch=start_from_epoch)
      if scheduler_type is not None:
        scheduler = scheduler_type(self._get_optimizer(loss_name), **scheduler_args)
      epoch = 0
      for epoch in range(1, num_epochs[loss_name] + 1):
        epoch_loss = 0
        epoch_cnt = 0

        with tqdm(train_dataloader, unit='batch', disable=epoch % self.verbose) as tepoch:
          tepoch.set_description(f'Epoch {epoch}')
          for data in tepoch:
            postfix = {}

            data = misc_torch_utils.carry_to_device(data, device)
            x_init = True if epoch_cnt == 0 else None

            if 'stage_1' in loss_name:
              z_pred, _, y_pred, zf_pred = self.model.stage_1(data['y'], x_init)

              if loss_name == 'stage_1_loss1':
                epoch_loss += loss_func(data, {'z': z_pred} | zf_pred, data['z_mask'])

              elif loss_name == 'stage_1_loss2':
                epoch_loss += loss_func(data, {'y': y_pred}, data['mask'].unsqueeze(-1).unsqueeze(-1))

            elif 'stage_2' in loss_name:
              rnn_in = data['y']
              if self.stage_1:
                with torch.no_grad():
                  z1_pred, stg1_states, y1_pred, zf_pred = self.model.stage_1(data['y'], x_init)
                out = self.model.stage_2(rnn_in, x_init, stg1_states)
              else:
                out = self.model.stage_2(rnn_in, x_init)

              if self.model.fit_Cz2:
                y_pred, _, z_pred, yf_pred = out
              else:
                y_pred, _, yf_pred = out

              if loss_name == 'stage_2_loss1':
                if self.stage_1:
                  y_pred = self.model._combine_stages(y1_pred, y_pred, self.model.Cy_observation_model)
                  for step_ahead_i in yf_pred.keys():
                    yf_pred = self.model._combine_stages(y1_pred, yf_pred[step_ahead_i],
                                                         self.model.Cy_observation_model)

                epoch_loss += loss_func(data, {'y': y_pred} | yf_pred, data['mask'].unsqueeze(-1).unsqueeze(-1))

              elif loss_name == 'stage_2_loss2':
                if self.stage_1:
                  z_pred = self.model._combine_stages(z1_pred, z_pred, self.model.Cz_observation_model)
                epoch_loss += loss_func(data, {'z': z_pred}, data['z_mask'])
            epoch_cnt += 1
            postfix[loss_name] = epoch_loss / epoch_cnt
            tepoch.set_postfix(**postfix)
            if scheduler_type is not None:
              scheduler.step()

        if early_stopping_measure == 'loss':
          if early_stopping.should_stop(postfix[loss_name], epoch):
            print(f"Early stopping after {epoch + 1} epochs.")
            break
        elif early_stopping_measure == 'val_loss' and epoch % val_every_epoch == 0:
          val_loss[epoch - 1] = self.validation(validation_dataloader, loss_name, device, verbose=epoch % self.verbose)
          if early_stopping.should_stop(val_loss[epoch - 1], epoch):
            print(f"Early stopping after {epoch + 1} epochs.")
            break
        loss[epoch - 1] = postfix[loss_name]

      history['loss_' + loss_name] = loss[:epoch]
      history['val_loss_' + loss_name] = val_loss[:epoch]

    if check_point_args is not None:
      save_checkpoint(self.model, self.model.A_config,
                      self.model.K_config, self.model.Cy_config, self.model.Cz_config,
                      self._get_list_of_optimizers(), epoch,
                      os.path.join(check_point_args.get('base_path', _BASE_PATH),
                                   f'%s.pth' % (check_point_args['name'])), loss[:epoch],
                      val_loss[:epoch])

    return history

  def fit(self, num_epochs, train_dataloader, validation_dataloader=None, device=_DEVICE,
          early_stopping_patience=3, early_stopping_measure='loss',
          start_from_epoch=10, restore_best_weights=False, val_every_epoch=10,
          check_point_args=None, scheduler_type=None, **scheduler_args):

    self.model.to(device)
    self.model.train()
    rnn_fit_args = {'early_stopping_patience': early_stopping_patience,
                    'early_stopping_measure': early_stopping_measure,
                    'start_from_epoch': start_from_epoch,
                    'restore_best_weights': restore_best_weights}
    if self.end_to_end:
      if type(num_epochs) == int:
        temp = num_epochs
        num_epochs = {'stage_1': temp, 'stage_2': temp}
      history = self._fit_sequential(num_epochs, train_dataloader, validation_dataloader, device, **rnn_fit_args)
    else:
      if type(num_epochs) == int:
        temp = num_epochs
        num_epochs = {'stage_1_loss1': temp, 'stage_1_loss2': temp, 'stage_2_loss1': temp, 'stage_2_loss2': temp}

      history = self._fit_sequential(num_epochs, train_dataloader, validation_dataloader, device, check_point_args,
                                     val_every_epoch, **rnn_fit_args, scheduler_type=scheduler_type, **scheduler_args)
    return history

  def predict(self, dataloader, device=_DEVICE, stage1_only=False, predict_steps_ahead_prediction=None, verbose=False):
    self.model.to(device)
    self.model.eval()

    postfix = {}
    nx = self.model.n1 if stage1_only else self.model.nx
    num_samples = len(dataloader.dataset)

    sample_batch = next(iter(dataloader))
    _, seq_len, ny, y_pixel, x_pixel = sample_batch['y'].shape
    _, seq_len, nz = sample_batch['z'].shape
    if self.model.Cz_observation_model in ['categorical', 'categorical_focal']:
      self.num_classes = max(2, self.num_classes)
      nz = nz * self.num_classes
    max_ahead = 0
    yf_cat = {}
    zf_cat = {}
    if self.model.stage_1:
      state_xp_dim, state_yp_dim = self.model.stage_1.rnn.x_pixels, self.model.stage_1.rnn.y_pixels
      if self.model.stage_1.step_ahead_prediction is not None:
        if predict_steps_ahead_prediction is not None:
          temp = np.copy(self.model.stage_1.step_ahead_prediction)
          self.model.stage_1.step_ahead_prediction = predict_steps_ahead_prediction
        max_ahead = self.model.stage_1.step_ahead_prediction[-1] - 1
        for step_ahead_i in self.model.stage_1.step_ahead_prediction:
          zf_cat[step_ahead_i] = np.zeros((num_samples, seq_len - max_ahead, nz))

    if self.model.stage_2:
      state_xp_dim, state_yp_dim = self.model.stage_2.rnn.x_pixels, self.model.stage_2.rnn.y_pixels
      if self.model.stage_2.step_ahead_prediction is not None:
        if predict_steps_ahead_prediction is not None:
          temp = np.copy(self.model.stage_2.step_ahead_prediction)
          self.model.stage_2.step_ahead_prediction = predict_steps_ahead_prediction
        max_ahead = self.model.stage_2.step_ahead_prediction[-1] - 1
        for step_ahead_i in self.model.stage_2.step_ahead_prediction:
          yf_cat[step_ahead_i] = np.zeros((num_samples, seq_len - max_ahead, ny, y_pixel, x_pixel))

    seq_len = seq_len - max_ahead
    y_cat = np.zeros((num_samples, seq_len, ny, y_pixel, x_pixel))
    x_cat = np.zeros((num_samples, seq_len, nx, state_yp_dim, state_xp_dim))
    z_cat = np.zeros((num_samples, seq_len, nz))

    sampler = dataloader.sampler
    batch_size = dataloader.batch_size

    with tqdm(dataloader, unit='batch', disable=verbose) as tepoch:
      tepoch.set_description(f'predicting ... ')

      for batch_idx, item in enumerate(tepoch):
        tepoch.set_postfix(**postfix)

        x_init = True if batch_idx == 0 else False
        if hasattr(sampler, 'indices'):
          sample_idx = sampler.indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        else:
          sample_idx = range(batch_idx * batch_size, batch_idx * batch_size + len(item['y']))

        with torch.no_grad():
          if stage1_only:
            out = self.model.stg1_infer(item['y'].to(device), x_init)
          else:
            out = self.model(item['y'].to(device), x_init)

        if self.model.Cz_observation_model == 'categorical':
          out['z'] = F.softmax(out['z'].view(out['z'].shape[0], out['z'].shape[1], -1, self.num_classes), dim=3)
        elif self.model.Cz_observation_model == 'categorical_focal':
          probs = torch.sigmoid(out['z'])
          out['z'] = torch.stack([1 - probs, probs], dim=-1).view(probs.shape[0], probs.shape[1], -1)

        y_cat[sample_idx] = out['y'].cpu().detach().numpy().reshape(-1, seq_len, ny, y_pixel, x_pixel)
        x_cat[sample_idx] = out['x'].cpu().detach().numpy().reshape(-1, seq_len, nx, state_yp_dim, state_xp_dim)
        z_cat[sample_idx] = out['z'].cpu().detach().numpy().reshape(-1, seq_len, nz)

        for step_ahead_i in out['zf'].keys():
          zf_cat[step_ahead_i][sample_idx] = out['zf'][step_ahead_i].cpu().detach().numpy().reshape(-1, seq_len, nz)
        for step_ahead_i in out['yf'].keys():
          yf_cat[step_ahead_i][sample_idx] = out['yf'][step_ahead_i].cpu().detach().numpy().reshape(-1, seq_len, ny,
                                                                                                    y_pixel, x_pixel)

    y_cat = y_cat.reshape((-1, ny, y_pixel, x_pixel))
    x_cat = x_cat.reshape((-1, nx, state_yp_dim, state_xp_dim))
    z_cat = z_cat.reshape((-1, nz))

    if self.model.Cz_observation_model in ['categorical', 'categorical_focal']:
      z_cat = z_cat.reshape(-1, z_cat.shape[-1] // self.num_classes, self.num_classes)

    for step_ahead_i in zf_cat.keys():
      zf_cat[step_ahead_i] = zf_cat[step_ahead_i].reshape(-1, nz)
    for step_ahead_i in yf_cat.keys():
      yf_cat[step_ahead_i] = yf_cat[step_ahead_i].reshape(-1, ny, y_pixel, x_pixel)

    if predict_steps_ahead_prediction is not None:
      if self.model.stage_1:
        self.model.stage_1.step_ahead_prediction = temp
      if self.model.stage_2:
        self.model.stage_2.step_ahead_prediction = temp

    if zf_cat == {} and yf_cat == {}:
      return y_cat, x_cat, z_cat
    else:
      return y_cat, x_cat, z_cat, yf_cat, zf_cat

  def validation(self, dataloader, loss_name, device, verbose=True):
    self.model.to(device)
    self.model.eval()
    postfix = {}

    loss, num_sequences = 0, 0
    with torch.no_grad():
      with tqdm(dataloader, unit='batch', disable=verbose) as tepoch:
        tepoch.set_description(f'validation loss')
        for data in tepoch:

          x_init = True if num_sequences == 0 else False
          data = misc_torch_utils.carry_to_device(data, device)
          if 'stage_1' in loss_name:
            z_pred, _, y_pred, zf_pred = self.model.stage_1(data['y'], x_init)

            if loss_name == 'stage_1_loss1':
              loss += self._stage_1_loss1(data, {'z': z_pred} | zf_pred, data['z_mask'], eval=True)
            elif loss_name == 'stage_1_loss2':
              loss += self._stage_1_loss2(data, {'y': y_pred}, data['mask'].unsqueeze(-1).unsqueeze(-1), eval=True)

          elif 'stage_2' in loss_name:
            rnn_in = data['y']
            if self.stage_1:
              with torch.no_grad():
                z1_pred, stg1_states, y1_pred, zf_pred = self.model.stage_1(data['y'], x_init)
              out = self.model.stage_2(rnn_in, x_init, stg1_states)
            else:
              out = self.model.stage_2(rnn_in, x_init)

            if self.model.fit_Cz2:
              y_pred, _, z_pred, yf_pred = out
            else:
              y_pred, _, yf_pred = out

            if loss_name == 'stage_2_loss1':
              if self.stage_1:
                y_pred = self.model._combine_stages(y1_pred, y_pred, self.model.Cy_observation_model)
                for step_ahead_i in yf_pred.keys():
                  yf_pred = self.model._combine_stages(y1_pred, yf_pred[step_ahead_i], self.model.Cy_observation_model)

              loss += self._stage_2_loss1(data, {'y': y_pred} | yf_pred, data['mask'].unsqueeze(-1).unsqueeze(-1),
                                          eval=True)
            elif loss_name == 'stage_2_loss2':
              if self.stage_1:
                z_pred = self.model._combine_stages(z1_pred, z_pred, self.model.Cz_observation_model)
              loss += self._stage_2_loss2(data, {'z': z_pred}, data['z_mask'], eval=True)

          num_sequences += 1
          postfix[loss_name] = loss.item() / num_sequences
          tepoch.set_postfix(**postfix)

    self.model.train()
    avg_loss = loss / num_sequences
    return avg_loss
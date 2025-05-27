
from dataclasses import dataclass, field
import math
from .model_helpers import LayerConfig
from .model_helpers import ConvConfig
import torch
import torch.nn as nn
from typing import Callable, List, Union, Tuple, Optional



class MLP(nn.Module):
  """Multilayer Perceptron model.

  Args:
    dense_config: DenseConfig.

  Exmple usage:
    input_size, output_size, hidden_layers = 10, 1, [128]
    dense_config = DenseConfig(input_size, output_size, hidden_layers)
    mlp = MLP(dense_config)
  """
  def __init__(self, dense_config):
    super(MLP, self).__init__()
    self.mlp = dense_layers(dense_config)
  
  def forward(self, x):
    return self.mlp(x)

@dataclass
class DenseConfig:
  """Config for construction dense (fully-connected) network.

  Example usage:
    # If using single-elements for activation, etc.
    input_size, output_size, hidden_layers = 10, 1, [128]
    num_layers = len(hidden_layers) + 1
    activations = nn.ReLU
    dense_config = DenseConfig(input_size, output_size, hidden_layers,
                               activation=activations)

    # If providing lists, identical to above use.
    input_size, output_size, hidden_layers = 10, 1, [128]
    num_layers = len(hidden_layers) + 1
    activations = [nn.ReLU for _ in range(num_layers - 1)]
    activations.append(None) # no activation for last layer
    dense_config = DenseConfig(input_size, output_size, hidden_layers,
                               activation=activations)

  Args:
    input_size: int. Input dimension.
    output_size: int. Output dimension.
    hidden_layers: list of ints. Each integer corresonds to the size of the
      corresponding hidden layer.
    activations: single callabe or list of callable activation methods. Must
      correspond to a no argument callable activation function. If None then there
      is no activation for the corresponding layer (i.e., linear only). If list,
      must match the number of total layers. Default is no activation (i.e.,
      linear only all layers). See PyTorch examples: https://pytorch.org/docs/stable/nn.html
    init_funcs: list of no argument callable weight initialization functions. If
      None then PyTorch's default initialization will be used. Default is None for
      all layers.
    init_kwargs: list of dictionaries. These must be valid keyword arguments for
      the provided init_funcs. Must also match in length if provided. Default is
      no keyword arguments will be used.
    output_activation: callable. The final layer's activation function, otherwise
      none/linear (Default).
    bias: bool. All linear layers will fit a bias term. Default True.

  Returns:
    PyTorch nn.Sequential model.
  """
  input_size: int
  output_size: int
  hidden_layers: list = field(default_factory=list) # list[int]
  activation: Union[Callable, list] = None # list[Callable]
  # Default initialization for PyTorch linear layers:
  # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L103
  init_func: Union[Callable, list] = None # list[Callable]
  init_kwargs: Union[dict, list] = field(default_factory=dict) # list[dict]
  output_activation: Callable = None # Default is nn.Identity (i.e., linear)
  bias: bool = True
  dropout: Union[float, list] = None
  norm: bool = False

  def __post_init__(self):
    self.io_pairs = construct_io_pairs(self.input_size, self.output_size, self.hidden_layers)
    if self.dropout and isinstance(self.dropout, float):
      self.dropout = [self.dropout for _ in range(len(self.io_pairs) - 1)]
      self.dropout.append(0.0)
    if self.activation and isinstance(self.activation, Callable):
      self.activation = [self.activation for _ in range(len(self.io_pairs) - 1)]
      self.activation.append(self.output_activation)
        # for last layer
    if isinstance(self.init_func, Callable):
      self.init_func = [self.init_func for _ in range(len(self.io_pairs))]
    if isinstance(self.init_kwargs, dict):
      self.init_kwargs = [self.init_kwargs for _ in range(len(self.io_pairs))]

  def update_input_output_dims(self, input_size : int, output_size : int):
    self.input_size, self.output_size = input_size, output_size
    self.__post_init__()
    return self

import numpy as np
import torch.nn as nn

def dense_layers(dense_config : DenseConfig):
  layers = []
  for ind, io_pair in enumerate(dense_config.io_pairs):
    input_size, output_size = io_pair
    this_layer = nn.Linear(input_size, output_size, bias=dense_config.bias)
    if dense_config.init_func:
      dense_config.init_func[ind](this_layer.weight, **dense_config.init_kwargs[ind])
    layers.append(this_layer)
    if dense_config.activation and dense_config.activation[ind]:
      layers.append(dense_config.activation[ind]())
    if dense_config.dropout and dense_config.dropout[ind] > 0.0:
      layers.append(nn.Dropout(dense_config.dropout[ind]))
  if dense_config.norm:
    layers.append(nn.LayerNorm(normalized_shape=output_size))
  return nn.Sequential(*layers)

def construct_io_pairs(input_size, output_size, hidden_layers=[]):
  if not isinstance(hidden_layers, list):
    hidden_layers = list(hidden_layers)

  hidden_layers = [input_size] + hidden_layers + [output_size]
  return np.vstack((hidden_layers[:-1], hidden_layers[1:])).T

class MHAScaledDotProduct(nn.Module):
  def __init__(self, d_in, d_out, num_heads, embedding_dim, dropout=0.0, qkv_bias=False):
    super().__init__()

    assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

    self.num_heads = num_heads
    self.embedding_dim = embedding_dim
    self.head_dim = embedding_dim // num_heads
    self.d_out = d_out

    self.qkv = nn.Linear(d_in, 3 * embedding_dim, bias=qkv_bias)
    self.proj = nn.Linear(embedding_dim, d_out)
    self.dropout = dropout

  def forward(self, x):
    batch_size, num_tokens, embed_dim = x.shape

    qkv = self.qkv(x)

    qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

    qkv = qkv.permute(2, 0, 3, 1, 4)

    queries, keys, values = qkv

    use_dropout = 0. if not self.training else self.dropout

    context_vec = nn.functional.scaled_dot_product_attention(
      queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=False)

    context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.embedding_dim)

    context_vec = self.proj(context_vec)

    return context_vec




class ImageSelfAttention(nn.Module):
  def __init__(self, in_channels: int, patch_size: int = 4, num_heads: int = 1, scaling_factor: float = 0.15,
               embedding_dim=None, pos_embedding_type='none', max_num_patches=1000, reduced_channels=None, dropout_p=0.0):
    super(ImageSelfAttention, self).__init__()
    self.in_channels = in_channels
    self.patch_size = patch_size

    self.reduced_channels = reduced_channels
    if self.reduced_channels is None:
      self.embedding_dim = embedding_dim or in_channels * patch_size ** 2
      self.patch_dim = in_channels * patch_size ** 2
    else:
      self.embedding_dim = embedding_dim or reduced_channels * patch_size ** 2
      self.patch_dim = reduced_channels * patch_size ** 2

      self.reduce_channels = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
      self.restore_channels = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)

    self.num_heads = num_heads if num_heads < self.patch_dim else self.patch_dim
    self.mha = MHAScaledDotProduct(d_in=self.patch_dim, d_out=self.patch_dim, num_heads=self.num_heads,
                                   embedding_dim=self.embedding_dim, dropout=dropout_p)

    self.scaling_factor = nn.Parameter(torch.tensor(scaling_factor))

    self.pos_embedding_type = pos_embedding_type

    if pos_embedding_type == 'learnable':
      self.pos_embedding = nn.Parameter(torch.zeros(1, max_num_patches, self.patch_dim))
      nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    if pos_embedding_type == 'learnable_1d':
      self.pos_embedding = nn.Parameter(torch.zeros(1, 1024, 1))
      nn.init.uniform_(self.pos_embedding, -0.02, 0.02)

    elif pos_embedding_type == 'fixed':
      self.register_buffer('pos_embedding', None, persistent=False)

    elif pos_embedding_type == 'index':
      self.register_buffer('pos_embedding', None, persistent=False)

    self.norm1 = nn.LayerNorm(self.patch_dim)
    self.norm2 = nn.LayerNorm(self.patch_dim)

    self.dropout = nn.Dropout(p=dropout_p)

  def forward(self, y):
    batch_size, channels, height, width = y.size()

    pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
    pad_w = (self.patch_size - width % self.patch_size) % self.patch_size

    x = nn.functional.pad(y, (0, pad_w, 0, pad_h))
    _, _, padded_height, padded_width = x.size()
    if self.reduced_channels is not None:
      x = self.reduce_channels(x)
      channels = self.reduced_channels

    x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
    num_patches_h, num_patches_w = x.shape[2], x.shape[3]
    num_patches = num_patches_h * num_patches_w
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(batch_size, -1, channels * self.patch_size ** 2)

    if self.pos_embedding_type == 'learnable':
      x = x + self.pos_embedding[:, :num_patches, :]

    elif self.pos_embedding_type == 'learnable_1d':
      x = x + self.pos_embedding[:, :num_patches, :].expand_as(x)

    elif self.pos_embedding_type == 'fixed':
      if self.pos_embedding is None or self.pos_embedding.size(1) != num_patches:
        self.pos_embedding = self._generate_sinusoidal_embeddings(num_patches, self.patch_dim).to(x.device)
      x = x + self.pos_embedding

    elif self.pos_embedding_type == 'index':
      if self.pos_embedding is None or self.pos_embedding.size(1) != num_patches:
          self.pos_embedding = self._generate_index_embeddings(num_patches).to(x.device)
      x = x + self.pos_embedding

    x = self.norm1(x)

    x = self.mha(x)

    x = self.dropout(x)

    x = self.norm2(x)

    x = x.view(batch_size, num_patches_h, num_patches_w, channels, self.patch_size, self.patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(batch_size, channels, padded_height, padded_width)

    if self.reduced_channels is not None:
      x = self.restore_channels(x)

    x = x[:, :, :height, :width]

    x = y + self.scaling_factor * x

    return x

  @staticmethod
  def _generate_index_embeddings(num_patches):
      indices = torch.arange(num_patches, dtype=torch.float)
      indices = (indices - indices.mean()) / indices.std()
      return indices.view(1, -1, 1)

  @staticmethod
  def _generate_sinusoidal_embeddings(num_patches, dim):
      pos_emb = torch.zeros(num_patches, dim)
      div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(1000.0) / dim))
      pos = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
      pos_emb[:, 0::2] = torch.sin(pos * div_term)
      pos_emb[:, 1::2] = torch.cos(pos * div_term)
      return pos_emb.unsqueeze(0)

class ImageCrossAttention(nn.Module):
  def __init__(self, in_channels: int, patch_size: int = 4, num_heads: int = 1, scaling_factor: float = 0.15,
               embedding_dim=None, pos_embedding_type='none', max_num_patches=1000, rnn_hidden_size=128):
    super(ImageCrossAttention, self).__init__()
    self.in_channels = in_channels
    self.patch_size = patch_size
    self.num_heads = num_heads
    self.embedding_dim = embedding_dim or in_channels * patch_size ** 2
    self.patch_dim = in_channels * patch_size ** 2
    self.rnn_hidden_size = rnn_hidden_size
    self.mha = MHAScaledDotProduct(d_in=self.patch_dim, d_out=self.patch_dim, num_heads=num_heads,
                                   embedding_dim=self.embedding_dim)

    self.hidden_state_proj = nn.Linear(self.rnn_hidden_size, self.patch_dim)

    self.scaling_factor = nn.Parameter(torch.tensor(scaling_factor))

    self.pos_embedding_type = pos_embedding_type

    if pos_embedding_type == 'learnable':
      self.pos_embedding = nn.Parameter(torch.zeros(1, max_num_patches + 1, self.patch_dim))

    elif pos_embedding_type == 'fixed':
      self.register_buffer('pos_embedding', None, persistent=False)

    self.norm1 = nn.LayerNorm(self.patch_dim)
    self.norm2 = nn.LayerNorm(self.patch_dim)

  def _generate_sinusoidal_embeddings(self, num_tokens, dim):
    pos_emb = torch.zeros(num_tokens, dim)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    pos = torch.arange(0, num_tokens, dtype=torch.float).unsqueeze(1)

    pos_emb[:, 0::2] = torch.sin(pos * div_term)
    pos_emb[:, 1::2] = torch.cos(pos * div_term)

    pos_emb[-1, :] = 0.0

    return pos_emb.unsqueeze(0)

  def forward(self, y, hidden_state):
    batch_size, channels, height, width = y.size()

    pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
    pad_w = (self.patch_size - width % self.patch_size) % self.patch_size

    x = nn.functional.pad(y, (0, pad_w, 0, pad_h))
    _, _, padded_height, padded_width = x.size()

    x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
    num_patches_h, num_patches_w = x.shape[2], x.shape[3]
    num_patches = num_patches_h * num_patches_w
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(batch_size, -1, channels * self.patch_size ** 2)

    rnn_hidden_proj = self.hidden_state_proj(hidden_state)
    rnn_hidden_proj = rnn_hidden_proj.unsqueeze(1)

    x = torch.cat((x, rnn_hidden_proj), dim=1)

    if self.pos_embedding_type == 'learnable':
      if self.pos_embedding.size(1) != num_patches + 1:
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, self.patch_dim).to(x.device))
      x = x + self.pos_embedding

    elif self.pos_embedding_type == 'fixed':
      if self.pos_embedding is None or self.pos_embedding.size(1) != num_patches + 1:
        self.pos_embedding = self._generate_sinusoidal_embeddings(num_patches + 1, self.patch_dim).to(x.device)
      x = x + self.pos_embedding

    x = self.norm1(x)

    x = self.mha(x)

    x = self.norm2(x)

    x = x[:, :-1, :].view(batch_size, num_patches_h, num_patches_w, channels, self.patch_size, self.patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(batch_size, channels, padded_height, padded_width)

    x = x[:, :, :height, :width]

    x = y + self.scaling_factor * x

    return x



class ConvEncoder(nn.Module):
  """
  Convolutional Encoder module for building convolutional neural network architectures.
  """

  def __init__(self, config: 'ConvConfig'):
    """
    Initializes the ConvEncoder module.

    Args:
        config (ConvConfig): Configuration for the convolutional encoder.
    """
    super().__init__()
    self.config = config
    if self.config.variational:
      self.config.out_channels *= 2
      self.config.encoder_layers[-1].out_channels *= 2
      self.config.decoder_layers[0].in_channels *= 2
    self.build()

  def build(self):
    """Constructs the neural network architecture."""
    self.model = nn.ModuleList()
    if isinstance(self.config.batch_norm, bool):
      self.config.batch_norm = [self.config.batch_norm] * len(self.config.encoder_layers)

    for i_layer, layer_config in enumerate(self.config.encoder_layers):
      if layer_config.layer_type == 'conv':
        this_layer = nn.Conv2d(**self._get_conv2d_layer_args(i_layer))
        if layer_config.init_func:
          layer_config.init_func(this_layer.weight, **layer_config.init_kwargs)
        self.model.add_module(f'{i_layer}_conv', this_layer)
        if layer_config.dropout > 0:
          self.model.add_module(f'{i_layer}_dropout', nn.Dropout2d(p=layer_config.dropout))
        if self.config.batch_norm[i_layer]:
          self.model.add_module(f'{i_layer}_batchnorm', nn.BatchNorm2d(
            layer_config.out_channels,
            momentum=self.config.batch_norm_momentum,
            track_running_stats=self.config.track_running_stats
          ))
        if i_layer < len(self.config.encoder_layers) - 1 and self.config.encoder_layers[
          i_layer + 1].layer_type == 'maxpool':
          self.model.add_module(f'{i_layer + 1}_maxpool', nn.MaxPool2d(
            **self._get_conv2d_layer_args(i_layer + 1)))

        if layer_config.attention is not None:
          if layer_config.attention == 'ImageSelfAttention':
            self.model.add_module(f'{i_layer}_attention', ImageSelfAttention(in_channels=layer_config.out_channels,
                                                                             **layer_config.attention_kwargs))
          elif issubclass(layer_config.attention, nn.Module):
            self.model.add_module(f'{i_layer}_attention', layer_config.attention(in_channels=layer_config.out_channels, **layer_config.attention_kwargs))

        if layer_config.activation is not None:
          if i_layer != len(self.config.encoder_layers) - 1 or (i_layer == len(self.config.encoder_layers) - 1 and not self.config.fcn_layer_last):
             self.model.add_module(f'{i_layer}_activation', layer_config.activation)

        if hasattr(layer_config, "use_residual") and layer_config.use_residual:
          if i_layer == 0:
            prev_in_channels = self.config.in_channels
            prev_in_x_dim = self.config.x_pixels
            prev_in_y_dim = self.config.y_pixels
          else:
            prev_in_channels = self.config.encoder_layers[i_layer - 1].out_channels
            prev_in_x_dim = self.config.encoder_layers[i_layer - 1].in_x_dim
            prev_in_y_dim = self.config.encoder_layers[i_layer - 1].in_y_dim

          if (
                  prev_in_channels != layer_config.out_channels
                  or prev_in_x_dim != layer_config.in_x_dim
                  or prev_in_y_dim != layer_config.in_y_dim
          ):
            kernel_size = 1
            stride = 1
            padding = 0

            if prev_in_x_dim != layer_config.in_x_dim or prev_in_y_dim != layer_config.in_y_dim:
              stride = max(
                prev_in_x_dim // layer_config.in_x_dim,
                prev_in_y_dim // layer_config.in_y_dim
              )

            residual_conv = nn.Conv2d(
              in_channels=layer_config.in_channels,
              out_channels=layer_config.out_channels,
              kernel_size=kernel_size,
              stride=stride,
              padding=padding,
              bias=False
            )
            self.model.add_module(f'{i_layer}_residual', residual_conv)
          else:
            self.model.add_module(f'{i_layer}_residual', nn.Identity())

      if layer_config.layer_type == 'fcn':
        if self.config.variational and i_layer == len(self.config.encoder_layers) - 1:
          layer_config.out_channels *= 2
        if self.config.encoder_layers[i_layer - 1].layer_type != 'fcn':
          last_conv_size = self.config.encoder_layers[i_layer - 1].in_y_dim * self.config.encoder_layers[
            i_layer - 1].in_x_dim * self.config.encoder_layers[i_layer - 1].out_channels
          this_layer = nn.Linear(last_conv_size, layer_config.out_channels, bias=self.config.bias)
        else:
          this_layer = nn.Linear(layer_config.in_channels, layer_config.out_channels, bias=self.config.bias)
        if layer_config.init_func:
          layer_config.init_func(this_layer.weight, **layer_config.init_kwargs)
        self.model.add_module(f'{i_layer}_fc', this_layer)

        if isinstance(layer_config.activation, Callable):
          if i_layer != len(self.config.encoder_layers) - 1 or (i_layer == len(self.config.encoder_layers) - 1 and not self.config.fcn_layer_last):
            self.model.add_module(f'{i_layer}_activation', layer_config.activation)

        if layer_config.dropout > 0:
          self.model.add_module(f'{i_layer}_dropout', nn.Dropout(p=layer_config.dropout))

  def forward(self, x, p_ind_in=None, target_output_size_in=None):
    original_shape = x.shape
    if len(x.shape) > 4:
      x = x.view(-1, *x.shape[-3:])

    skip_input = x.clone()
    p_ind = []
    target_o_size = []
    skip_input_updated = False

    for layer_name, layer in self.model.named_children():
      if isinstance(layer, nn.MaxPool2d):
        target_o_size.append(x.size())
        x, idx = layer(x)
        p_ind.append(idx)
      elif isinstance(layer, nn.Linear):
        x = x.view(x.size(0), -1)
        x = layer(x)

      elif isinstance(layer, nn.ZeroPad2d):
          skip_input = x.clone()
          skip_input_updated = True
          x = layer(x)

      elif '_residual' in layer_name:
        if isinstance(layer, nn.Identity):
          pass
        else:
          skip_input = layer(skip_input)
        x = x + skip_input
      elif isinstance(layer, nn.Conv2d):
        if not skip_input_updated:
            skip_input = x.clone()
        skip_input_updated = False
        x = layer(x)

      else:
        x = layer(x)

    if len(original_shape) > 4:
      x = x.view(*original_shape[:-3], *x.shape[1:])

    if p_ind_in is not None:
      return x
    else:
      return x, p_ind, target_o_size


  def _get_conv2d_layer_args(self, layer):
    layer_config = self.config.encoder_layers[layer]
    x_pad_0, x_pad_1, y_pad_0, y_pad_1 = layer_config.padding
    if (x_pad_0 == x_pad_1) and (y_pad_0 == y_pad_1):
      padding = (y_pad_0, x_pad_0)
    else:
      self.model.add_module(str('%i_zero_pad' % layer), nn.ZeroPad2d((x_pad_0, x_pad_1, y_pad_0, y_pad_1)))
      padding = 0
    return {
      'in_channels': layer_config.in_channels,
      'out_channels': layer_config.out_channels,
      'kernel_size': layer_config.kernel_size,
      'stride': layer_config.stride,
      'padding': padding,
      'bias': self.config.bias
    }

import torch.nn as nn
import torch.nn.functional as functional
from typing import List

class ConvDecoder(nn.Module):
  """
  Convolutional Decoder module for building convolutional neural network architectures.
  """

  def __init__(self, config: 'ConvConfig'):
    """
    Initializes the ConvDecoder module.

    Args:
        config (ConvConfig): Configuration for the convolutional decoder.
    """
    super().__init__()
    self.config = config
    self.conv_t_pads = {}
    self.model = self.build_layers(config.decoder_layers)

  def build_layers(self, layer_configs: List['LayerConfig']):
    """
    Constructs the layers for the decoder.

    Args:
        layer_configs (List[LayerConfig]): List of layer configurations.

    Returns:
        nn.ModuleList: List of decoder layers.
    """
    layers = nn.ModuleList()
    for i_layer, layer_config in enumerate(layer_configs):
      if layer_config.layer_type == 'fcn':
        if i_layer + 1 < len(layer_configs) and layer_configs[i_layer + 1].layer_type != 'fcn':
          first_conv_size = (layer_configs[i_layer + 1].in_channels *
                             layer_configs[i_layer + 1].in_y_dim *
                             layer_configs[i_layer + 1].in_x_dim)
          this_layer = nn.Linear(layer_config.in_channels, first_conv_size, bias=self.config.bias)
        else:
          this_layer = nn.Linear(layer_config.in_channels, layer_config.out_channels, bias=self.config.bias)
        if layer_config.init_func:
          layer_config.init_func(this_layer.weight, **layer_config.init_kwargs)
        layers.add_module(f'{i_layer}_fc', this_layer)
        if layer_config.activation:
          layers.add_module(f'{i_layer}_activation', layer_config.activation)
        if layer_config.dropout > 0:
          layers.add_module(f'{i_layer}_dropout', nn.Dropout(layer_config.dropout))

      elif layer_config.layer_type == 'convtranspose':
        if i_layer > 0 and layer_configs[i_layer - 1].layer_type in ['maxunpool', 'unpool']:
          if layer_configs[i_layer - 1].layer_type == 'maxunpool':
            layers.add_module(f'{i_layer-1}_unpool', self._get_maxunpool_layer(i_layer - 1))
          elif layer_configs[i_layer - 1].layer_type == 'unpool':
            layers.add_module(f'{i_layer}_unpool', nn.Upsample(size=(layer_config.in_x_dim, layer_config.in_y_dim),
                                      mode=layer_configs[i_layer - 1].kernel_size))

        if hasattr(layer_config, "use_pixelshuffle") and layer_config.use_pixelshuffle:
          if layer_config.stride[0] != layer_config.stride[1]:
            raise ValueError(f"Stride values {layer_config.stride} must be symmetric (equal for x and y).")

          conv_out_channels = layer_config.out_channels * ((layer_config.stride[0]) ** 2)

          if layer_config.padding_type == 'same':
            padding = (layer_config.kernel_size[0] - 1) // 2

          elif layer_config.padding_type == 'valid':
            padding = 0
          else:
            raise ValueError(f"Unknown padding type: {layer_config.padding_type}")

          layers.add_module(f'{i_layer}_conv', nn.Conv2d(
            in_channels=layer_config.in_channels,
            out_channels=conv_out_channels,
            kernel_size=layer_config.kernel_size,
            stride=1,
            padding=padding,
            bias=self.config.bias))

          pixelshuffle_layer = nn.PixelShuffle(upscale_factor=layer_config.stride[0])
          layers.add_module(f'{i_layer}_pixelshuffle', pixelshuffle_layer)

        else:
          layers.add_module(f'{i_layer}_convT', self._get_convt_layer(i_layer))

        if hasattr(layer_config, "use_residual") and layer_config.use_residual:
          if i_layer == len(layer_configs) - 1:
            next_in_channels = self.config.in_channels
            next_in_x_dim = self.config.x_pixels
            next_in_y_dim = self.config.y_pixels
          else:
            next_in_channels = layer_configs[i_layer + 1].in_channels
            next_in_x_dim = layer_configs[i_layer + 1].in_x_dim
            next_in_y_dim = layer_configs[i_layer + 1].in_y_dim

          if (
                  layer_config.in_channels != next_in_channels
                  or layer_config.in_x_dim != next_in_x_dim
                  or layer_config.in_y_dim != next_in_y_dim
          ):
            stride = max(
              next_in_x_dim // layer_config.in_x_dim,
              next_in_y_dim // layer_config.in_y_dim,
            ) if (layer_config.in_x_dim != next_in_x_dim or layer_config.in_y_dim != next_in_y_dim) else 1

            residual_conv = nn.ConvTranspose2d(
              in_channels=layer_config.in_channels,
              out_channels=next_in_channels,
              kernel_size=1 if stride==1 else 2,
              stride=stride,
              padding=0,
              bias=False,
            )
            layers.add_module(f'{i_layer}_residual', residual_conv)
          else:
            layers.add_module(f'{i_layer}_residual', nn.Identity())

        if layer_config.dropout > 0:
          layers.add_module(f'{i_layer}_dropout', nn.Dropout2d(layer_config.dropout))
        if self.config.batch_norm:
          if i_layer != len(layer_configs) - 1 or (i_layer == len(layer_configs) - 1 and not self.config.fcn_layer_last):
            layers.add_module(f'{i_layer}_batchnorm', nn.BatchNorm2d(layer_config.out_channels,
                                       momentum=self.config.batch_norm_momentum,
                                       track_running_stats=self.config.track_running_stats))

        if layer_config.attention is not None:
          if layer_config.attention == 'ImageSelfAttention':
            layers.add_module(f'{i_layer}_attention', ImageSelfAttention(in_channels=layer_config.out_channels,
                                                                         **layer_config.attention_kwargs))
          elif issubclass(layer_config.attention, nn.Module):
            layers.add_module(f'{i_layer}_attention', layer_config.attention(in_channels=layer_config.out_channels, **layer_config.attention_kwargs))

        if layer_config.activation:
          if i_layer != len(layer_configs) - 1 or (i_layer == len(layer_configs) - 1 and not self.config.fcn_layer_last):
            layers.add_module(f'{i_layer}_activation', layer_config.activation)
    return layers

  def forward(self, x, pool_idx=None, target_output_size=None):
    """
    Forward pass of the ConvDecoder.

    Args:
        x: Input data.
        pool_idx: Pooling indices.
        target_output_size: Target output size.

    Returns:
        torch.Tensor: Output tensor.
    """
    original_shape = x.shape
    if len(x.shape) > 4:
      x = x.view(-1, *x.shape[-3:])

    i_layer = 0
    skip_input = x.clone()
    for name_f, layer_f in self.model.named_children():
      if isinstance(layer_f, nn.Linear):
        x = x.view(x.size(0), -1)
        x = layer_f(x)
        if i_layer + 1 < len(self.config.decoder_layers) and \
                self.config.decoder_layers[i_layer + 1].layer_type != 'fcn':
          next_layer = self.config.decoder_layers[i_layer + 1]
          x = x.view(x.size(0),
                     next_layer.in_channels,
                     next_layer.in_y_dim,
                     next_layer.in_x_dim)
        i_layer += 1
      elif isinstance(layer_f, nn.MaxUnpool2d):
        next_layer = self.config.decoder_layers[i_layer + 1]
        idx = pool_idx.pop(-1)
        outsize = target_output_size.pop(-1)
        x = layer_f(x, idx, outsize)
        i_layer += 1

      elif '_residual' in name_f:
        skip_input = layer_f(skip_input)
        x = x + skip_input

      elif isinstance(layer_f, nn.ConvTranspose2d):
        skip_input = x.clone()
        x = layer_f(x)
        i_layer += 1
        if self.conv_t_pads[name_f] is not None:
          x = functional.pad(x, [-i for i in self.conv_t_pads[name_f]])

      else:
        x = layer_f(x)

    if len(original_shape) > 4:
      x = x.view(*original_shape[:-3], *x.shape[1:])

    return x

  def _get_convt_layer(self, layer):
    """
    Get convolution transpose layer and padding layer.

    Args:
        layer (layer): Configuration for the layer.

    Returns:
        tuple: Convolution transpose layer and padding layer.
    """
    layer_config = self.config.decoder_layers[layer]
    in_channels = layer_config.in_channels
    out_channels = layer_config.out_channels
    kernel_size = layer_config.kernel_size
    stride = layer_config.stride

    x_pad_0, x_pad_1, y_pad_0, y_pad_1 = layer_config.padding
    padding_type = layer_config.padding_type

    if padding_type == 'valid':
      input_y = layer_config.in_y_dim
      y_output_padding = layer_config.out_y_dim - (
              (input_y - 1) * stride + kernel_size[0])

      input_x = layer_config.in_x_dim
      x_output_padding = layer_config.out_x_dim - (
              (input_x - 1) * stride + kernel_size[1])

      input_padding = (y_pad_0, x_pad_0)
      output_padding = (y_output_padding, x_output_padding)
      self.conv_t_pads[str('%i_convT' % layer)] = None

    elif padding_type == 'same':
      if (x_pad_0 == x_pad_1) and (y_pad_0 == y_pad_1):
        input_padding = (y_pad_0, x_pad_0)
        output_padding = 0
        self.conv_t_pads[str('%i_convT' % layer)] = None
      else:
        input_padding = 0
        output_padding = 0
        self.conv_t_pads[str('%i_convT' % layer)] = [x_pad_0, x_pad_1, y_pad_0, y_pad_1]
    else:
      raise ValueError(f'"{padding_type}" is not a valid padding type')

    this_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=input_padding, output_padding=output_padding, bias=self.config.bias)

    if layer_config.init_func:
      layer_config.init_func(this_layer.weight, **layer_config.init_kwargs)
    return this_layer











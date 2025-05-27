from dataclasses import dataclass, field
import numpy as np
import torch.nn as nn
from typing import Callable, List, Union, Tuple, Optional


_LeakyReLU_ns = 0.1

@dataclass
class DenseConfig:
  """Config for construction dense (fully-connected) network."""
  input_size: int
  output_size: int
  hidden_layers: list = field(default_factory=list)
  activation: Union[Callable, list] = None
  init_func: Union[Callable, list] = nn.init.xavier_uniform_
  init_kwargs: Union[dict, list] = field(default_factory=dict)
  output_activation: Callable = None
  bias: bool = True

  def __post_init__(self):
    self.io_pairs = construct_io_pairs(self.input_size, self.output_size, self.hidden_layers)
    if self.activation and isinstance(self.activation, Callable):
      self.activation = [self.activation for _ in range(len(self.io_pairs) - 1)]
      self.activation.append(self.output_activation)

    if isinstance(self.init_func, Callable):
      self.init_func = [self.init_func for _ in range(len(self.io_pairs))]
    if isinstance(self.init_kwargs, dict):
      self.init_kwargs = [self.init_kwargs for _ in range(len(self.io_pairs))]

  def update_input_output_dims(self, input_size : int, output_size : int):
    self.input_size, self.output_size = input_size, output_size
    self.__post_init__()
    return self

def dense_layers(dense_config : DenseConfig):
  """Creates dense MLP with optional activations after each layer."""
  layers = []
  for ind, io_pair in enumerate(dense_config.io_pairs):
    input_size, output_size = io_pair
    this_layer = nn.Linear(input_size, output_size, bias=dense_config.bias)
    dense_config.init_func[ind](this_layer.weight, **dense_config.init_kwargs[ind])
    layers.append(this_layer)
    if dense_config.activation and dense_config.activation[ind]:
      layers.append(dense_config.activation[ind]())
  return nn.Sequential(*layers)

def construct_io_pairs(input_size, output_size, hidden_layers=[]):
  """
  Args:
    input_size: int. Network input size.
    output_size: int. Network output size.
    hidden_layers: list of ints. Dimension of hidden layers. Default [].

  Returns:
    Layer input-output pairs as np.ndarray of size (num_layers, 2). Number of
    layers corresponds to len(hidden_layers) + 1.
  """
  if not isinstance(hidden_layers, list):
    hidden_layers = list(hidden_layers)

  hidden_layers = [input_size] + hidden_layers + [output_size]
  return np.vstack((hidden_layers[:-1], hidden_layers[1:])).T

@dataclass
class LayerConfig:
  in_channels: int
  out_channels: int
  layer_type: str
  kernel_size: Union[int, Tuple[int, int], str] = None
  stride: Union[int, Tuple[int, int]] = None
  padding: Optional[Union[int, Tuple[int, int, int, int]]] = None
  activation: Optional[Callable] = None
  attention: str = None
  attention_kwargs: dict = field(default_factory=dict)
  dropout: Optional[float] = 0.0
  padding_type: Optional[str] = None
  in_x_dim: Optional[int] = None
  in_y_dim: Optional[int] = None
  init_func: Callable = None
  init_kwargs: dict = field(default_factory=dict)
  use_residual: bool = False
  use_pixelshuffle: bool = False

  def update_in_dim(self, in_channels: int):
    self.in_channels = in_channels
    return self

  def update_out_dim(self, out_channels: int):
    self.out_channels = out_channels
    return self

@dataclass
class ConvConfig:
  in_channels: int
  out_channels: int
  x_pixels: int
  y_pixels: int
  encoder_layers: List[LayerConfig] = field(default_factory=list)
  decoder_layers: List[LayerConfig] = field(default_factory=list)
  bias: bool = True,
  batch_norm: Union[bool, list] = False
  batch_norm_momentum: float = 0.1
  track_running_stats: bool = True
  activation: Optional[str] = None
  fc_activation: Optional[str] = None
  encoder_drop_out: List[float] = field(default_factory=list)
  symmetric: bool = False
  fcn_layer_last: bool = False
  variational: bool = False
  agnostic_decoder: bool = False
  unpool_method: str = None

  def __post_init__(self):
    input_x, input_y = self.x_pixels, self.y_pixels

    if len(self.encoder_layers) > 0:
      self._update_model_configs(self.encoder_layers, self.x_pixels, self.y_pixels)

      if self.symmetric:
        self.decoder_layers = [
          LayerConfig(
            in_channels=layer.out_channels,
            out_channels=layer.in_channels,
            layer_type=(
              'fcn' if layer.layer_type == 'fcn' else 'maxunpool' if layer.layer_type == 'maxpool' and not self.agnostic_decoder else 'unpool' if layer.layer_type == 'maxpool' and self.agnostic_decoder else 'convtranspose' if layer.layer_type == 'conv' else 'unpool' if layer.layer_type == 'pool' else None),
            kernel_size=(
              self.unpool_method if layer.layer_type == 'maxpool' and self.agnostic_decoder else layer.kernel_size),
            stride=layer.stride,
            padding=layer.padding,
            activation=layer.activation,
            attention=layer.attention,
            attention_kwargs=layer.attention_kwargs,
            dropout=layer.dropout,
            padding_type=layer.padding_type,
            in_x_dim=layer.in_x_dim,
            in_y_dim=layer.in_y_dim,
            use_residual=layer.use_residual,
            use_pixelshuffle=layer.use_pixelshuffle,

        ) for layer in reversed(self.encoder_layers)
        ]
        if self.fcn_layer_last:
          self.decoder_layers[-1].activation = nn.Sigmoid()

    elif len(self.decoder_layers) > 0:
      self._update_model_configs(self.decoder_layers, self.x_pixels, self.y_pixels)
      self.encoder_layers = self.decoder_layers
    return self

  def _update_model_configs(self, layers, input_x, input_y):
    for layer in layers:
      if layer.layer_type == 'fcn':
        if self.fc_activation == 'relu':
          layer.activation = nn.ReLU()
        elif self.fc_activation == 'leakyrelu':
          layer.activation = nn.LeakyReLU(_LeakyReLU_ns)
        elif self.fc_activation == 'sigmoid':
          layer.activation = nn.Sigmoid()
        layer.padding = None
        layer.padding_type = None
        layer.in_x_dim = 1
        layer.in_y_dim = 1
      else:
        layer.padding_type = 'same'
        if isinstance(layer.stride, int):
          layer.stride = (layer.stride, layer.stride)
        if isinstance(layer.kernel_size, int):
          layer.kernel_size = (layer.kernel_size, layer.kernel_size)

        output_x = (input_x // layer.stride[0])
        output_y = (input_y // layer.stride[1])

        if not isinstance(layer.kernel_size, str):

            pad_x = (output_x - 1) * layer.stride[0] - input_x + layer.kernel_size[0]
            pad_y = (output_y - 1) * layer.stride[1] - input_y + layer.kernel_size[1]

            dim_pad_x = max(pad_x // 2, 0)
            dim_pad_y = max(pad_y // 2, 0)
            pad_x_plus = pad_x % 2
            pad_y_plus = pad_y % 2
            layer.padding = (dim_pad_x, dim_pad_x + pad_x_plus, dim_pad_y, dim_pad_y + pad_y_plus)
        else:
            layer.padding = (0, 0, 0, 0)
        layer.in_x_dim = output_x
        layer.in_y_dim = output_y

        if self.activation == 'relu':
          layer.activation = nn.ReLU()
        elif self.activation == 'leaky_relu':
          layer.activation = nn.LeakyReLU(_LeakyReLU_ns)
        elif self.activation == 'sigmoid':
          layer.activation = nn.Sigmoid()

        if layer.init_func and self.activation:
          layer.init_kwargs = nn.init.calculate_gain(self.activation, 0.05)
        input_x, input_y = output_x, output_y

  def update_input_output_dims(self, in_channels: int, out_channels: int, base_x_pixels: int, base_y_pixels: int):
    if (self.x_pixels == 1 and self.y_pixels == 1) or (len(self.decoder_layers)==1 and self.decoder_layers[0].layer_type=='fcn'):
      self.encoder_layers[0].update_in_dim(in_channels)
      self.encoder_layers[-1].update_out_dim(out_channels * base_x_pixels * base_y_pixels)
      self.decoder_layers[0].update_in_dim(out_channels * base_x_pixels * base_y_pixels)
      self.decoder_layers[-1].update_out_dim(in_channels)
      self.in_channels = in_channels
      self.out_channels = out_channels * base_x_pixels * base_y_pixels
    else:
      self.encoder_layers[0].update_in_dim(in_channels)
      self.encoder_layers[-1].update_out_dim(out_channels)
      self.decoder_layers[0].update_in_dim(out_channels)
      self.decoder_layers[-1].update_out_dim(in_channels)
      self.in_channels = in_channels
      self.out_channels = out_channels

    return self

def initialize_conv_kernels(module):
    """
    Initializes Conv2d and ConvTranspose2d layers.
    """
    for submodule in module.modules():
        if isinstance(submodule, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(submodule.weight)

            if submodule.bias is not None:
                submodule.bias.data.fill_(0)

def get_conv_config(config_dict, decoder=False):
  encoder_layers = [
    LayerConfig(
      layer_type=config_dict['encoder_layer_type'][i],
      in_channels=config_dict['channels'][i],
      out_channels=config_dict['channels'][i + 1],
      kernel_size=config_dict['kernel_size'][i],
      stride=config_dict['stride'][i],
      attention=config_dict['attention'][i] if 'attention' in config_dict else None,
      attention_kwargs=config_dict['attention_kwargs'][i] if 'attention_kwargs' in config_dict else None,
      dropout=config_dict['encoder_drop_out'][i] if 'encoder_drop_out' in config_dict else False,
      use_residual=config_dict['use_residual'][i] if 'use_residual' in config_dict else False,
      use_pixelshuffle=config_dict['use_pixelshuffle'][i] if 'use_pixelshuffle' in config_dict else False,

    ) for i in range(len(config_dict['encoder_layer_type']))
  ]

  if decoder:
    conv_config = ConvConfig(
      in_channels=config_dict['channels'][0],
      out_channels=config_dict['channels'][-1],
      x_pixels=config_dict['x_pixels'],
      y_pixels=config_dict['y_pixels'],
      decoder_layers=encoder_layers,
      batch_norm=config_dict['batch_norm'],
      batch_norm_momentum=0.1,
      track_running_stats=True,
      activation=config_dict['activation'],
      fc_activation=config_dict['fc_activation'],
      encoder_drop_out=config_dict['encoder_drop_out'],
      symmetric=config_dict['symmetric'],
      fcn_layer_last=config_dict['last_FF'],
      agnostic_decoder=False
    )
  else:
    conv_config = ConvConfig(
      in_channels=config_dict['channels'][0],
      out_channels=config_dict['channels'][-1],
      x_pixels=config_dict['x_pixels'],
      y_pixels=config_dict['y_pixels'],
      encoder_layers=encoder_layers,
      batch_norm=config_dict['batch_norm'],
      batch_norm_momentum=0.1,
      track_running_stats=True,
      activation=config_dict['activation'],
      fc_activation=config_dict['fc_activation'],
      encoder_drop_out=config_dict['encoder_drop_out'],
      symmetric=config_dict['symmetric'],
      fcn_layer_last=config_dict['last_FF'],
      agnostic_decoder=False
    )

  return conv_config

def get_attention_identifier(config_dict):
  attention_layers = config_dict['attention']
  num_layers = len(attention_layers)
  identifier = ""

  for attention_module in attention_layers:
    if attention_module is None:
      identifier += "0"
    else:
      identifier += "1"

  if any(attention_layers):
    last_attention = [module for module in attention_layers if module is not None][-1]
    attention_type = last_attention if isinstance(last_attention, str) else "GenericAttention"
    identifier = f"{attention_type}_{identifier}"
  else:
    identifier = f"{identifier}"

  return identifier


def create_general_model_configs(
    patch_size, num_heads, embedding_dim, pos_embedding, nx, nz, yTrain_shape, ds,
    A_kernel_size, K_kernel_size, Cy_kernel_size, Cz_kernel_size,
    K_channel_sizes, Cy_channel_sizes, Cz_channel_sizes,
    K_strides=None, Cy_strides=None, Cz_strides=None,
    K_use_residuals=None, Cy_use_residuals=None, Cz_use_residuals=None,
    K_use_pixelshuffle=None, Cy_use_pixelshuffle=None, Cz_use_pixelshuffle=None,
    K_encoder_layer_type=None, Cy_encoder_layer_type=None, Cz_encoder_layer_type=None,
        Cz_dropout=None, Cz_batch_norm=None, Cz_activation=None, unified_K=False, reduce_att_ch=None, att_dropout=0.0,
):
    attention_kwargs = {
        "patch_size": patch_size,
        "num_heads": num_heads,
        "scaling_factor": 0.2,
        "embedding_dim": embedding_dim,
        "pos_embedding_type": pos_embedding,
        "max_num_patches": 128,
        "reduced_channels": reduce_att_ch,
        "dropout_p":att_dropout,
    }

    if K_encoder_layer_type is not None:
        ds = len(K_encoder_layer_type)

    if isinstance(K_kernel_size, int):
        K_kernel_size = ds * [K_kernel_size]
    else:
        ds = len(K_kernel_size)

    if isinstance(Cy_kernel_size, int):
        Cy_kernel_size = ds * [Cy_kernel_size]
    else:
        ds = len(Cy_kernel_size)

    def standardize_stride(stride, layers, default=2):
        return layers * [default] if stride is None else stride

    def ensure_list_Cz(channel_sizes, default_length):
        if isinstance(channel_sizes, int):
            return (default_length - 1) * [channel_sizes]
        return channel_sizes

    def ensure_list(channel_sizes, default_length, unified, last_channel):
        if unified:
            if isinstance(channel_sizes, int):
                return (default_length) * [channel_sizes]
        else:
            if isinstance(channel_sizes, int):
                return (default_length - 1) * [channel_sizes] + [last_channel]
        return channel_sizes

    K_channel_sizes = ensure_list(K_channel_sizes, ds, unified_K, nx)

    KCy_config_dict = {
        'encoder_layer_type': (ds * ['conv'] if K_encoder_layer_type is None else K_encoder_layer_type),
        'kernel_size': K_kernel_size,
        'stride': standardize_stride(K_strides, ds),
        'x_pixels': yTrain_shape[2],
        'y_pixels': yTrain_shape[3],
        'channels': [yTrain_shape[1]] + K_channel_sizes,
        'batch_norm': True,
        'activation': 'leaky_relu',
        'fc_activation': 'relu',
        'encoder_drop_out': ds * [0.0],
        'symmetric': True,
        'last_FF': True,
        'attention': [None] + (ds - 1) * [None],
        'attention_kwargs': [{}] + (ds - 1) * [{}],
        'use_residual': ds * [False] if K_use_residuals is None else K_use_residuals,
        'use_pixelshuffle': ds * [False] if K_use_pixelshuffle is None else K_use_pixelshuffle,
    }
    K_config = get_conv_config(KCy_config_dict)

    if Cy_encoder_layer_type is not None:
        ds = len(Cy_encoder_layer_type)

    Cy_channel_sizes = ensure_list(Cy_channel_sizes, ds, unified_K, nx)
    Cy_config_dict = {
        'encoder_layer_type': (ds * ['conv'] if Cy_encoder_layer_type is None else Cy_encoder_layer_type),
        'kernel_size': Cy_kernel_size,
        'stride': standardize_stride(Cy_strides, ds),
        'x_pixels': yTrain_shape[2],
        'y_pixels': yTrain_shape[3],
        'channels': [yTrain_shape[1]] + Cy_channel_sizes,
        'batch_norm': True,
        'activation': 'leaky_relu',
        'fc_activation': 'relu',
        'encoder_drop_out': ds * [0.0],
        'symmetric': True,
        'last_FF': True,
        'attention': [None] + (ds - 1) * [None],
        'attention_kwargs': [{}] + (ds - 1) * [{}],
        'use_residual': ds * [False] if Cy_use_residuals is None else Cy_use_residuals,
        'use_pixelshuffle': ds * [False] if Cy_use_pixelshuffle is None else Cy_use_pixelshuffle,
    }
    Cy_config = get_conv_config(Cy_config_dict)

    state_x = K_config.encoder_layers[-1].in_x_dim
    state_y = K_config.encoder_layers[-1].in_y_dim

    down_samp = int(np.sqrt(yTrain_shape[2] // state_y))

    if isinstance(Cz_strides, list):
        Cz_ds = len(Cz_strides) + 1
        if isinstance(Cz_kernel_size, int):
            Cz_kernel_size = (Cz_ds-1) * [Cz_kernel_size]
        Cz_channel_sizes = ensure_list_Cz(Cz_channel_sizes, Cz_ds)

        Cz_config_dict = {
            'encoder_layer_type': ((Cz_ds-1) * ['conv'] if Cz_encoder_layer_type is None else Cz_encoder_layer_type),
            'kernel_size': Cz_kernel_size + [None],
            'stride': Cz_strides + [None],
            'x_pixels': state_x,
            'y_pixels': state_y,
            'channels': [nx] + Cz_channel_sizes + [nz],
            'batch_norm': Cz_ds * [False] if Cz_batch_norm is None else Cz_batch_norm,
            'activation': 'leaky_relu' if Cz_activation is None else Cz_activation,
            'fc_activation': None,
            'encoder_drop_out': Cz_ds * [0.0] if Cz_dropout is None else Cz_dropout,
            'symmetric': True,
            'last_FF': False,
            'attention': Cz_ds * [None],
            'attention_kwargs': Cz_ds * [{}],
            'use_residual': Cz_ds * [False] if Cz_use_residuals is None else Cz_use_residuals,
            'use_pixelshuffle': Cz_ds * [False] if Cz_use_pixelshuffle is None else Cz_use_pixelshuffle,
        }

    else:
        Cz_ds = 6 - down_samp
        Cz_channel_sizes = ensure_list_Cz(Cz_channel_sizes, 6 - down_samp)

        if isinstance(Cz_kernel_size, int):
            Cz_kernel_size = (Cz_ds-1) * [Cz_kernel_size]

        Cz_config_dict = {
            'encoder_layer_type': ((Cz_ds-1) * ['conv'] if Cz_encoder_layer_type is None else Cz_encoder_layer_type),
            'kernel_size': Cz_kernel_size + [None],
            'stride': standardize_stride(Cz_strides, Cz_ds-1) + [None],
            'x_pixels': state_x,
            'y_pixels': state_y,
            'channels': [nx] + Cz_channel_sizes + [nz],
            'batch_norm': Cz_ds * [False] if Cz_batch_norm is None else Cz_batch_norm,
            'activation': 'leaky_relu' if Cz_activation is None else Cz_activation,
            'fc_activation': None,
            'encoder_drop_out': Cz_ds * [0.0] if Cz_dropout is None else Cz_dropout,
            'symmetric': True,
            'last_FF': False,
            'attention': Cz_ds * [None],
            'attention_kwargs': Cz_ds * [{}],
            'use_residual': Cz_ds * [False] if Cz_use_residuals is None else Cz_use_residuals,
            'use_pixelshuffle': Cz_ds * [False] if Cz_use_pixelshuffle is None else Cz_use_pixelshuffle,
        }

    Cz_config = get_conv_config(Cz_config_dict, decoder=True)

    A_config_dict = {
        'encoder_layer_type': ['conv'],
        'kernel_size': [A_kernel_size],
        'stride': [1],
        'x_pixels': state_x,
        'y_pixels': state_y,
        'channels': [nx, nx],
        'batch_norm': False,
        'activation': None,
        'fc_activation': 'relu',
        'encoder_drop_out': [0.0],
        'symmetric': True,
        'last_FF': True,
        'attention': ['ImageSelfAttention' if embedding_dim > 0 else None],
        'attention_kwargs': [attention_kwargs],
        'use_residual': [None],
        'use_pixelshuffle': [None],
    }
    A_config = get_conv_config(A_config_dict)

    A_identifier = get_attention_identifier(A_config_dict)
    K_identifier = get_attention_identifier(KCy_config_dict)
    Cy_identifier = get_attention_identifier(Cy_config_dict)

    return A_config, K_config, Cy_config, Cz_config, A_identifier, K_identifier, Cy_identifier
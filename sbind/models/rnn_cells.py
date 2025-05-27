import copy
from sbind.models import models
from sbind.models import model_helpers
import torch
import torch.nn as nn

class ConvRNNCell(nn.Module):
  def __init__(self, A_conv_config,
               device: str = 'cuda',
               step_ahead_prediction: bool=False,
               cell_activation=nn.Identity, stateful=False, Af_ks = None):
    super(ConvRNNCell, self).__init__()

    self.unified_A = True if A_conv_config.in_channels != A_conv_config.out_channels else False
    self.state_size = A_conv_config.out_channels
    self.x_pixels = A_conv_config.x_pixels
    self.y_pixels = A_conv_config.y_pixels
    self.stateful = stateful
    self.A = models.ConvEncoder(A_conv_config)
    self.cell_activation = cell_activation
    self.state = None
    self.hidden_state = None
    self.device = torch.device(device)

    Af_conv_config = copy.deepcopy(A_conv_config)
    if Af_ks is not None:
      Af_conv_config.encoder_layers[0].kernel_size = (Af_ks,Af_ks)
      padding = (Af_ks - 1) // 2
      Af_conv_config.encoder_layers[0].padding = (padding,padding,padding,padding)
    self.A_f = models.ConvEncoder(Af_conv_config)

  def forward(self, y):
    if self.state is None:
      self.set_state(nn.init.zeros_(torch.empty(y.shape[0], self.state_size, self.y_pixels, self.x_pixels)).to(self.device))

    if self.unified_A:
      direct_state = self.cell_activation()(self.A(torch.cat((self.state, y), -3))[0])
    else:
      direct_state = self.cell_activation()(self.A(self.state)[0] + y)

    self.state = direct_state
    return self.state, self.hidden_state

  def set_state(self, new_state):
    if new_state.shape[1] != self.state_size or (len(new_state.shape) > 2 and (new_state.shape[2] != self.y_pixels or new_state.shape[3] != self.x_pixels)):
      raise ValueError(f'Incorrect state size/height/width: {new_state.shape[0]} != {self.state_size} or {new_state.shape[2]} != {self.y_pixels}  or {new_state.shape[3]} != {self.x_pixels}')
    self.state = new_state.detach().to(self.device)

  def get_state(self):
    return self.state

class ConvGRUAttCell(nn.Module):
    def __init__(self, A_conv_config, device='cuda', attention_module=nn.Sigmoid, cell_activation=[nn.Identity, nn.Identity], stateful=False,attention_module2=None):
        super(ConvGRUAttCell, self).__init__()

        self.state_size = A_conv_config.out_channels
        self.x_pixels = A_conv_config.x_pixels
        self.y_pixels = A_conv_config.y_pixels
        self.stateful = stateful

        self.cell_activation = cell_activation[0]
        self.cell_activation_c = cell_activation[1]

        self.device = torch.device(device)

        self.A = models.ConvEncoder(A_conv_config)
        self.A_c = models.ConvEncoder(A_conv_config)
        self.hidden_rnn = False

        if attention_module == nn.Sigmoid:
            self.attention_module = attention_module
        else:
            self.attention_module = attention_module(in_channels=A_conv_config.out_channels)
        if attention_module2 is not None:
            self.attention_module2 = attention_module2(in_channels=A_conv_config.out_channels)
        else:
            self.attention_module2 = None
        self.reset_state()

    def forward(self, y):
        if self.state is None or self.cell_state is None:
            self.set_state(
                nn.init.zeros_(torch.empty(y.shape[0], self.state_size, self.y_pixels, self.x_pixels)).to(self.device),
                nn.init.zeros_(torch.empty(y.shape[0], self.state_size, self.y_pixels, self.x_pixels)).to(self.device)
            )

        c_t = self.cell_activation_c()(self.A_c(torch.cat((self.cell_state, y), dim=1))[0])
        if self.attention_module2 is not None:
            c_t = self.attention_module2(c_t)

        if self.attention_module == nn.Sigmoid:
            z_t = self.attention_module()(c_t)
        else:
            z_t = self.attention_module(c_t)

        h_tilde_t = self.cell_activation()(self.A(torch.cat((self.state, y), dim=1))[0])

        self.state = (1 - z_t) * self.state + z_t * h_tilde_t
        self.cell_state = c_t

        return self.state, self.cell_state

    def set_state(self, new_state, new_cell_state=None):
        if new_cell_state is None:
            new_cell_state = new_state
        if new_state.shape[1] != self.state_size or new_state.shape[2] != self.y_pixels or new_state.shape[3] != self.x_pixels:
            raise ValueError(f'Incorrect state size/height/width: {new_state.shape[0]} != {self.state_size} or {new_state.shape[2]} != {self.y_pixels}  or {new_state.shape[3]} != {self.x_pixels}')
        if new_cell_state.shape[1] != self.state_size or new_cell_state.shape[2] != self.y_pixels or new_cell_state.shape[3] != self.x_pixels:
            raise ValueError(f'Incorrect cell state size/height/width: {new_cell_state.shape[0]} != {self.state_size} or {new_cell_state.shape[2]} != {self.y_pixels}  or {new_cell_state.shape[3]} != {self.x_pixels}')
        self.state = new_state.detach().to(self.device)
        self.cell_state = new_cell_state.detach().to(self.device)

    def reset_state(self):
        self.state = None
        self.cell_state = None

    def get_state(self):
        return self.state, self.cell_state

class ConvLSTMCell(nn.Module):
    def __init__(self, A_conv_config, W_config=None,
                 device='cuda', cell_activation=nn.Identity, stateful=False):
        super(ConvLSTMCell, self).__init__()

        self.state_size = A_conv_config.out_channels // 4
        self.x_pixels = A_conv_config.x_pixels
        self.y_pixels = A_conv_config.y_pixels
        self.stateful = stateful
        self.cell_activation = cell_activation
        self.device = torch.device(device)

        self.A = models.ConvEncoder(A_conv_config)
        self.reset_state()
        self.hidden_rnn = False

    def forward(self, y):
        if self.state is None or self.cell_state is None:
            self.set_state(
                nn.init.zeros_(torch.empty(y.shape[0], self.state_size, self.y_pixels, self.x_pixels)).to(self.device),
                nn.init.zeros_(torch.empty(y.shape[0], self.state_size, self.y_pixels, self.x_pixels)).to(self.device)
            )

        i_t, f_t, g_t, o_t = self.A(torch.cat((self.state, y), dim=1))[0].chunk(4, dim=1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        self.cell_state = f_t * self.cell_state + i_t * g_t
        self.state = o_t * torch.tanh(self.cell_state)

        return self.state, self.cell_state

    def set_state(self, new_state, new_cell_state=None):
        if new_cell_state is None:
            new_cell_state = new_state
        if new_state.shape[1] != self.state_size or new_state.shape[2] != self.y_pixels or new_state.shape[3] != self.x_pixels:
            raise ValueError(f'Incorrect state size/height/width: {new_state.shape[0]} != {self.state_size} or {new_state.shape[2]} != {self.y_pixels}  or {new_state.shape[3]} != {self.x_pixels}')
        if new_cell_state.shape[1] != self.state_size or new_cell_state.shape[2] != self.y_pixels or new_cell_state.shape[3] != self.x_pixels:
            raise ValueError(f'Incorrect cell state size/height/width: {new_cell_state.shape[0]} != {self.state_size} or {new_cell_state.shape[2]} != {self.y_pixels}  or {new_cell_state.shape[3]} != {self.x_pixels}')
        self.state = new_state.detach().to(self.device)
        self.cell_state = new_cell_state.detach().to(self.device)

    def reset_state(self):
        self.state = None
        self.cell_state = None

    def get_state(self):
        return self.state, self.cell_state

class LSTMCell(nn.Module):
  def __init__(self, A_dense_config : model_helpers.DenseConfig,
               device : str = 'cuda', cell_activation=nn.Identity, stateful=False, state_norm=None):
    super(LSTMCell, self).__init__()

    self.state_size = int(A_dense_config.output_size / 4)
    self.stateful = stateful
    self.A = models.MLP(A_dense_config)
    A_dense_config.update_input_output_dims(self.state_size, self.state_size)
    self.A_f = models.MLP(A_dense_config)
    self.cell_activation = cell_activation
    self.state = None
    self.device = torch.device(device)

    self.state_norm = state_norm
    if self.state_norm == 'bn':
      self.norm_layer = nn.BatchNorm1d(num_features=self.state_size)
    elif self.state_norm == 'ln':
      self.norm_layer = nn.LayerNorm(normalized_shape=self.state_size)
    else:
      self.norm_layer = None

    self.device = torch.device(device)

    self.state = None
    self.cell_state = None

  def forward(self, y):
    if self.state is None or self.cell_state is None:
      self.set_state(nn.init.zeros_(torch.empty(y.shape[0], self.state_size)).to(self.device),
                     nn.init.zeros_(torch.empty(y.shape[0], self.state_size)).to(self.device))

    i_t, f_t, g_t, o_t = self.A(torch.cat((self.state, y), dim=1)).chunk(4, dim=1)
    i_t = torch.sigmoid(i_t)
    f_t = torch.sigmoid(f_t)
    g_t = torch.tanh(g_t)
    o_t = torch.sigmoid(o_t)

    self.cell_state = f_t * self.cell_state + i_t * g_t
    self.state = o_t * torch.tanh(self.cell_state)

    return self.state

  def set_state(self, new_state, new_cell_state=None):
    if new_cell_state is None:
      new_cell_state = new_state
    if new_state.shape[1] != self.state_size:
      raise ValueError(f'Incorrect hidden state size: {new_state.shape[1]} != {self.state_size}')
    if new_cell_state.shape[1] != self.state_size:
      raise ValueError(f'Incorrect cell state size: {new_cell_state.shape[1]} != {self.state_size}')
    self.state = new_state.detach().to(self.device)
    self.cell_state = new_cell_state.detach().to(self.device)

  def reset_state(self):
    self.state = None
    self.cell_state = None
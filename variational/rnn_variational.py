import math

import torch
from torch.distributions.kl import kl_divergence
from torch.nn import GRUCell, ModuleList, Sequential, init, Module
from torch.nn.parameter import Parameter

from variational import LinearFlipout

from .posteriors import PosteriorNormal
from .priors import PriorLaplace, PriorNormal
from .utils import random_rademacher
from .variational import GaussianVariationalModule


class _RNNCellBasePatwhise(GaussianVariationalModule):

    def __init__(self, input_size, hidden_size, bias, num_chunks):
        super(_RNNCellBasePatwhise, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # input to hidden
        self.weight_ih_loc = Parameter(torch.Tensor(
            num_chunks * hidden_size, input_size))
        self.weight_ih_scale = Parameter(torch.Tensor(
            num_chunks * hidden_size, input_size))
        self.weight_ih_posterior = PosteriorNormal(
            self.weight_ih_loc, self.weight_ih_scale)
        self.weight_ih_prior = PriorNormal(0, 1)
        # hidden to hidden
        self.weight_hh_loc = Parameter(torch.Tensor(
            num_chunks * hidden_size, hidden_size))
        self.weight_hh_scale = Parameter(torch.Tensor(
            num_chunks * hidden_size, hidden_size))
        self.weight_hh_posterior = PosteriorNormal(
            self.weight_hh_loc, self.weight_hh_scale)
        self.weight_hh_prior = PriorNormal(0, 1)
        self.use_bias = bias
        if self.use_bias:
            # input to hidden
            self.bias_ih_loc = Parameter(
                torch.Tensor(num_chunks * hidden_size))
            self.bias_ih_scale = Parameter(
                torch.Tensor(num_chunks * hidden_size))
            self.bias_ih_posterior = PosteriorNormal(
                self.bias_ih_loc, self.bias_ih_scale)
            self.bias_ih_prior = PriorNormal(0, 1)
            # hidden to hidden
            self.bias_hh_loc = Parameter(
                torch.Tensor(num_chunks * hidden_size))
            self.bias_hh_scale = Parameter(
                torch.Tensor(num_chunks * hidden_size))
            self.bias_hh_posterior = PosteriorNormal(
                self.bias_hh_loc, self.bias_hh_scale)
            self.bias_hh_prior = PriorNormal(0, 1)

        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    @property
    def hidden_weight_perturbation(self):
        eps = _standard_normal(shape=self.weight_posterior_logscale.shape,
                               dtype=self.weight_posterior_logscale.dtype,
                               device=self.weight_posterior_logscale.device)
        return eps * softplus(self.weight_posterior_logscale)

    @property
    def hidden_weight_posterior_sample(self):
        return self.weight_posterior_mean + self.weight_perturbation

    @property
    def hidden_bias_posterior_sample(self):
        if not self.use_bias:
            return None
        eps = _standard_normal(shape=self.bias_posterior_logscale.shape,
                               dtype=self.bias_posterior_logscale.dtype,
                               device=self.bias_posterior_logscale.device)
        return self.bias_posterior_mean + softplus(self.bias_posterior_logscale) * eps
    
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, state, hidden_label=''):
        if input.size(0) != state.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, state.size(0)))

        if state.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, state.size(1), self.hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


    def kl_loss(self, scaling=None):
        weight_posterior_loss = kl_normal_normal(self.weight_posterior_mean,
                                                 softplus(
                                                     self.weight_posterior_logscale),
                                                 self.weight_prior_mean,
                                                 self.weight_prior_scale)
        total_loss = weight_posterior_loss.sum()
        hidden_weight_posterior_loss = kl_normal_normal(self.hidden_weight_posterior_mean,
                                                 softplus(
                                                     self.hidden_weight_posterior_logscale),
                                                 self.hidden_weight_prior_mean,
                                                 self.hidden_weight_prior_scale)
        total_loss += hidden_weight_posterior_loss.sum()
        if self.use_bias:
            bias_loss = kl_normal_normal(self.bias_posterior_mean,
                                         softplus(
                                             self.bias_posterior_logscale),
                                         self.bias_prior_mean,
                                         self.bias_prior_scale)
            total_loss += bias_loss.sum()
        return total_loss

    def kl_loss(self):
        weight_ih_loss = kl_divergence(self.weight_ih_posterior,
                                       self.weight_ih_prior)
        weight_hh_loss = kl_divergence(self.weight_hh_posterior,
                                       self.weight_hh_prior)
        total_loss = weight_ih_loss.sum() + weight_hh_loss.sum()
        if self.use_bias:
            bias_ih_loss = kl_divergence(self.bias_ih_posterior,
                                         self.bias_ih_prior)
            bias_hh_loss = kl_divergence(self.bias_hh_posterior,
                                         self.bias_hh_prior)
            total_loss += bias_ih_loss.sum() + bias_hh_loss.sum()
        return total_loss


class GRUCellPathwise(_RNNCellBasePatwhise):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCellPathwise, self).__init__(
            input_size, hidden_size, bias, num_chunks=3)

    def forward(self, input, state=None):
        self.check_forward_input(input)
        if state is None:
            state = input.new_zeros(input.size(0),
                                    self.hidden_size,
                                    requires_grad=False)
        self.check_forward_hidden(input, state)
        output = torch.gru_cell(input, state,
                                self.weight_ih_posterior_sample,
                                self.weight_hh_posterior_sample,
                                self.bias_ih_sample,
                                self.bias_hh_sample)
        return output


class GRUCellFlipout(Module):

    __constants__ = ['input2hidden, hidden2hidden']

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCellFlipout, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input2hidden = LinearFlipout(input_size, hidden_size * 3, bias)
        self.hidden2hidden = LinearFlipout(hidden_size, hidden_size * 3, bias)
        #self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in self.parameters():
    #         init.uniform_(weight, -stdv, stdv)

    def kl_loss(self):
        return self.input2hidden.kl_loss() + self.hidden2hidden.kl_loss()

    def forward(self, input, state=None):
        #input = input.view(-1, input.size(1))
        gate_input = self.input2hidden(input).squeeze()
        if state is None:
            state = input.new_zeros(input.size(0),
                                    self.hidden_size,
                                    requires_grad=False)
        gate_hidden = self.hidden2hidden(state).squeeze()

        i_r, i_i, i_n = gate_input.chunk(3, 1)
        h_r, h_i, h_n = gate_hidden.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        output = newgate + inputgate * (state - newgate)
        return output


class GRULayer(Module):

    def __init__(self, cell, input_size, hidden_size, bias=True):
        super(GRULayer, self).__init__()
        self.cell = cell(input_size, hidden_size, bias)

    def kl_loss(self):
        return self.cell.kl_loss()

    def forward(self, inputs):
        outputs = []
        out = None
        for input in inputs:
            out = self.cell(input, out)
            outputs += [out]
        return outputs, out


class GRUBase(Module):

    def __init__(self, cell, input_size, hidden_size, num_layers=2, bias=True):
        super(GRUBase, self).__init__()
        layers = [GRULayer(cell, input_size, hidden_size, bias)]
        layers += [GRULayer(cell, hidden_size, hidden_size, bias)]
        self.layers = ModuleList(layers)

    def kl_loss(self):
        return sum(l.kl_loss() for l in self.layers)

    def forward(self, inputs):
        states = []
        out = inputs.unbind(0)
        for layer in self.layers:
            out, state = layer(out)
            states += [state]
        return torch.stack(out), torch.stack(states)


class GRUPathwise(GRUBase):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super(GRUPathwise, self).__init__(
            GRUCellPathwise, input_size, hidden_size, num_layers, bias)


class GRUFlipout(GRUBase):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super(GRUFlipout, self).__init__(
            GRUCellFlipout, input_size, hidden_size, num_layers, bias)

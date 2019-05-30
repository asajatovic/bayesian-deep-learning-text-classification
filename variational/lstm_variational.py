import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import lstm_cell as lstm_step
from torch.nn.parameter import Parameter

from .posteriors import PosteriorNormal
from .priors import PriorNormal, PriorLaplace
from .variational import (BayesByBackpropModule, random_rademacher,
                          random_rademacher_like)


@torch.jit.script
def flipout_lstm_step(input, state,
                      weight_ih_mean, weight_hh_mean,
                      weight_ih_perturbation, weight_hh_perturbation,
                      bias_hh, bias_ih,
                      sign_input, sign_hidden, sign_output):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor, Tensor, Tensor)
    # -> Tuple[Tensor, Tuple[Tensor, Tensor]]
    hx, cx = state
    perturbed_input = input * sign_input
    perturbed_hx = hx * sign_hidden
    perturbed_gates = (torch.mm(perturbed_input, weight_ih_perturbation.t()) +
                       torch.mm(perturbed_hx, weight_hh_perturbation.t())) * sign_output

    gates = (torch.mm(input, weight_ih_mean.t()) +
             torch.mm(hx, weight_hh_mean.t()))
    if bias_hh is not None and bias_ih is not None:
        gates += bias_ih + bias_hh

    gates += perturbed_gates

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, (hy, cy)


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        if self.use_bias:
            self.bias_ih = Parameter(torch.randn(4 * hidden_size))
            self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state=None):
        seq_len, batch_size, input_size = input.size()
        assert self.input_size == input_size
        self.seq_len = seq_len
        inputs = input.unbind(0)
        if state is None:
            zeros = torch.zeros(batch_size,
                                self.hidden_size,
                                dtype=input.dtype,
                                device=input.device)
            state = (zeros, zeros)
        outputs = []
        for input in inputs:
            state = lstm_step(input, state,
                              self.weight_ih, self.weight_hh,
                              self.bias_hh, self.bias_ih)
            outputs += [state[0]]
        return torch.stack(outputs), state

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        s += ', bias={self.use_bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s


class LSTMPathwise(BayesByBackpropModule):
    def __init__(self, input_size, hidden_size, bias=True, prior_args=(0, 0.1)):
        super(LSTMPathwise, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih_posterior = PosteriorNormal(
            Parameter(torch.Tensor(4 * hidden_size, input_size)),
            Parameter(torch.Tensor(4 * hidden_size, input_size)),
            self, "weight_ih")
        self.weight_hh_posterior = PosteriorNormal(
            Parameter(torch.Tensor(4 * hidden_size, hidden_size)),
            Parameter(torch.Tensor(4 * hidden_size, hidden_size)),
            self, "weight_hh")
        self.use_bias = bias
        if self.use_bias:
            self.bias_ih_posterior = PosteriorNormal(
                Parameter(torch.Tensor(4 * hidden_size)),
                Parameter(torch.Tensor(4 * hidden_size)),
                self, "bias_ih")
            self.bias_hh_posterior = PosteriorNormal(
                Parameter(torch.Tensor(4 * hidden_size)),
                Parameter(torch.Tensor(4 * hidden_size)),
                self, "bias_hh")
        else:
            self.register_parameter('bias', None)
        self.prior = PriorLaplace(*prior_args, self)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        init.normal_(self.weight_ih_loc, mean=0.0, std=stdv)
        init.normal_(self.weight_ih_scale, mean=-7.0, std=stdv)
        init.normal_(self.weight_hh_loc, mean=0.0, std=stdv)
        init.normal_(self.weight_hh_scale, mean=-7.0, std=stdv)
        if self.use_bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_ih_loc)
            bound = 1 / math.sqrt(fan_in)
            init.normal_(self.bias_ih_loc, mean=0.0, std=bound)
            init.normal_(self.bias_ih_scale, mean=-7.0, std=bound)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_hh_loc)
            bound = 1 / math.sqrt(fan_in)
            init.normal_(self.bias_hh_loc, mean=0.0, std=bound)
            init.normal_(self.bias_hh_scale, mean=-7.0, std=bound)

    def kl_loss(self):
        total_loss = (self.weight_ih_posterior.log_prob(self.weight_ih_sample) -
                      self.prior.log_prob(self.weight_ih_sample)).sum()
        total_loss = (self.weight_hh_posterior.log_prob(self.weight_hh_sample) -
                      self.prior.log_prob(self.weight_hh_sample)).sum()
        if self.use_bias:
            total_loss += (self.bias_ih_posterior.log_prob(self.bias_ih_sample) -
                           self.prior.log_prob(self.bias_ih_sample)).sum()
            total_loss += (self.bias_hh_posterior.log_prob(self.bias_hh_sample) -
                           self.prior.log_prob(self.bias_hh_sample)).sum()
        return total_loss / self.seq_len

    def sample_weights(self):
        self.weight_ih_sample = self.weight_ih_posterior.rsample()
        self.weight_hh_sample = self.weight_hh_posterior.rsample()
        self.bias_ih_sample = None
        self.bias_hh_sample = None
        if self.use_bias:
            self.bias_ih_sample = self.bias_ih_posterior.rsample()
            self.bias_hh_sample = self.bias_hh_posterior.rsample()

    def forward(self, input, state=None):
        seq_len, batch_size, input_size = input.size()
        assert self.input_size == input_size
        self.seq_len = seq_len
        inputs = input.unbind(0)
        if state is None:
            zeros = torch.zeros(batch_size,
                                self.hidden_size,
                                dtype=input.dtype,
                                device=input.device)
            state = (zeros, zeros)
        outputs = []
        for input in inputs:
            self.sample_weights()
            state = lstm_step(input, state,
                              self.weight_ih_sample, self.weight_hh_sample,
                              self.bias_hh_sample, self.bias_ih_sample)
            outputs += [state[0]]
        return torch.stack(outputs), state

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        s += ', bias={self.use_bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s


class LSTMFlipout(LSTMPathwise):

    def __init__(self, input_size, hidden_size, bias=True, prior_args=(0, 0.1)):
        super(LSTMFlipout, self).__init__(
            input_size, hidden_size, bias, prior_args)

    def sample_weights(self):
        self.weight_ih_perturbation = self.weight_ih_posterior.perturb()
        self.weight_ih_sample = self.weight_ih_loc + self.weight_ih_perturbation
        self.weight_hh_perturbation = self.weight_hh_posterior.perturb()
        self.weight_hh_sample = self.weight_hh_loc + self.weight_hh_perturbation
        self.bias_ih_sample = None
        self.bias_hh_sample = None
        if self.use_bias:
            self.bias_ih_sample = self.bias_ih_posterior.rsample()
            self.bias_hh_sample = self.bias_hh_posterior.rsample()

    def forward(self, input, state=None):
        seq_len, batch_size, input_size = input.size()
        assert self.input_size == input_size
        self.seq_len = seq_len
        inputs = input.unbind(0)
        if state is None:
            zeros = torch.zeros(batch_size,
                                self.hidden_size,
                                dtype=input.dtype,
                                device=input.device)
            state = (zeros, zeros)
        outputs = []
        for input in inputs:
            self.sample_weights()
            sign_input = random_rademacher_like(input)
            sign_hidden = random_rademacher_like(state[0])
            sign_output = random_rademacher(size=(batch_size, self.hidden_size*4),
                                            dtype=input.dtype,
                                            device=input.device)
            out, state = flipout_lstm_step(input, state,
                                           self.weight_ih_loc, self.weight_hh_loc,
                                           self.weight_ih_perturbation, self.weight_hh_perturbation,
                                           self.bias_hh_sample, self.bias_ih_sample,
                                           sign_input, sign_hidden, sign_output)
            outputs += [out]
        return torch.stack(outputs), state

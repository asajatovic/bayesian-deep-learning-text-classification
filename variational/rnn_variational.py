import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter

from .linear_variational import LinearFlipout, LinearPathwise
from .variational import BayesByBackpropModule
from .posteriors import PosteriorNormal
from .priors import PriorNormal


class RNNCellBase(BayesByBackpropModule):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias, num_chunks, prior_args=(0, 0.1)):
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias
        self.weight_ih_posterior = PosteriorNormal(
            Parameter(torch.Tensor(num_chunks * hidden_size, input_size)),
            Parameter(torch.Tensor(num_chunks * hidden_size, input_size)),
            self, "weight_ih")
        self.weight_hh_posterior = PosteriorNormal(
            Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size)),
            Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size)),
            self, "weight_hh")
        if bias:
            self.bias_ih_posterior = PosteriorNormal(
                Parameter(torch.Tensor(num_chunks * hidden_size)),
                Parameter(torch.Tensor(num_chunks * hidden_size)),
                self, "bias_ih")
            self.bias_hh_posterior = PosteriorNormal(
                Parameter(torch.Tensor(num_chunks * hidden_size)),
                Parameter(torch.Tensor(num_chunks * hidden_size)),
                self, "bias_hh")
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.prior = PriorNormal(*prior_args, self)
        self.reset_parameters()

    # def extra_repr(self):
    #     s = '{input_size}, {hidden_size}'
    #     if 'bias' in self.__dict__ and self.bias is not True:
    #         s += ', bias={bias}'
    #     if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
    #         s += ', nonlinearity={nonlinearity}'
    #     return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        # type: (Tensor, Tensor, str) -> None
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

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
        total_loss += (self.weight_hh_posterior.log_prob(self.weight_hh_sample) -
                       self.prior.log_prob(self.weight_hh_sample)).sum()
        if self.use_bias:
            total_loss += (self.bias_ih_posterior.log_prob(self.bias_ih_sample) -
                           self.prior.log_prob(self.bias_ih_sample)).sum()
            total_loss += (self.bias_hh_posterior.log_prob(self.bias_hh_sample) -
                           self.prior.log_prob(self.bias_hh_sample)).sum()
        return total_loss

class GRUCellPathwise(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, prior_args=(0.0, 0.1)):
        super(GRUCellPathwise, self).__init__(input_size, hidden_size, bias, num_chunks=3, prior_args=prior_args)

    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        self.sample_weights()
        return torch.gru_cell(
            input, hx,
            self.weight_ih_sample, self.weight_hh_sample,
            self.bias_ih_sample, self.bias_hh_sample,
        )

    def sample_weights(self):
        self.weight_ih_sample = self.weight_ih_posterior.rsample()
        self.weight_hh_sample = self.weight_hh_posterior.rsample()
        self.bias_ih_sample = self.bias_ih_posterior.rsample()
        self.bias_hh_sample = self.bias_hh_posterior.rsample()
        

# class GRUCellPathwise(BayesByBackpropModule):

#     __constants__ = ['input2hidden, hidden2hidden']

#     def __init__(self, input_size, hidden_size, bias=True):
#         super(GRUCellPathwise, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.input2hidden = LinearPathwise(input_size, hidden_size * 3, bias)
#         self.hidden2hidden = LinearPathwise(hidden_size, hidden_size * 3, bias)
#         # self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             nn.init.uniform_(weight, -stdv, stdv)

#     def kl_loss(self):
#         return self.input2hidden.kl_loss() + self.hidden2hidden.kl_loss()

#     def forward(self, input, state=None):
#         #input = input.view(-1, input.size(1))
#         gate_input = self.input2hidden(input).squeeze()
#         if state is None:
#             state = input.new_zeros(input.size(0),
#                                     self.hidden_size,
#                                     requires_grad=False)
#         gate_hidden = self.hidden2hidden(state).squeeze()

#         i_r, i_i, i_n = gate_input.chunk(3, 1)
#         h_r, h_i, h_n = gate_hidden.chunk(3, 1)

#         resetgate = torch.sigmoid(i_r + h_r)
#         inputgate = torch.sigmoid(i_i + h_i)
#         newgate = torch.tanh(i_n + resetgate * h_n)
#         output = newgate + inputgate * (state - newgate)
#         return output


class GRUCellFlipout(BayesByBackpropModule):

    # __constants__ = ['input2hidden, hidden2hidden']

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCellFlipout, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input2hidden = LinearFlipout(input_size, hidden_size * 3, bias)
        self.hidden2hidden = LinearFlipout(hidden_size, hidden_size * 3, bias)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

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


class GRULayer(BayesByBackpropModule):

    # __constants__ = ['cell']

    def __init__(self, cell, input_size, hidden_size, bias=True):
        super(GRULayer, self).__init__()
        self.cell = cell(input_size, hidden_size, bias)

    def kl_loss(self):
        return self.cell.kl_loss() / self.cuts

    def forward(self, inputs):
        # self.cell.sample_weights()
        outputs = []
        out = None
        self.cuts = len(inputs)
        for input in inputs:
            out = self.cell(input, out)
            outputs += [out]
        return outputs, out


class _GRUBase(BayesByBackpropModule):

    # __constants__ = ['layers']

    def __init__(self, cell, input_size, hidden_size, num_layers=2, bias=True):
        super(_GRUBase, self).__init__()
        layers = [GRULayer(cell, input_size, hidden_size, bias)]
        layers += [GRULayer(cell, hidden_size, hidden_size, bias)
                   for _ in range(num_layers-1)]
        self.layers = nn.ModuleList(layers)

    def kl_loss(self):
        total_loss = 0.0
        for layer in self.layers:
            total_loss += layer.kl_loss()
        return total_loss
        # return sum(l.kl_loss() for l in self.layers)

    def forward(self, inputs):
        states = []
        out = inputs.unbind(0)
        for layer in self.layers:
            out, state = layer(out)
            states += [state]
        return torch.stack(out), torch.stack(states)


class GRUPathwise(_GRUBase):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super(GRUPathwise, self).__init__(
            GRUCellPathwise, input_size, hidden_size, num_layers, bias)


class GRUFlipout(_GRUBase):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super(GRUFlipout, self).__init__(
            GRUCellFlipout, input_size, hidden_size, num_layers, bias)

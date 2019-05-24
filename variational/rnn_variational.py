import math

import torch
import torch.nn as nn

from .linear_variational import LinearFlipout, LinearPathwise
from .variational import BayesByBackpropModule, random_rademacher


class GRUCellPathwise(BayesByBackpropModule):

    __constants__ = ['input2hidden, hidden2hidden']

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCellPathwise, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input2hidden = LinearPathwise(input_size, hidden_size * 3, bias)
        self.hidden2hidden = LinearPathwise(hidden_size, hidden_size * 3, bias)
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


class GRUCellFlipout(BayesByBackpropModule):

    __constants__ = ['input2hidden, hidden2hidden']

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


class _GRUBase(BayesByBackpropModule):

    def __init__(self, cell, input_size, hidden_size, num_layers=2, bias=True):
        super(_GRUBase, self).__init__()
        layers = [GRULayer(cell, input_size, hidden_size, bias)]
        layers += [GRULayer(cell, hidden_size, hidden_size, bias)]
        self.layers = nn.ModuleList(layers)

    def kl_loss(self):
        return sum(l.kl_loss() for l in self.layers)

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

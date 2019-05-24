import math

import torch
from torch.distributions.kl import kl_divergence
from torch.nn import init
from torch.nn.modules.utils import _single
from torch.nn.parameter import Parameter

from .posteriors import PosteriorNormal
from .priors import PriorLaplace, PriorNormal
from .utils import random_rademacher
from .variational import GaussianVariationalModule


class _ConvNdPathwise(GaussianVariationalModule):

    __constants__ = ['stride', 'padding',
                     'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_ConvNdPathwise, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight_posterior = PosteriorNormal(
                torch.Tensor(in_channels, out_channels // groups, *kernel_size),
                torch.Tensor(in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight_posterior = PosteriorNormal(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size),
                torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.weight_prior = PriorNormal(0, 1)
        self.use_bias = bias
        if self.use_bias:
            self.bias_posterior = PosteriorNormal(torch.Tensor(out_channels),
                                                  torch.Tensor(out_channels))
            self.bias_prior = PriorNormal(0, 1)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight_posterior.loc, mean=0.0, std=0.1)
        init.normal_(self.weight_posterior.scale, mean=-3.0, std=0.1)
        if self.use_bias:
            init.uniform(self.bias_posterior.loc, -0.1, 0.1)
            init.normal_(self.weight_posterior.scale, mean=-3.0, std=0.1)

    def kl_loss(self):
        weight_loss = kl_divergence(self.weight_posterior, self.weight_prior)
        total_loss = weight_loss.sum()
        if self.use_bias:
            bias_loss = kl_divergence(self.bias_posterior, self.bias_prior)
            total_loss += bias_loss.sum()
        return total_loss

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.use_bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv1dPathwise(_ConvNdPathwise):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1dPathwise, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode='zeros')

    def forward(self, input):
        bias_sample = self.bias_posterior.rsample() if self.use_bias else None
        output = torch.conv1d(input,
                              self.weight_posterior.rsample(),
                              bias_sample,
                              self.stride,
                              self.padding,
                              self.dilation,
                              self.groups)
        return output


class Conv1dFlipout(Conv1dPathwise):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        self.seed = sum(ord(i) for i in type(self).__name__)
        super(Conv1dFlipout, self).__init__(in_channels, out_channels,
                                            kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        # print('-'*50)
        # print(self.weight_posterior.loc.device)
        # print()
        # print(self.weight_posterior.loc.data.device)
        # print('-'*50)
        bias_loc = self.bias_posterior.loc if self.use_bias else None
        output = torch.conv1d(input,
                              self.weight_posterior.loc,
                              bias_loc,
                              self.stride,
                              self.padding,
                              self.dilation,
                              self.groups)

        sign_input = random_rademacher(input)
        sign_output = random_rademacher(output)

        bias_sample = self.bias_posterior.fsample() if self.use_bias else None
        perturbed_input = torch.conv1d(input * sign_input,
                                       self.weight_posterior.fsample(),
                                       bias_sample,
                                       self.stride,
                                       self.padding,
                                       self.dilation,
                                       self.groups) * sign_output
        output += perturbed_input
        return output

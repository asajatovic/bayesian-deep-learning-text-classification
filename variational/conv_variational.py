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
                 groups, bias, padding_mode, prior):
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
            self.weight_posterior_mean = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.weight_posterior_logscale = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight_posterior_mean = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.weight_posterior_logscale = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        self.register_buffer("weight_prior_mean",
                             torch.zeros_like(self.weight_posterior_mean) + prior[0])
        self.register_buffer("weight_prior_scale",
                             torch.zeros_like(self.weight_posterior_logscale) + prior[1])
        self.use_bias = bias
        if self.use_bias:
            self.bias_posterior_mean = Parameter(torch.Tensor(out_channels))
            self.bias_posterior_logscale = Parameter(
                torch.Tensor(out_channels))
            self.register_buffer("bias_prior_mean",
                                 torch.zeros_like(self.bias_posterior_mean) + prior[0])
            self.register_buffer("bias_prior_scale",
                                 torch.zeros_like(self.bias_posterior_logscale) + prior[1])
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

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
                 bias=True, padding_mode='zeros', prior=(0, 1)):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1dPathwise, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode='zeros', prior=prior)

    def forward(self, input):
        output = torch.conv1d(input,
                              self.weight_posterior_sample,
                              self.bias_posterior_sample,
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
        output = torch.conv1d(input,
                              self.weight_posterior_sample,
                              self.bias_posterior_sample,
                              self.stride,
                              self.padding,
                              self.dilation,
                              self.groups)

        sign_input = random_rademacher(input)
        sign_output = random_rademacher(output)

        perturbed_input = torch.conv1d(input * sign_input,
                                       self.weight_posterior_sample,
                                       None,
                                       self.stride,
                                       self.padding,
                                       self.dilation,
                                       self.groups) * sign_output
        output += perturbed_input
        return output

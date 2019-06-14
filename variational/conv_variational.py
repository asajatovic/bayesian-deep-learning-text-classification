import math

import torch
from torch.nn import init
from torch.nn.modules.utils import _single
from torch.nn.parameter import Parameter

from .posteriors import PosteriorNormal
from .priors import PriorLaplace, PriorNormal
from .variational import BBBModule


class _ConvNdPathwise(BBBModule):

    __constants__ = ['stride', 'padding',
                     'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, prior_args):
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
                Parameter(torch.Tensor(
                    in_channels, out_channels // groups, *kernel_size)),
                Parameter(torch.Tensor(
                    in_channels, out_channels // groups, *kernel_size)),
                self, "weight")
        else:
            self.weight_posterior = PosteriorNormal(
                Parameter(torch.Tensor(
                    out_channels, in_channels // groups, *kernel_size)),
                Parameter(torch.Tensor(
                    out_channels, in_channels // groups, *kernel_size)),
                self, "weight")
        self.use_bias = bias
        if self.use_bias:
            self.bias_posterior = PosteriorNormal(
                Parameter(torch.Tensor(out_channels)),
                Parameter(torch.Tensor(out_channels)),
                self, "bias")
        else:
            self.register_parameter('bias', None)
        self.prior = PriorLaplace(*prior_args, self)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_channels)
        init.normal_(self.weight_loc, mean=0.0, std=stdv)
        init.normal_(self.weight_scale, mean=-7.0, std=stdv)
        if self.use_bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_loc)
            bound = 1 / math.sqrt(fan_in)
            init.normal_(self.bias_loc, mean=0.0, std=bound)
            init.normal_(self.bias_scale, mean=-7.0, std=bound)

    def kl_loss(self):
        total_loss = (self.weight_posterior.log_prob(self.weight_sample).sum() -
                      self.prior.log_prob(self.weight_sample).sum())
        if self.use_bias:
            total_loss += (self.bias_posterior.log_prob(self.bias_sample).sum() -
                           self.prior.log_prob(self.bias_sample).sum())
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
                 bias=True, padding_mode='zeros', prior_args=(0, 0.1)):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1dPathwise, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode='zeros', prior_args=prior_args)

    def forward(self, input):
        self.weight_sample = self.weight_posterior.rsample()
        self.bias_sample = None
        if self.use_bias:
            self.bias_sample = self.bias_posterior.rsample()
        output = torch.conv1d(input,
                              self.weight_sample,
                              self.bias_sample,
                              self.stride,
                              self.padding,
                              self.dilation,
                              self.groups)
        return output

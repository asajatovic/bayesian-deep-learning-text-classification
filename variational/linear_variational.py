import math

import torch
from torch.nn import init
from torch.nn.parameter import Parameter

from .posteriors import PosteriorNormal
from .priors import PriorLaplace, PriorNormal
from .variational import BBBModule


class LinearPathwise(BBBModule):

    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True, prior_args=(0, 0.1)):
        super(LinearPathwise, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_posterior = PosteriorNormal(
            Parameter(torch.Tensor(out_features, in_features)),
            Parameter(torch.Tensor(out_features, in_features)),
            self, "weight")
        self.use_bias = bias
        if self.use_bias:
            self.bias_posterior = PosteriorNormal(
                Parameter(torch.Tensor(out_features)),
                Parameter(torch.Tensor(out_features)),
                self, "bias")
        else:
            self.register_parameter('bias', None)
        self.prior = PriorNormal(*prior_args, self)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        init.normal_(self.weight_loc, mean=0.0, std=stdv)
        init.normal_(self.weight_scale, mean=-7.0, std=stdv)
        if self.use_bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_loc)
            bound = 1 / math.sqrt(fan_in)
            init.normal_(self.bias_loc, mean=0.0, std=bound)
            init.normal_(self.bias_scale, mean=-7.0, std=bound)

    def kl_loss(self):
        total_loss = (self.weight_posterior.log_prob(self.weight_sample) -
                      self.prior.log_prob(self.weight_sample)).sum()
        if self.use_bias:
            total_loss += (self.bias_posterior.log_prob(self.bias_sample) -
                           self.prior.log_prob(self.bias_sample)).sum()
        return total_loss

    def forward(self, input):
        self.weight_sample = self.weight_posterior.rsample()
        output = torch.matmul(input, self.weight_sample.t())
        if self.use_bias:
            self.bias_sample = self.bias_posterior.rsample()
            output += self.bias_sample
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.use_bias
        )

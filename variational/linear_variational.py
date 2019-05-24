import math

import torch
from torch.distributions.kl import kl_divergence
from torch.distributions.utils import _standard_normal
from torch.nn import init
from torch.nn.functional import softplus
from torch.nn.parameter import Parameter

from .posteriors import PosteriorNormal
from .priors import PriorLaplace, PriorNormal
from .utils import random_rademacher, kl_normal_normal
from .variational import GaussianVariationalModule


class LinearPathwise(GaussianVariationalModule):

    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True, prior=(0, 1)):
        super(LinearPathwise, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_posterior_mean = Parameter(
            torch.Tensor(out_features, in_features))
        self.weight_posterior_logscale = Parameter(
            torch.Tensor(out_features, in_features))
        # self.weight_posterior_sample = None
        self.register_buffer("weight_prior_mean",
                             torch.zeros_like(self.weight_posterior_mean) + prior[0])
        self.register_buffer("weight_prior_scale",
                             torch.zeros_like(self.weight_posterior_logscale) + prior[1])
        self.use_bias = bias
        if self.use_bias:
            self.bias_posterior_mean = Parameter(torch.Tensor(out_features))
            self.bias_posterior_logscale = Parameter(
                torch.Tensor(out_features))
            # self.bias_posterior_sample = None
            self.register_buffer("bias_prior_mean",
                                 torch.zeros_like(self.bias_posterior_mean) + prior[0])
            self.register_buffer("bias_prior_scale",
                                 torch.zeros_like(self.bias_posterior_logscale) + prior[1])
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        output = torch.matmul(input, self.weight_posterior_sample.t())
        if self.use_bias:
            output += self.weight_posterior_sample
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.use_bias
        )


class LinearFlipout(LinearPathwise):

    def __init__(self, in_features, out_features, bias=True, prior=(0, 1)):
        super(LinearFlipout, self).__init__(
            in_features, out_features, bias, prior)

    def forward(self, input):
        output = torch.matmul(input, self.weight_posterior_mean.t())
        if self.use_bias:
            output += self.bias_posterior_sample

        sign_input = random_rademacher(input)
        sign_output = random_rademacher(output)
        perturbed_input = torch.matmul(input * sign_input,
                                       self.weight_perturbation.t()) * sign_output
        output += perturbed_input
        return output

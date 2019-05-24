import math

import torch
from torch.distributions.kl import kl_divergence
from torch.distributions.utils import _standard_normal
from torch.nn import init
from torch.nn.parameter import Parameter

from .posteriors import PosteriorNormal
from .priors import PriorNormal
from .variational import BayesByBackpropModule, random_rademacher


class LinearPathwise(BayesByBackpropModule):

    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True, prior_args=(0, 1)):
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
        init.normal_(self.weight_loc, mean=0.0, std=0.1)
        init.normal_(self.weight_scale, mean=-3.0, std=0.1)
        if self.use_bias:
            init.uniform_(self.bias_loc, -0.1, 0.1)
            init.normal_(self.bias_scale, mean=-3.0, std=0.1)

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


class LinearFlipout(LinearPathwise):

    def __init__(self, in_features, out_features, bias=True, prior=(0, 1)):
        super(LinearFlipout, self).__init__(
            in_features, out_features, bias, prior)

    def forward(self, input):
        output = torch.matmul(input, self.weight_loc.t())
        if self.use_bias:
            self.bias_sample = self.bias_posterior.rsample()
            output += self.bias_sample

        sign_input = random_rademacher(input)
        sign_output = random_rademacher(output)
        self.weight_perturbation = self.weight_posterior.perturb()
        self.weight_sample = self.weight_loc + self.weight_perturbation
        perturbed_input = torch.matmul(input * sign_input,
                                       self.weight_perturbation.t()) * sign_output
        output += perturbed_input
        return output

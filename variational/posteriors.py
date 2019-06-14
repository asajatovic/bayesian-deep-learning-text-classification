import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal
from torch.nn.functional import softplus


class PosteriorNormal(Distribution):

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return softplus(self.scale)

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale, module, name, validate_args=None):
        self.loc = loc
        self.scale = scale
        module.register_parameter(f'{name}_loc', self.loc)
        module.register_parameter(f'{name}_scale', self.scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(PosteriorNormal, self).__init__(
            batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        warnings.warn('Expaning the posterior distribution is not allowed!')
        return self

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.stddev.expand(shape))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(
            shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.stddev

    def perturb(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(
            shape, dtype=self.loc.dtype, device=self.loc.device)
        return eps * self.stddev

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        # var = (self.stddev ** 2)
        log_scale = self.stddev.log()
        return -((value - self.loc) ** 2) / (2 * self.variance) - log_scale - math.log(math.sqrt(2 * math.pi))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf((value - self.loc) * self.stddev.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.loc + self.stddev * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.stddev)

    @property
    def _natural_params(self):
        return (self.loc / self.stddev.pow(2), -0.5 * self.stddev.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)

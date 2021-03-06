import math
import warnings
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal

def prior_builder(prior_type, module):
    if prior_type.lower() == "normal":
        return PriorNormal(module, loc=0.0, scale=0.1)
    elif prior_type.lower() == "laplace":
        return PriorLaplace(module, loc=0.0, scale=0.1)

class PriorNormal(Distribution):

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, module, loc=0.0, scale=0.1, validate_args=None):
        self.loc = torch.tensor(float(loc))
        self.scale = torch.tensor(float(scale))
        module.register_buffer('prior_loc', self.loc)
        module.register_buffer('prior_scale', self.scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(PriorNormal, self).__init__(
            batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        warnings.warn('Expanding the prior distribution is not allowed!')
        return self

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(
            shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(
            self.scale, Number) else self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)


class PriorLaplace(Distribution):

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return 2 * self.scale.pow(2)

    @property
    def stddev(self):
        return (2 ** 0.5) * self.scale

    def __init__(self, module, loc=0.0, scale=0.1, validate_args=None):
        self.loc = torch.tensor(float(loc))
        self.scale = torch.tensor(float(scale))
        module.register_buffer('prior_loc', self.loc)
        module.register_buffer('prior_scale', self.scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(PriorLaplace, self).__init__(
            batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        warnings.warn('Expanding the prior distribution is not allowed!')
        return self

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        finfo = torch.finfo(self.loc.dtype)
        if torch._C._get_tracing_state():
            # [JIT WORKAROUND] lack of support for .uniform_()
            u = torch.rand(shape, dtype=self.loc.dtype,
                           device=self.loc.device) * 2 - 1
            return self.loc - self.scale * u.sign() * torch.log1p(-u.abs().clamp(min=finfo.tiny))
        u = self.loc.new(shape).uniform_(finfo.eps - 1, 1)
        return self.loc - self.scale * u.sign() * torch.log1p(-u.abs())

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -torch.log(2 * self.scale) - torch.abs(value - self.loc) / self.scale

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 - 0.5 * (value - self.loc).sign() * torch.expm1(-(value - self.loc).abs() / self.scale)

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        term = value - 0.5
        return self.loc - self.scale * (term).sign() * torch.log1p(-2 * term.abs())

    def entropy(self):
        return 1 + torch.log(2 * self.scale)

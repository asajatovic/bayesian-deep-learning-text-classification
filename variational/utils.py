import math

import torch
from torch.distributions.kl import register_kl
from torch.nn.functional import softplus

from .posteriors import PosteriorNormal
from .priors import PriorLaplace, PriorNormal


def random_rademacher(input):
    return 2 * torch.zeros_like(input).bernoulli(p=0.5) - 1
    # return result.to(input.device)


@register_kl(PosteriorNormal, PriorNormal)
def kl_posteriornormal_priornormal(p, q):
    var_ratio = (p.stddev / q.stddev).pow(2)
    t1 = ((p.mean - q.mean) / q.stddev).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


@register_kl(PosteriorNormal, PriorLaplace)
def kl_posteriornormal_priorlaplace(p, q):
    raise NotImplementedError


def kl_normal_normal(p_mean, p_logscale, q_mean, q_scale):
    var_ratio = (sotfplus(p_logscale) / q_scale).pow(2)
    t1 = ((p_mean - q_mean) / q_scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


def stochastic_kl(posterior_mean, posterior_logscale, prior_mean, prior_logscale, sample):
    posterior_log_prob = (-((sample - posterior_mean) ** 2) / (2 * (softplus(
        posterior_logscale)**2)) - posterior_logscale - math.log(math.sqrt(2 * math.pi)))
    prior_log_prob = (-((sample - prior_mean) ** 2) / (2 * (softplus(prior_logscale)
                                                            ** 2)) - prior_logscale - math.log(math.sqrt(2 * math.pi)))
    return posterior_log_prob.item() - prior_log_prob.item()

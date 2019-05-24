from torch.distributions.utils import _standard_normal
from torch.nn import Module, init
from torch.nn.functional import softplus


def random_rademacher(input):
    return 2 * torch.zeros_like(input).bernoulli(p=0.5) - 1


def kl_normal_normal(p_mean, p_scale, q_mean, q_scale):
    # p_scale = softplus(p_logscale)
    var_ratio = (p_scale / q_scale).pow(2)
    t1 = ((p_mean - q_mean) / q_scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

# class BayesByBackpropModule(Module)
class GaussianVariationalModule(Module):

    def reset_parameters(self):
        init.normal_(self.weight_posterior_mean, mean=0.0, std=0.1)
        init.normal_(self.weight_posterior_logscale, mean=-3.0, std=0.1)
        if self.use_bias:
            init.uniform(self.bias_posterior_mean, -0.1, 0.1)
            init.normal_(self.bias_posterior_logscale, mean=-3.0, std=0.1)

    @property
    def weight_perturbation(self):
        eps = _standard_normal(shape=self.weight_posterior_logscale.shape,
                               dtype=self.weight_posterior_logscale.dtype,
                               device=self.weight_posterior_logscale.device)
        return eps * softplus(self.weight_posterior_logscale)

    @property
    def weight_posterior_sample(self):
        return self.weight_posterior_mean + self.weight_perturbation

    @property
    def bias_posterior_sample(self):
        if not self.use_bias:
            return None
        eps = _standard_normal(shape=self.bias_posterior_logscale.shape,
                               dtype=self.bias_posterior_logscale.dtype,
                               device=self.bias_posterior_logscale.device)
        return self.bias_posterior_mean + softplus(self.bias_posterior_logscale) * eps

    def kl_loss(self, scaling=None):
        weight_posterior_loss = kl_normal_normal(self.weight_posterior_mean,
                                                 softplus(
                                                     self.weight_posterior_logscale),
                                                 self.weight_prior_mean,
                                                 self.weight_prior_scale)
        total_loss = weight_posterior_loss.sum()
        if self.use_bias:
            bias_loss = kl_normal_normal(self.bias_posterior_mean,
                                         softplus(
                                             self.bias_posterior_logscale),
                                         self.bias_prior_mean,
                                         self.bias_prior_scale)
            total_loss += bias_loss.sum()
        return total_loss

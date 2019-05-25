import torch
import torch.nn as nn


def random_rademacher(input):
    return 2 * torch.zeros_like(input).bernoulli(p=0.5) - 1


class BayesByBackpropModule(nn.Module):

    def kl_loss(self):
        raise NotImplementedError


class ELBO(nn.Module):

    def __init__(self, model, data_cost):
        super(ELBO, self).__init__()
        self.model = model
        self.data_cost = data_cost

    def forward(self, input, target):
        num_examples = len(target)
        scaling = 1
        scaled_kl_loss = scaling * self.model.kl_loss() / num_examples
        return scaled_kl_loss + self.data_cost(input, target)

import torch
import torch.nn as nn


class BBBModule(nn.Module):

    def kl_loss(self):
        raise NotImplementedError


class ELBO(nn.Module):

    def __init__(self, model, data_cost, num_batches):
        super(ELBO, self).__init__()
        self.model = model
        self.data_cost = data_cost
        self.num_batches = num_batches

    def forward(self, input, target):
        scaling_factor = len(target) * self.num_batches
        scaled_kl_loss = self.model.kl_loss() / scaling_factor
        return scaled_kl_loss + self.data_cost(input, target)

from torch.distributions.laplace import Laplace
from torch.distributions.normal import Normal


class PriorNormal(Normal):

    def _apply(self, fn):
        if self.loc is not None:
            self.loc.data = fn(self.loc.data)
            if self.loc._grad is not None:
                self.loc._grad.data = fn(self.loc._grad.data)
        if self.scale is not None:
            self.scale.data = fn(self.scale.data)
            if self.scale._grad is not None:
                self.scale._grad.data = fn(self.scale._grad.data)


class PriorLaplace(Laplace):

    def _apply(self, fn):
        if self.loc is not None:
            self.loc.data = fn(self.loc.data)
            if self.loc._grad is not None:
                self.loc._grad.data = fn(self.loc._grad.data)
        if self.scale is not None:
            self.scale.data = fn(self.scale.data)
            if self.scale._grad is not None:
                self.scale._grad.data = fn(self.scale._grad.data)

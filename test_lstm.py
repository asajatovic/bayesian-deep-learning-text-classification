import torch
from variational import LSTMLayer, LSTMPathwise, LSTMFlipout

input = torch.randn(2,5,4)

ll = LSTMLayer(4,7)
ll(input)

lp = LSTMPathwise(4,7)
lp(input)

lf = LSTMFlipout(4,7)
lf(input)
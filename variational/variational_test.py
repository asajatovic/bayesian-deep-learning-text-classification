import torch
import torch.nn as nn

from conv_variational import Conv1dPathwise
from linear_variational import LinearPathwise
from rnn_variational import GRUPathwise


def linear_variational_test():
    def copy_weights(source, target):
        src_params = list(source.parameters())[0::2]  # skip scales
        trg_params = list(target.parameters())
        for src_param, trg_param in zip(src_params, trg_params):
            assert src_param.shape == trg_param.shape
            with torch.no_grad():
                trg_param.copy_(src_param)

    lp = LinearPathwise(2, 4, bias=True)
    # set scales close to zero (softplus)
    nn.init.constant_(lp.weight_posterior.scale, -100)
    nn.init.constant_(lp.bias_posterior.scale, -100)

    l = nn.Linear(2, 4, bias=True)
    copy_weights(source=lp, target=l)

    input = torch.randn(1, 5, 2)

    out_p = lp(input)
    out = l(input)
    assert torch.equal(lp(input), l(input))


def conv1d_variational_test():
    def copy_weights(source, target):
        src_params = list(source.parameters())[0::2]  # skip scales
        trg_params = list(target.parameters())
        for src_param, trg_param in zip(src_params, trg_params):
            assert src_param.shape == trg_param.shape
            with torch.no_grad():
                trg_param.copy_(src_param)

    cp = Conv1dPathwise(5, 7, kernel_size=3, bias=True)
    # set scales close to zero (softplus)
    nn.init.constant_(cp.weight_posterior.scale, -100)
    nn.init.constant_(cp.bias_posterior.scale, -100)

    c = nn.Conv1d(5, 7, kernel_size=3, bias=True)
    copy_weights(source=cp, target=c)

    input = torch.randn(1, 5, 10)
    assert torch.equal(cp(input), c(input))


def rnn_variational_test():
    def copy_weights(source, target):
        src_params = list(source.parameters())[0::2]  # skip scales
        src_params = src_params[0::2] + src_params[1::2]  # weights then biases
        trg_params = target.all_weights[0]
        for src_param, trg_param in zip(src_params, trg_params):
            assert src_param.shape == trg_param.shape
            with torch.no_grad():
                trg_param.copy_(src_param)

    gp = GRUPathwise(5, 7, num_layers=1, bias=True)
    # set scales close to zero (softplus)
    nn.init.constant_(
        gp.layers[0].cell.input2hidden.weight_posterior.scale, -100)
    nn.init.constant_(
        gp.layers[0].cell.input2hidden.bias_posterior.scale, -100)
    nn.init.constant_(
        gp.layers[0].cell.hidden2hidden.weight_posterior.scale, -100)
    nn.init.constant_(
        gp.layers[0].cell.hidden2hidden.bias_posterior.scale, -100)

    g = nn.GRU(5, 7, num_layers=1, bias=True)
    copy_weights(source=gp, target=g)

    input = torch.randn(3, 4, 5)
    out_p, state_p = gp(input)
    out, state = g(input)
    assert torch.equal(out_p, out)
    assert torch.equal(state_p, state)


if __name__ == "__main__":
    linear_variational_test()
    conv1d_variational_test()
    rnn_variational_test()
    print("PASS")

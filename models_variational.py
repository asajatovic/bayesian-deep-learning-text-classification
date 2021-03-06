import torch
import torch.nn as nn
import torch.nn.functional as F

from variational import (BBBModule, Conv1dPathwise, LinearPathwise, LSTMLayer,
                         LSTMPathwise)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, prior_type):
        super(TemporalBlock, self).__init__()
        self.conv1 = Conv1dPathwise(n_inputs, n_outputs, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation,
                                    prior_type=prior_type)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = Conv1dPathwise(n_outputs, n_outputs, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation,
                                    prior_type=prior_type)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2)
        self.downsample = Conv1dPathwise(
            n_inputs, n_outputs, stride=1, prior_type=prior_type) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        # self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, prior_type):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, prior_type=prior_type)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TextTCN(BBBModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, kernel_size, num_classes, mode, prior_type="normal", weights=None):
        super(TextTCN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        if weights is not None:
            self.load_embeddings(weights)
        self.tcn = TemporalConvNet(num_inputs=embedding_dim,
                                   num_channels=[hidden_dim] * num_layers,
                                   kernel_size=kernel_size,
                                   prior_type=prior_type)
        self.fc = LinearPathwise(
            hidden_dim, num_classes, prior_type=prior_type)
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x.transpose(1, 0)).transpose(1, 2)
        x = self.tcn(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(dim=-1)  # global avg pool
        x = self.fc(x)
        return self.activation(x).squeeze()

    def kl_loss(self):
        total_loss = 0.0
        for module in self.children():
            if issubclass(type(module), BBBModule):
                total_loss += module.kl_loss()
        return total_loss

    def load_embeddings(self, weights):
        if 'static' in self.mode:
            self.embedding.weight.data.copy_(weights)
            if 'non' not in self.mode:
                self.embedding.weight.data.requires_grad = False
                print('Loaded pretrained embeddings, weights are not trainable.')
            else:
                self.embedding.weight.data.requires_grad = True
                print('Loaded pretrained embeddings, weights are trainable.')
        elif self.mode == 'rand':
            print('Randomly initialized embeddings are used.')
        else:
            raise ValueError(
                'Unexpected value of mode. Please choose from static, nonstatic, rand.')


class TextLSTM(BBBModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, mode, prior_type="normal", weights=None):
        super(TextLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        if weights is not None:
            self.load_embeddings(weights)
        lstms = [LSTMPathwise(input_size=embedding_dim,
                              hidden_size=hidden_dim,
                              bias=True,
                              prior_type=prior_type)]
        lstms += [LSTMPathwise(input_size=hidden_dim,
                               hidden_size=hidden_dim,
                               bias=True,
                               prior_type=prior_type) for i in range(num_layers-1)]
        self.lstms = nn.ModuleList(lstms)
        self.num_directions = 1
        self.fc = LinearPathwise(hidden_dim*self.num_directions, num_classes, prior_type=prior_type)
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.LogSoftmax(dim=1)

    def kl_loss(self):
        total_loss = 0.0
        for module in self.children():
            if issubclass(type(module), BBBModule):
                total_loss += module.kl_loss()
        return total_loss

    def forward(self, x):
        _, batch_size = x.shape
        x = self.embedding(x)
        state = None
        out = x
        for lstm in self.lstms:
            out, state = lstm(out, state)
        x, _ = state  # final hidden state
        x = x.view(-1,
                   self.num_directions,
                   batch_size,
                   self.hidden_dim)
        x = x[-1][-1]  # final hidden state
        x = self.fc(x.squeeze())
        return self.activation(x).squeeze()

    def load_embeddings(self, weights):
        if 'static' in self.mode:
            self.embedding.weight.data.copy_(weights)
            if 'non' not in self.mode:
                self.embedding.weight.data.requires_grad = False
                print('Loaded pretrained embeddings, weights are not trainable.')
            else:
                self.embedding.weight.data.requires_grad = True
                print('Loaded pretrained embeddings, weights are trainable.')
        elif self.mode == 'rand':
            print('Randomly initialized embeddings are used.')
        else:
            raise ValueError(
                'Unexpected value of mode. Please choose from static, nonstatic, rand.')

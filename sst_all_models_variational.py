import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss, RunningAverage
from torchtext import data, datasets
from torchtext.vocab import GloVe
from variational import (ELBO, BayesByBackpropModule, Conv1dFlipout,
                         Conv1dPathwise, GRUFlipout, GRUPathwise,
                         LinearFlipout, LinearPathwise)

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT = data.Field(lower=True, tokenize='spacy',
                  batch_first=False)  # , fix_length=400)
LABEL = data.LabelField(dtype=torch.float)


train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL,
                                                        root='/tmp/sst/',
                                                        fine_grained=False,
                                                        filter_pred=lambda ex:
                                                        ex.label != 'neutral')

assert sum(len(data) for data in (train_data, valid_data, test_data)) == 9613
assert len(test_data) == 1821

TEXT.build_vocab(train_data, vectors=GloVe(
    name='6B', dim=100, cache='/tmp/glove/'))
LABEL.build_vocab(train_data)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                           batch_size=32*4,
                                                                           device=device)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters, num_classes, d_prob, mode):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.d_prob = d_prob
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.load_embeddings()
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                             out_channels=num_filters,
                                             kernel_size=k, stride=1) for k in kernel_sizes])
        self.dropout = nn.Dropout(d_prob)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.LogSoftmax(dim=1)

    def forward(self, x):
        sequence_length, batch_size = x.shape
        x = self.embedding(x).transpose(1, 2)
        x = [F.relu(conv(x)) for conv in self.conv]
        #x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = [F.adaptive_max_pool1d(c, 1).squeeze(dim=-1)
             for c in x]  # global max pool
        x = torch.cat(x, dim=1)
        x = self.fc(self.dropout(x))
        return self.activation(x).squeeze()

    def load_embeddings(self):
        if 'static' in self.mode:
            self.embedding.weight.data.copy_(TEXT.vocab.vectors)
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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = Conv1dFlipout(n_inputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = Conv1dFlipout(n_outputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2)
        self.downsample = Conv1dFlipout(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
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
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TextTCN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, kernel_size, num_classes, d_prob, mode):
        super(TextTCN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.d_prob = d_prob
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.load_embeddings()
        self.tcn = TemporalConvNet(num_inputs=embedding_dim,
                                   num_channels=[hidden_dim] * num_layers,
                                   kernel_size=kernel_size,
                                   dropout=d_prob)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.LogSoftmax(dim=1)

    def forward(self, x):
        sequence_length, batch_size = x.shape
        x = self.embedding(x.transpose(1, 0)).transpose(1, 2)
        x = self.tcn(x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)  # global max pool
        x = self.fc(x)
        return self.activation(x).squeeze()

    def kl_loss(self):
        total_loss = 0.0
        for module in self.children():
            if issubclass(type(module), BayesByBackpropModule):
                total_loss += module.kl_loss()
        return total_loss

    def load_embeddings(self):
        if 'static' in self.mode:
            self.embedding.weight.data.copy_(TEXT.vocab.vectors)
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


class TextGRU(BayesByBackpropModule):
    def __init__(self, gru, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, d_prob, mode):
        super(TextGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.d_prob = d_prob
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.load_embeddings()
        self.gru = gru(input_size=embedding_dim,
                       hidden_size=hidden_dim,
                       num_layers=num_layers,
                       #    dropout=d_prob,
                       bias=True)
        self.num_directions = 1
        # self.dropout = nn.Dropout(d_prob)
        # self.fc = LinearFlipout(hidden_dim*self.num_directions, num_classes)
        self.fc = LinearPathwise(hidden_dim*self.num_directions, num_classes)
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.LogSoftmax(dim=1)
        self.debug = True

    def kl_loss(self):
        return self.gru.kl_loss() + self.fc.kl_loss()
        # total_loss = 0.0
        # for module in self.children():
        #     if issubclass(type(module), BayesByBackpropModule):
        #         total_loss += module.kl_loss()
        # return total_loss

    def forward(self, x):
        sequence_length, batch_size = x.shape
        x = self.embedding(x)
        _, x = self.gru(x)
        if self.debug:
            print(x.shape)
        x = x.view(-1,
                   self.num_directions,
                   batch_size,
                   self.hidden_dim)[-1, :, :, :]
        if self.debug:
            print(x.shape)
        if self.debug:
            print(x.shape)
        x = self.fc(x.squeeze())
        if self.debug:
            print(x.shape)
        self.debug = False
        return self.activation(x).squeeze()

    def load_embeddings(self):
        if 'static' in self.mode:
            self.embedding.weight.data.copy_(TEXT.vocab.vectors)
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


vocab_size, embedding_dim = TEXT.vocab.vectors.shape

cnn = TextCNN(vocab_size=vocab_size,
              embedding_dim=embedding_dim,
              kernel_sizes=[3, 4, 5],
              num_filters=100,
              num_classes=1,
              d_prob=0.5,
              mode='static')

tcn = TextTCN(vocab_size=vocab_size,
              embedding_dim=embedding_dim,
              hidden_dim=100,
              num_layers=3,  # 2
              kernel_size=5,  # 2
              num_classes=1,
              d_prob=0.5,
              mode='static')

num_layers = 1
d_prob = 0.5


gru_ = GRUFlipout  # 5 times slower on IMDB
gru_ = GRUPathwise
# gru_ = nn.GRU

gru = TextGRU(gru_,
              vocab_size=vocab_size,
              embedding_dim=embedding_dim,
              hidden_dim=100,
              num_layers=num_layers,
              num_classes=1,
              d_prob=d_prob,
              mode='static')

model = gru
# model = tcn

model.to(device)
# , weight_decay=1e-3) #, lr=1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = ELBO(model, nn.BCELoss())
# criterion = nn.BCELoss()

print(model)


def train_on_batch(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch.text, batch.label
    y_pred = model(x)
    loss = criterion(y_pred, y)
    # loss += model.kl_loss()/len(y)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_on_batch(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch.text, batch.label
        y_pred = model(x)
        return y_pred, y


trainer = Engine(train_on_batch)
train_evaluator = Engine(eval_on_batch)
validation_evaluator = Engine(eval_on_batch)

RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')


def predict_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y


Accuracy(output_transform=predict_transform).attach(
    train_evaluator, 'accuracy')
Loss(criterion).attach(train_evaluator, 'bce')

Accuracy(output_transform=predict_transform).attach(
    validation_evaluator, 'accuracy')
Loss(criterion).attach(validation_evaluator, 'bce')

pbar = ProgressBar(persist=True, bar_format="")
pbar.attach(trainer, ['loss'])


def score_function(engine):
    val_loss = engine.state.metrics['bce']
    return -val_loss


handler = EarlyStopping(
    patience=10, score_function=score_function, trainer=trainer)
# validation_evaluator.add_event_handler(Events.COMPLETED, handler)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_iterator)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_bce = metrics['bce']
    pbar.log_message(
        "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, avg_accuracy, avg_bce))


def log_validation_results(engine):
    validation_evaluator.run(valid_iterator)
    metrics = validation_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_bce = metrics['bce']
    pbar.log_message(
        "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, avg_accuracy, avg_bce))
    pbar.n = pbar.last_print_n = 0


trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

checkpointer = ModelCheckpoint('/tmp/models', 'sst_model', save_interval=1,
                               n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED,
                          checkpointer, {'sst_model': model})

trainer.run(train_iterator, max_epochs=10*3)
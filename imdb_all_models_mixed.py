# -*- coding: utf-8 -*-
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
from variational import (GRUFlipout, GRUPathwise, LinearFlipout,
                         LinearPathwise, VariationalModule)

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

TEXT = data.Field(lower=True, fix_length=400)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='/tmp/imdb/')
train_data, valid_data = train_data.split(
    split_ratio=0.8, random_state=random.seed(SEED))

print('type(train_data) : ', type(train_data))
print('type(train_data[0]) : ', type(train_data[0]))
example = train_data[0]
print('Attributes of Example : ', [
      attr for attr in dir(example) if '_' not in attr])
print('example.label : ', example.label)
print('example.text[:10] : ', example.text[:10])

TEXT.build_vocab(train_data, vectors=GloVe(
    name='6B', dim=100, cache='/tmp/glove/'))
print('Attributes of TEXT : ', [attr for attr in dir(TEXT) if '_' not in attr])
print('Attributes of TEXT.vocab : ', [
      attr for attr in dir(TEXT.vocab) if '_' not in attr])
print('First 5 values TEXT.vocab.itos : ', TEXT.vocab.itos[0:5])
print('First 5 key, value pairs of TEXT.vocab.stoi : ', {
      key: value for key, value in list(TEXT.vocab.stoi.items())[0:5]})
print('Shape of TEXT.vocab.vectors.shape : ', TEXT.vocab.vectors.shape)

LABEL.build_vocab(train_data)
print('Attributes of LABEL : ', [
      attr for attr in dir(LABEL) if '_' not in attr])
print('Attributes of LABEL.vocab : ', [
      attr for attr in dir(LABEL.vocab) if '_' not in attr])
print('First 5 values LABEL.vocab.itos : ', LABEL.vocab.itos)
print('First 5 key, value pairs of LABEL.vocab.stoi : ', {
      key: value for key, value in list(LABEL.vocab.stoi.items())})
print('Shape of LABEL.vocab.vectors : ', LABEL.vocab.vectors)

BATCH_SIZE = 32*1
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                           batch_size=BATCH_SIZE,
                                                                           device=device)


batch = next(iter(train_iterator))
print('batch.label[0] : ', batch.label[0])
print('batch.text[0] : ', batch.text[0][batch.text[0] != 1])

lengths = []
for i, batch in enumerate(train_iterator):
    x = batch.text
    lengths.append(x.shape[1])
    if i == 10:
        break

print('Lengths of first 10 batches : ', lengths)


class TextGRU(VariationalModule):
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
                       bias=True)
        self.num_directions = 1
        self.dropout = nn.Dropout(d_prob)
        self.fc = LinearPathwise(hidden_dim*self.num_directions, num_classes)
        # self.fc = nn.Linear(hidden_dim*self.num_directions, num_classes)
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.LogSoftmax(dim=1)
        self.debug = True

    def kl_loss(self):
        total_loss = 0.0
        for module in self.children():
            if issubclass(type(module), VariationalModule):
                total_loss += module.kl_loss()
        return total_loss

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
        # if self.gru.bidirectional:
        #  x = torch.cat((x[0,:,:], x[1,:,:]), dim = 1)
        if self.debug:
            print(x.shape)
        x = self.fc(self.dropout(x.squeeze()))
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

num_layers = 2
d_prob = 0

gru_ = GRUFlipout  # 5 times slower
gru_ = GRUPathwise
gru_ = nn.GRU

gru = TextGRU(gru_,
              vocab_size=vocab_size,
              embedding_dim=embedding_dim,
              hidden_dim=100,
              num_layers=num_layers,
              num_classes=1,
              d_prob=d_prob,
              mode='static')

model = gru

model.to(device)
# , weight_decay=1e-3) #, lr=1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

print(model)


def train_on_batch(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch.text, batch.label
    y_pred = model(x)
    #assert (y.data.cpu().numpy().all() >= 0. & y.data.cpu().numpy().all() <= 1.)
    #print("isnan = ", torch.isnan(y_pred).all().item() == 1)
    assert torch.ge(y_pred, torch.zeros_like(y_pred)).all() and torch.le(y_pred, torch.ones_like(y_pred)).all()
    loss = criterion(y_pred, y)
    loss += model.kl_loss()/len(y)
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
#validation_evaluator.add_event_handler(Events.COMPLETED, handler)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_iterator)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_bce = metrics['bce']
    pbar.log_message(
        "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, avg_accuracy, avg_bce))

# @trainer.on(Events.EPOCH_COMPLETED)


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

checkpointer = ModelCheckpoint('/tmp/models', 'textcnn', save_interval=1,
                               n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED,
                          checkpointer, {'textcnn': model})

trainer.run(train_iterator, max_epochs=5*10)

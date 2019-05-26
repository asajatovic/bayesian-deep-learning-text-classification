import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss, RunningAverage
from models import TextLSTM, TextTCN
from models_variational import TextLSTM as TextLSTMVariational
from models_variational import TextTCN as TextTCNVariational
from torchtext import data, datasets
from torchtext.vocab import GloVe
from variational import ELBO
from yelp_reviews import YELP

parser = argparse.ArgumentParser(
    description='PyTorch Text Classification training')
parser.add_argument('--batch_size', default=128, type=float, help='batch_size')
parser.add_argument('--hidden', default=100, type=int, help='hidden_size')
parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
parser.add_argument('--model', type=str, help='tcn or lstm')
parser.add_argument('--epochs', default=20, type=int, help='max_epochs')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', type=str, help='sst or imdb')
parser.add_argument('--variational', type=bool, default=False,
                    help='variational or ordinary model')
args = parser.parse_args()

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on: {device}')

TEXT = data.Field(fix_length=None, lower=True,
                  tokenize='spacy', batch_first=False)
LABEL = data.LabelField(dtype=torch.float)

dataset = args.dataset
if dataset == 'imdb':
    train_data, test_data = datasets.IMDB.splits(
        TEXT, LABEL, root='./data/imdb/')
    train_data, valid_data = train_data.split(
        split_ratio=0.8, random_state=random.seed(SEED))

if dataset == 'sst':
    train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL,
                                                            root='./data/sst/',
                                                            fine_grained=False,
                                                            filter_pred=lambda ex:
                                                            ex.label != 'neutral')

if dataset == 'yelp':
    train_data, test_data = YELP.splits(
        TEXT, LABEL, root='./data/yelp/')
    train_data, valid_data = train_data.split(
        split_ratio=0.8, random_state=random.seed(SEED))
    LABEL = data.LabelField(dtype=torch.long)

TEXT.build_vocab(train_data, vectors=GloVe(
    name='6B', dim=100, cache='./glove/'))
LABEL.build_vocab(train_data)

batch_size = args.batch_size
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                           batch_size=batch_size,
                                                                           device=device)

vocab_size, embedding_dim = TEXT.vocab.vectors.shape

model_name = args.model
hidden_dim = args.hidden
num_labels = len(LABEL.vocab.itos)
num_classes = 1 if num_labels == 2 else num_labels
dropout = args.dropout

variational = args.variational
if variational is True:
    if model_name == 'lstm':
        model = TextLSTMVariational(vocab_size=vocab_size,
                                    embedding_dim=embedding_dim,
                                    hidden_dim=hidden_dim,
                                    num_layers=2,
                                    num_classes=num_classes,
                                    d_prob=dropout,
                                    mode='static',
                                    weights=TEXT.vocab.vectors)

    if model_name == 'tcn':
        model = TextTCNVariational(vocab_size=vocab_size,
                                   embedding_dim=embedding_dim,
                                   hidden_dim=hidden_dim,
                                   num_layers=3,
                                   kernel_size=5,
                                   num_classes=num_classes,
                                   d_prob=dropout,
                                   mode='static',
                                   weights=TEXT.vocab.vectors)
    model_name += "_variational"
else:
    if model_name == 'lstm':
        model = TextLSTM(vocab_size=vocab_size,
                         embedding_dim=embedding_dim,
                         hidden_dim=hidden_dim,
                         num_layers=2,
                         num_classes=num_classes,
                         d_prob=dropout,
                         mode='static',
                         weights=TEXT.vocab.vectors)

    if model_name == 'tcn':
        model = TextTCN(vocab_size=vocab_size,
                        embedding_dim=embedding_dim,
                        hidden_dim=hidden_dim,
                        num_layers=3,
                        kernel_size=5,
                        num_classes=num_classes,
                        d_prob=dropout,
                        mode='static',
                        weights=TEXT.vocab.vectors)


if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

model.to(device)
lr = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss() if num_classes == 1 else nn.NLLLoss()
if variational is True:
    print("Using ELBO loss")
    criterion = ELBO(model, criterion)

print(model)


def train_on_batch(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch.text, batch.label
    y_pred = model(x)
    loss = criterion(y_pred, y)
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
    # modify for softmax
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
    val_acc = engine.state.metrics['accuracy']
    return val_acc


handler = EarlyStopping(
    patience=5, score_function=score_function, trainer=trainer)
validation_evaluator.add_event_handler(Events.COMPLETED, handler)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_iterator)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_bce = metrics['bce']
    pbar.log_message(
        "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, avg_accuracy, avg_bce))


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    validation_evaluator.run(valid_iterator)
    metrics = validation_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_bce = metrics['bce']
    pbar.log_message(
        "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, avg_accuracy, avg_bce))
    pbar.n = pbar.last_print_n = 0


checkpointer = ModelCheckpoint('./saved_models', f'best_{model_name}_{dataset}', save_interval=1,
                               n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED,
                          checkpointer, {f'best_{model_name}_{dataset}': model})

max_epochs = args.epochs
trainer.run(train_iterator, max_epochs=max_epochs)

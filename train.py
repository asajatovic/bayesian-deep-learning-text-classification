import argparse
import math
import os
import random
import time

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
parser.add_argument('--lr', default=0.01, type=float, help='learning_rate')
parser.add_argument('--model', type=str, help='tcn or lstm')
parser.add_argument('--layers', type=int, default=2, help='num_layers')
parser.add_argument('--kernel', type=int, default=3, help='kernel_size')
parser.add_argument('--epochs', default=30, type=int, help='max_epochs')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--prior', default="normal", type=str, help='prior type (normal or laplace)')
parser.add_argument('--dataset', type=str, help='sst, imdb or yelp')
parser.add_argument('--variational', type=str, default='no',
                    help='variational or ordinary model')
args = parser.parse_args()

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
rand_state = random.getstate()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on: {device}')
print("Preparing data...")
start_time = time.time()

dataset = args.dataset
if dataset == 'imdb':
    TEXT = data.Field(fix_length=400, lower=True,
                      tokenize='spacy', batch_first=False)
    LABEL = data.LabelField(dtype=torch.long)
    train_data, test_data = datasets.IMDB.splits(
        TEXT, LABEL, root='./data/imdb/')
    train_data, valid_data = train_data.split(
        split_ratio=0.8, stratified=False, random_state=rand_state)

if dataset == 'sst':
    TEXT = data.Field(fix_length=30, lower=True,
                      tokenize='spacy', batch_first=False)
    LABEL = data.LabelField(dtype=torch.long)
    train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL,
                                                            root='./data/sst/',
                                                            fine_grained=False,
                                                            filter_pred=lambda ex:
                                                            ex.label != 'neutral')

if dataset == 'yelp':
    TEXT = data.Field(fix_length=300, lower=True,
                      tokenize='spacy', batch_first=False)
    LABEL = data.LabelField(dtype=torch.long)
    train_data, test_data = YELP.splits(
        TEXT, LABEL, root='./data/yelp/')
    train_data, valid_data = train_data.split(
        split_ratio=0.9, stratified=False, random_state=rand_state)

TEXT.build_vocab(train_data, vectors=GloVe(
    name='6B', dim=100, cache='./glove/'))
LABEL.build_vocab(train_data)

batch_size = args.batch_size
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                           batch_size=batch_size,
                                                                           device=device)

total_time = time.time() - start_time
print(f"...prepared data in {int(total_time/60)} minutes")
TRAIN_BATCHES = len(train_iterator)
VALID_BATCHES = len(valid_iterator)

vocab_size, embedding_dim = TEXT.vocab.vectors.shape

model_name = args.model
hidden_dim = args.hidden
num_layers = args.layers
kernel_size = args.kernel
num_labels = len(LABEL.vocab.itos)
num_classes = num_labels
dropout = args.dropout

variational = args.variational
if variational == 'yes':
    prior_type = args.prior
    print(prior_type)
    if model_name == 'lstm':
        model = TextLSTMVariational(vocab_size=vocab_size,
                                    embedding_dim=embedding_dim,
                                    hidden_dim=hidden_dim,
                                    num_layers=num_layers,
                                    num_classes=num_classes,
                                    mode='static',
                                    weights=TEXT.vocab.vectors,
                                    prior_type=prior_type)
    lr = 0.01
    if model_name == 'tcn':
        model = TextTCNVariational(vocab_size=vocab_size,
                                   embedding_dim=embedding_dim,
                                   hidden_dim=hidden_dim,
                                   num_layers=num_layers,
                                   kernel_size=kernel_size,
                                   num_classes=num_classes,
                                   mode='static',
                                   weights=TEXT.vocab.vectors,
                                   prior_type=prior_type)
    model_name += "_variational_" + prior_type
else:
    if model_name == 'lstm':
        model = TextLSTM(vocab_size=vocab_size,
                         embedding_dim=embedding_dim,
                         hidden_dim=hidden_dim,
                         num_layers=num_layers,
                         num_classes=num_classes,
                         d_prob=dropout,
                         mode='static',
                         weights=TEXT.vocab.vectors)
    lr = 0.01
    if model_name == 'tcn':
        model = TextTCN(vocab_size=vocab_size,
                        embedding_dim=embedding_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        kernel_size=kernel_size,
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
train_criterion = nn.NLLLoss()
valid_criterion = nn.NLLLoss()
if variational is True:
    print("Using ELBO loss")
    train_criterion = ELBO(model, criterion, TRAIN_BATCHES)
    valid_criterion = ELBO(model, criterion, VALID_BATCHES)

print(model)


def train_on_batch(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch.text, batch.label
    y_pred = model(x)
    loss = train_criterion(y_pred, y)
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

Accuracy().attach(train_evaluator, 'accuracy')
Loss(train_criterion).attach(train_evaluator, 'loss')

Accuracy().attach(validation_evaluator, 'accuracy')
Loss(valid_criterion).attach(validation_evaluator, 'loss')

pbar = ProgressBar(persist=True, bar_format="")
pbar.attach(trainer, ['loss'])


def score_function(engine):
    val_acc = engine.state.metrics['accuracy']
    return val_acc


handler = EarlyStopping(patience=10,
                        score_function=score_function,
                        trainer=trainer)
validation_evaluator.add_event_handler(Events.COMPLETED, handler)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_iterator)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    pbar.log_message(
        "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, avg_accuracy, avg_loss))


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    validation_evaluator.run(valid_iterator)
    metrics = validation_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    pbar.log_message(
        "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, avg_accuracy, avg_loss))
    pbar.n = pbar.last_print_n = 0

filename_prefix = 'best'
filename_prefix += '_lr' if lr==0.0001 else ""
filename_prefix += '_drop' if dropout==0.5 else ""
filename_prefix += '_NO_drop' if dropout==0.0 else ""
checkpointer = ModelCheckpoint(dirname=f'./models_final/{dataset}',
                               filename_prefix=filename_prefix,
                               # save_interval=2,
                               score_function=score_function,
                               score_name='val_acc',
                               n_saved=1, require_empty=False,
                               create_dir=True,
                               save_as_state_dict=False)
validation_evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                       checkpointer, {model_name: model})

max_epochs = args.epochs
trainer.run(train_iterator, max_epochs=max_epochs)

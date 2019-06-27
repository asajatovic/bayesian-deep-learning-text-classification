import argparse
import math
import os
import random
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

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
parser.add_argument('--model', type=str, help='path to trained model')
parser.add_argument('--dataset', type=str, help='sst, imdb or yelp')
parser.add_argument('--T', type=int, default=10)
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
batch_size = 128
print(device)

def load_imdb():
    TEXT = data.Field(fix_length=400, lower=True,
                      tokenize='spacy', batch_first=False)
    LABEL = data.LabelField(dtype=torch.long)
    train_data, test_data = datasets.IMDB.splits(
        TEXT, LABEL, root='./data/imdb/')
    train_data, valid_data = train_data.split(
        split_ratio=0.8, stratified=False, random_state=rand_state)
    TEXT.build_vocab(train_data, vectors=GloVe(
    name='6B', dim=100, cache='./glove/'))
    print(len(TEXT.vocab.itos))
    print()
    LABEL.build_vocab(train_data)
    return train_data, valid_data, test_data, TEXT.vocab.vectors

def load_sst():
    TEXT = data.Field(fix_length=30, lower=True,
                      tokenize='spacy', batch_first=False)
    LABEL = data.LabelField(dtype=torch.long)
    train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL,
                                                            root='./data/sst/',
                                                            fine_grained=False,
                                                            filter_pred=lambda ex:
                                                            ex.label != 'neutral')
    TEXT.build_vocab(train_data, vectors=GloVe(
    name='6B', dim=100, cache='./glove/'))
    LABEL.build_vocab(train_data)
    return train_data, valid_data, test_data, TEXT.vocab.vectors

def load_yelp():
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
    return train_data, valid_data, test_data, TEXT.vocab.vectors


model_path = args.model
model = torch.load(model_path)
model.to(device)
print(model)

dataset = args.dataset
if dataset == 'imdb':
    _, _, test_data, vectors = load_imdb()
if dataset == 'sst':
    _, _, test_data, vectors = load_sst()
if dataset == 'yelp':
    _, _, test_data, vectors = load_yelp()

iterator = data.BucketIterator(test_data, batch_size=batch_size, device=device)

def predict(model, iterator):
    def predict_on_batch(model, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch.text, batch.label
            y_pred = model(x)
        return y_pred, y
    
    preds = []
    scores = []
    targets = []
    
    for batch in iterator:
        batch_scores, batch_targets = predict_on_batch(model, batch)
        batch_preds = torch.argmax(batch_scores, dim=1)      
        preds += [batch_preds.cpu().numpy()]
        scores += [batch_scores.cpu().numpy()]
        targets += [batch_targets.cpu().numpy()]
    
    preds = np.hstack(preds)
    scores = np.vstack(scores)
    targets = np.hstack(targets)
    print(preds.shape)
    print(scores.shape)
    print(targets.shape)
    
    return preds, scores, targets


def bbb_predict(model, iterator, T=10):
    def bbb_predict_on_batch(model, batch, T):
        model.eval()
        preds = []
        x, y = batch.text, batch.label
        targets = y
        for _ in range(T):
            with torch.no_grad():
                y_pred = model(x)
                preds += [y_pred]
        preds = torch.stack(preds)
        pred_means = torch.mean(preds, dim=0)
        pred_vars = torch.var(preds, 0, False, False)
        pred_vars = torch.mean(pred_vars, dim=-1)
        return pred_means, targets, pred_vars
    
    preds = []
    scores = []
    targets = []
    variances = []
    
    for batch in iterator:
        batch_scores, batch_targets, batch_vars = bbb_predict_on_batch(model, batch, T)
        batch_preds = torch.argmax(batch_scores, dim=1)      
        preds += [batch_preds.cpu().numpy()]
        scores += [batch_scores.cpu().numpy()]
        targets += [batch_targets.cpu().numpy()]
        variances += [batch_vars.cpu().numpy()]
    
    preds = np.hstack(preds)
    scores = np.vstack(scores)
    targets = np.hstack(targets)
    variances = np.hstack(variances)
    print(preds.shape)
    print(scores.shape)
    print(targets.shape)
    print(variances.shape)
    
    return preds, scores, targets, variances

variational = args.variational
if variational == 'yes':
    T=args.T
    preds, scores, targets, variances = bbb_predict(model, iterator, T=T)
else:
    preds, scores, targets = predict(model, iterator)

acc = np.mean(preds == targets)
print("Test set accuracy is: ", acc)

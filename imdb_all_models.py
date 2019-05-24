# -*- coding: utf-8 -*-
"""IMDB_all_models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dkV7_cfZUJAKYwAjcEay_51wLp3dCEuX

# Convolutional Neural Networks for Sentence Classification using Ignite

This is a tutorial on using Ignite to train neural network models, setup experiments and validate models.

In this experiment, we'll be replicating [
Convolutional Neural Networks for Sentence Classification by Yoon Kim](https://arxiv.org/abs/1408.5882)! This paper uses CNN for text classification, a task typically reserved for RNNs, Logistic Regression, Naive Bayes.

We want to be able to classify IMDB movie reviews and predict whether the review is positive or negative. IMDB Movie Review dataset comprises of 25000 positive and 25000 negative examples. The dataset comprises of text and label pairs. This is binary classification problem. We'll be using PyTorch to create the model, torchtext to import data and Ignite to train and monitor the models!

Lets get started!

## Required Dependencies 

In this example we only need torchvision package, assuming that `torch` and `ignite` are already installed. We can install it using `pip`:

`pip install torchtext spacy`

`python -m spacy download en`
"""

# pip install pytorch-ignite

"""## Import Libraries"""

import random

"""`torchtext` is a library that provides multiple datasets for NLP tasks, similar to `torchvision`. Below we import the following:
* **data**: A module to setup the data in the form Fields and Labels.
* **datasets**: A module to download NLP datasets.
* **GloVe**: A module to download and use pretrained GloVe embedings.
"""

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

"""We import torch, nn and functional modules to create our models!"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

"""`Ignite` is a High-level library to help with training neural networks in PyTorch. It comes with an `Engine` to setup a training loop, various metrics, handlers and a helpful contrib section! 

Below we import the following:
* **Engine**: Runs a given process_function over each batch of a dataset, emitting events as it goes.
* **Events**: Allows users to attach functions to an `Engine` to fire functions at a specific event. Eg: `EPOCH_COMPLETED`, `ITERATION_STARTED`, etc.
* **Accuracy**: Metric to calculate accuracy over a dataset, for binary, multiclass, multilabel cases. 
* **Loss**: General metric that takes a loss function as a parameter, calculate loss over a dataset.
* **RunningAverage**: General metric to attach to Engine during training. 
* **ModelCheckpoint**: Handler to checkpoint models. 
* **EarlyStopping**: Handler to stop training based on a score function. 
* **ProgressBar**: Handler to create a tqdm progress bar.
"""

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar

"""## Processing Data

The code below first sets up `TEXT` and `LABEL` as general data objects. 

* `TEXT` converts any text to lowercase and produces tensors with the batch dimension first. 
* `LABEL` is a data object that will convert any labels to floats.

Next IMDB training and test datasets are downloaded, the training data is split into training and validation datasets. It takes TEXT and LABEL as inputs so that the data is processed as specified.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

TEXT = data.Field(lower=True, fix_length=400)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='/tmp/imdb/')
train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))

"""Now we have three sets of the data - train, validation, test. Let's explore what these data objects are and how to extract data from them.
* `train_data` is **torchtext.data.dataset.Dataset**, this is similar to **torch.utils.data.Dataset**.
* `train_data[0]` is **torchtext.data.example.Example**, a Dataset is comprised of many Examples.
* Let's explore the attributes of an example. We see a few methods, but most importantly we see `label` and `text`.
* `example.text` is the text of that example and `example.label` is the label of the example.
"""

print('type(train_data) : ', type(train_data))
print('type(train_data[0]) : ',type(train_data[0]))
example = train_data[0]
print('Attributes of Example : ', [attr for attr in dir(example) if '_' not in attr])
print('example.label : ', example.label)
print('example.text[:10] : ', example.text[:10])

"""Now that we have an idea of what are split datasets look like, lets dig further into `TEXT` and `LABEL`. It is important that we build our vocabulary based on the train dataset as validation and test are **unseen** in our experimenting. 

For `TEXT`, let's download the pretrained **GloVE** 100 dimensional word vectors. This means each word is described by 100 floats! If you want to read more about this, here are a few resources.
* [StanfordNLP - GloVe](https://github.com/stanfordnlp/GloVe)
* [DeepLearning.ai Lecture](https://www.coursera.org/lecture/nlp-sequence-models/glove-word-vectors-IxDTG)
* [Stanford CS224N Lecture by Richard Socher](https://www.youtube.com/watch?v=ASn7ExxLZws)

We use `TEXT.build_vocab` to do this, let's explore the `TEXT` object more. 

Let's explore what `TEXT` object is and how to extract data from them. We see `TEXT` has a few attributes, let's explore vocab, since we just used the build_vocab function. 

`TEXT.vocab` has the following attributes:
* `extend` is used to extend the vocabulary
* `freqs` is a dictionary of the frequency of each word
* `itos` is a list of all the words in the vocabulary.
* `stoi` is a dictionary mapping every word to an index.
* `vectors` is a torch.Tensor of the downloaded embeddings
"""

TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=100, cache='/tmp/glove/'))
print ('Attributes of TEXT : ', [attr for attr in dir(TEXT) if '_' not in attr])
print ('Attributes of TEXT.vocab : ', [attr for attr in dir(TEXT.vocab) if '_' not in attr])
print ('First 5 values TEXT.vocab.itos : ', TEXT.vocab.itos[0:5]) 
print ('First 5 key, value pairs of TEXT.vocab.stoi : ', {key:value for key,value in list(TEXT.vocab.stoi.items())[0:5]}) 
print ('Shape of TEXT.vocab.vectors.shape : ', TEXT.vocab.vectors.shape)

"""Let's do the same with `LABEL`. We see that there are vectors related to `LABEL`, this is expected because `LABEL` provides the definition of a label of data."""

LABEL.build_vocab(train_data)
print ('Attributes of LABEL : ', [attr for attr in dir(LABEL) if '_' not in attr])
print ('Attributes of LABEL.vocab : ', [attr for attr in dir(LABEL.vocab) if '_' not in attr])
print ('First 5 values LABEL.vocab.itos : ', LABEL.vocab.itos) 
print ('First 5 key, value pairs of LABEL.vocab.stoi : ', {key:value for key,value in list(LABEL.vocab.stoi.items())})
print ('Shape of LABEL.vocab.vectors : ', LABEL.vocab.vectors)

"""Now we must convert our split datasets into iterators, we'll take advantage of **torchtext.data.BucketIterator**! BucketIterator pads every element of a batch to the length of the longest element of the batch."""

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data), 
                                                                           batch_size=32*1,
                                                                           device=device)

"""Let's actually explore what the output of the iterator is, this way we'll know what the input of the model is, how to compare the label to the output and how to setup are process_functions for Ignite's `Engine`.
* `batch.label[0]` is the label of a single example. We can see that `LABEL.vocab.stoi` was used to map the label that originally text into a float.
* `batch.text[0]` is the text of a single example. Similar to label, `TEXT.vocab.stoi` was used to convert each token of the example's text into indices.

Now let's print the lengths of the sentences of the first 10 batches of `train_iterator`. We see here that all the batches are of different lengths, this means that the bucket iterator is doing exactly as we hoped!
"""

batch = next(iter(train_iterator))
print('batch.label[0] : ', batch.label[0])
print('batch.text[0] : ', batch.text[0][batch.text[0] != 1])

lengths = []
for i, batch in enumerate(train_iterator):
    x = batch.text
    lengths.append(x.shape[1])
    if i == 10:
        break

print ('Lengths of first 10 batches : ', lengths)

"""## TextCNN Model

Here is the replication of the model, here are the operations of the model:
* **Embedding**: Embeds a batch of text of shape (N, L) to (N, L, D), where N is batch size, L is maximum length of the batch, D is the embedding dimension. 

* **Convolutions**: Runs parallel convolutions across the embedded words with kernel sizes of 3, 4, 5 to mimic trigrams, four-grams, five-grams. This results in outputs of (N, L - k + 1, D) per convolution, where k is the kernel_size. 

* **Activation**: ReLu activation is applied to each convolution operation.

* **Pooling**: Runs parallel maxpooling operations on the activated convolutions with window sizes of L - k + 1, resulting in 1 value per channel i.e. a shape of (N, 1, D) per pooling. 

* **Concat**: The pooling outputs are concatenated and squeezed to result in a shape of (N, 3D). This is a single embedding for a sentence.

* **Dropout**: Dropout is applied to the embedded sentence. 

* **Fully Connected**: The dropout output is passed through a fully connected layer of shape (3D, 1) to give a single output for each example in the batch. sigmoid is applied to the output of this layer.

* **load_embeddings**: This is a method defined for TextCNN to load embeddings based on user input. There are 3 modes - rand which results in randomly initialized weights, static which results in frozen pretrained weights, nonstatic which results in trainable pretrained weights. 


Let's note that this model works for variable text lengths! The idea to embed the words of a sentence, use convolutions, maxpooling and concantenation to embed the sentence as a single vector! This single vector is passed through a fully connected layer with sigmoid to output a single value. This value can be interpreted as the probability a sentence is positive (closer to 1) or negative (closer to 0).

The minimum length of text expected by the model is the size of the smallest kernel size of the model.
"""

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
        x = [F.adaptive_max_pool1d(c, 1).squeeze(dim=-1) for c in x] # global max pool
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
            raise ValueError('Unexpected value of mode. Please choose from static, nonstatic, rand.')

from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

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
        self.dropout = nn.Dropout(d_prob)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.LogSoftmax(dim=1)

    def forward(self, x):
        sequence_length, batch_size = x.shape
        x = self.embedding(x.transpose(0, 1))
        x = self.tcn(x.transpose(1,2))
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1) # global max pool
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
            raise ValueError('Unexpected value of mode. Please choose from static, nonstatic, rand.')

class MyGRUCell(nn.Module):

    __constants__ = ['input2hidden, hidden2hidden']

    def __init__(self, input_size, hidden_size, bias=True):
        super(MyGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(input_size, hidden_size * 3, bias)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size * 3, bias)
        self.reset_parameters()

    def reset_parameters(self):
        import math
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state=None):
        #input = input.view(-1, input.size(1))
        gate_input = self.input2hidden(input).squeeze()
        if state is None:
            state = input.new_zeros(input.size(0),
                                    self.hidden_size,
                                    requires_grad=False)
        gate_hidden = self.hidden2hidden(state).squeeze()

        i_r, i_i, i_n = gate_input.chunk(3, 1)
        h_r, h_i, h_n = gate_hidden.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        output = newgate + inputgate * (state - newgate)
        return output

class GRULayer(nn.Module):
  
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRULayer, self).__init__()
        #self.cell = nn.GRUCell(input_size, hidden_size, bias)
        self.cell = MyGRUCell(input_size, hidden_size, bias)
  
    def forward(self, inputs):
        #seq_len, batch, input_size = input.size()
        outputs = []
        out = None
        for input in inputs:
            out = self.cell(input, out)
            outputs += [out]
        return outputs, out
      
      
class MyGRU(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers=2, bias=True):
        super(MyGRU, self).__init__()
        layers = [GRULayer(input_size, hidden_size, bias)]
        layers += [GRULayer(hidden_size, hidden_size, bias)]
        self.layers = nn.ModuleList(layers)
  
    def forward(self, inputs):
        #seq_len, batch, input_size = input.size()
        states = []
        out = inputs.unbind(0)
        for layer in self.layers:
            out, state = layer(out)
            states += [state]
        return torch.stack(out), torch.stack(states)
    

class TextGRU(nn.Module):
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
        self.fc = nn.Linear(hidden_dim*self.num_directions, num_classes)
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.LogSoftmax(dim=1)
        self.debug = True

    def forward(self, x):
        sequence_length, batch_size = x.shape
        x = self.embedding(x)
        _, x = self.gru(x)
        if self.debug: print(x.shape)
        x = x.view(-1,
                   self.num_directions, 
                   batch_size, 
                   self.hidden_dim)[-1,:,:,:]
        if self.debug: print(x.shape)
        #if self.gru.bidirectional:
        #  x = torch.cat((x[0,:,:], x[1,:,:]), dim = 1)
        if self.debug: print(x.shape)
        x = self.fc(self.dropout(x.squeeze()))
        if self.debug: print(x.shape)
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
            raise ValueError('Unexpected value of mode. Please choose from static, nonstatic, rand.')

"""## Creating Model, Optimizer and Loss

Below we create an instance of the TextCNN model and load embeddings in **static** mode. The model is placed on a device and then a loss function of Binary Cross Entropy and Adam optimizer are setup.
"""

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
                num_layers=3,  #2
                kernel_size=5, #2
                num_classes=1, 
                d_prob=0.5,
                mode='static')

num_layers = 2
d_prob = 0


gru_ = MyGRU # 5 times slower
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
# model = tcn

model.to(device)
optimizer = torch.optim.Adam(model.parameters())#, weight_decay=1e-3) #, lr=1e-4
criterion = nn.BCELoss()

print(model)

"""## Training and Evaluating using Ignite

### Trainer Engine - process_function

Ignite's Engine allows user to define a process_function to process a given batch, this is applied to all the batches of the dataset. This is a general class that can be applied to train and validate models! A process_function has two parameters engine and batch. 


Let's walk through what the function of the trainer does:

* Sets model in train mode. 
* Sets the gradients of the optimizer to zero.
* Generate x and y from batch.
* Performs a forward pass to calculate y_pred using model and x.
* Calculates loss using y_pred and y.
* Performs a backward pass using loss to calculate gradients for the model parameters.
* model parameters are optimized using gradients and optimizer.
* Returns scalar loss. 

Below is a single operation during the trainig process. This process_function will be attached to the training engine.
"""

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

"""### Evaluator Engine - process_function

Similar to the training process function, we setup a function to evaluate a single batch. Here is what the eval_function does:

* Sets model in eval mode.
* Generates x and y from batch.
* With torch.no_grad(), no gradients are calculated for any succeding steps.
* Performs a forward pass on the model to calculate y_pred based on model and x.
* Returns y_pred and y.

Ignite suggests attaching metrics to evaluators and not trainers because during the training the model parameters are constantly changing and it is best to evaluate model on a stationary model. This information is important as there is a difference in the functions for training and evaluating. Training returns a single scalar loss. Evaluating returns y_pred and y as that output is used to calculate metrics per batch for the entire dataset.

All metrics in Ignite require y_pred and y as outputs of the function attached to the Engine.
"""

def eval_on_batch(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch.text, batch.label
        y_pred = model(x)
        return y_pred, y

"""### Instantiating Training and Evaluating Engines

Below we create 3 engines, a trainer, a training evaluator and a validation evaluator. You'll notice that train_evaluator and validation_evaluator use the same function, we'll see later why this was done!
"""

trainer = Engine(train_on_batch)
train_evaluator = Engine(eval_on_batch)
validation_evaluator = Engine(eval_on_batch)

"""### Metrics - RunningAverage, Accuracy and Loss

To start, we'll attach a metric of Running Average to track a running average of the scalar loss output for each batch.
"""

RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

"""Now there are two metrics that we want to use for evaluation - accuracy and loss. This is a binary problem, so for Loss we can simply pass the Binary Cross Entropy function as the loss_function. 

For Accuracy, Ignite requires y_pred and y to be comprised of 0's and 1's only. Since our model outputs from a sigmoid layer, values are between 0 and 1. We'll need to write a function that transforms `engine.state.output` which is comprised of y_pred and y. 

Below `thresholded_output_transform` does just that, it rounds y_pred to convert y_pred to 0's and 1's, and then returns rounded y_pred and y. This function is the output_transform function used to transform the `engine.state.output` to achieve `Accuracy`'s desired purpose.

Now, we attach `Loss` and `Accuracy` (with `thresholded_output_transform`) to train_evaluator and validation_evaluator. 

To attach a metric to engine, the following format is used:
* `Metric(output_transform=output_transform, ...).attach(engine, 'metric_name')`
"""

def predict_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y

Accuracy(output_transform=predict_transform).attach(train_evaluator, 'accuracy')
Loss(criterion).attach(train_evaluator, 'bce')

Accuracy(output_transform=predict_transform).attach(validation_evaluator, 'accuracy')
Loss(criterion).attach(validation_evaluator, 'bce')

"""### Progress Bar

Next we create an instance of Ignite's progess bar and attach it to the trainer and pass it a key of `engine.state.metrics` to track. In this case, the progress bar will be tracking `engine.state.metrics['loss']`
"""

pbar = ProgressBar(persist=True, bar_format="")
pbar.attach(trainer, ['loss'])

"""### EarlyStopping - Tracking Validation Loss

Now we'll setup a Early Stopping handler for this training process. EarlyStopping requires a score_function that allows the user to define whatever criteria to stop trainig. In this case, if the loss of the validation set does not decrease in 5 epochs, the training process will stop early.
"""

def score_function(engine):
    val_loss = engine.state.metrics['bce']
    return -val_loss

handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
validation_evaluator.add_event_handler(Events.COMPLETED, handler)

"""### Attaching Custom Functions to Engine at specific Events

Below you'll see ways to define your own custom functions and attaching them to various `Events` of the training process.

The functions below both achieve similar tasks, they print the results of the evaluator run on a dataset. One function does that on the training evaluator and dataset, while the other on the validation. Another difference is how these functions are attached in the trainer engine.

The first method involves using a decorator, the syntax is simple - `@` `trainer.on(Events.EPOCH_COMPLETED)`, means that the decorated function will be attached to the trainer and called at the end of each epoch. 

The second method involves using the add_event_handler method of trainer - `trainer.add_event_handler(Events.EPOCH_COMPLETED, custom_function)`. This achieves the same result as the above.
"""

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

"""### ModelCheckpoint

Lastly, we want to checkpoint this model. It's important to do so, as training processes can be time consuming and if for some reason something goes wrong during training, a model checkpoint can be helpful to restart training from the point of failure.

Below we'll use Ignite's `ModelCheckpoint` handler to checkpoint models at the end of each epoch.
"""

checkpointer = ModelCheckpoint('/tmp/models', 'lstm_torch', save_interval=1, n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'lstm_torch': model})

"""### Run Engine

Next, we'll run the trainer for 10 epochs and monitor results. Below we can see that progess bar prints the loss per iteration, and prints the results of training and validation as we specified in our custom function.
"""

trainer.run(train_iterator, max_epochs=50)

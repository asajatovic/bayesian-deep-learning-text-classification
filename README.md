# Bayesian Deep Learning for Text Classification

Source code repository for my Master Thesis at the [University of Zagreb, Faculty of Electrical Engineering and Computing](https://www.fer.unizg.hr/en).

## Bayesian Deep Learning
Bayesian deep learning merges Bayesian probability theory with deep learning, allowing principled uncertainty estimates from deep learning architectures. 
For an excellent quick introduction to Bayesian deep learning, check out [Demystifying Bayesian Deep Learning](https://ericmjl.github.io/bayesian-deep-learning-demystified/#/IntroductionSlide).
One of the most elegant practical Bayesian deep learning approaches is the Bayes-by-Backprop algorithm, first introduced in the paper titled *Weight Uncertainty in Neural Network*. 
The main idea is to **replace weights with weight distributions** and learn the weight distribution parameters instead of the network parameters directly.
The approach was extended from fully connected networks to both RNNs and CNNs.

Relevant papers:
* [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424)
* [Bayesian Recurrent Neural Networks](https://arxiv.org/abs/1704.02798)
* [Uncertainty Estimations by Softplus normalization in Bayesian Convolutional Neural Networks with Variational Inference](https://arxiv.org/abs/1806.05978)

## Text Classification
Text classification architectures can be expressed as a simple four-step procedure: [embed, encode, attend, predict](https://explosion.ai/blog/deep-learning-formula-nlp). 
The classifiers implemented in this repository omit the attend step, use GloVe embeddings, softmax layer for predictions, and use either an LSTM (Long short-term memory) or TCN (Temporal convolutional network) encoder.

Relevant papers:
* [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
* [Long Short-term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
* [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)

## Usage
Install the `conda` environment from [environment.yml](https://github.com/asajatovic/bayesian-deep-learning-text-classification/blob/master/environment.yml).
Run [train.py](https://github.com/asajatovic/bayesian-deep-learning-text-classification/blob/master/train.py) for training and [test.py](https://github.com/asajatovic/bayesian-deep-learning-text-classification/blob/master/test.py) for testing, with appropriate cmd arguments (see corresponding `.py` files for details).

## Results Summary
The goal was to compare Bayes-by-Backprop text classifiers with either Normal or Laplace weight priors to plain deep learning text classifiers with or without Dropout.
The tables below contain the test set accuracies on the binary version of the Stanford Sentiment Treebank dataset (SST-2), the IMDb dataset, and the fine-grained version of the Yelp 2015 dataset (Yelp-f).
**Bayes-by-Backprop text classifiers achieve performance comparable to non-Bayesian Dropout variants**, while doubling the number of parameters.

TCN classifier accuracies:

Dataset | Plain   | Dropout | BBB+Normal  | BBB+Laplace
------- | ------- | ------- | ------- | -------
SST-2   | **.83** | .82     | .81     | .81
IMDb    | .89     | **.89** | .88     | .88
Yelp-f  | .62     | .62     | .62     | .62

LSTM classifier accuracies:

Dataset | Plain   | Dropout | BBB+Normal  | BBB+Laplace 
------- | ------- | ------- | ------- | -------
SST-2   | .81     |.81      | .82     | .82
IMDb    | .83     | .83     | .81     | .81    
Yelp-f  | **.63** | .62     | .63     | .63

If you are interested in state-of-the-art performance on the used datasets, check out [NLP-Progress](http://nlpprogress.com/english/sentiment_analysis.html).

### Selective Classification
Another set of experiments, involving selective classification (or **classification with reject option**), yielded the same outcome as did the experiments in [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500) - baseline softmax activation value vastly outperforms Bayesian deep model uncertainty as a proxy for neural network prediction confidence.

## Credits
* MXNet Bayes-by-backprop tutorial [link](https://gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop.html)
* source code for *Bayesian Recurrent Neural Networks* [link](https://github.com/deepmind/sonnet/blob/master/sonnet/examples/brnn_ptb.py)
* source code for *Weight Uncertainty in Neural Networks* [link](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits)
* source code for PyTorch layers [link](https://github.com/pytorch/pytorch)
* source code for selective deep learning [link](https://github.com/geifmany/selective_deep_learning)

## Project status
No longer actively developed!

* Note: My goal was to code the Bayesian layers from scratch. For up-to-date Bayesian deep learning layer implementations, check out the awesome [TensorFlow Probability](https://www.tensorflow.org/probability).

## License
[MIT](https://choosealicense.com/licenses/mit/) © Antonio Šajatović

# Transformer

A minimal proof of concept implementation of the transformer in PyTorch.
The implementation includes a decoder only transformer network.

# Files

### constants.py:

Contains modifiable parameters such as batch size, context size, iterations and learning rate.

### data_loader.py:

Loads the training data tensors into RAM and then eventually onto the GPU memory.
Also allows loading and saving checkpoints during training.


### data_preprocessing.py:

Takes raw text data and converts it into a tokenized tensor for training. The text is tokenized in chunks of 1MB each to avoid running out of memory on large datasets and faster pre-processing.

### model.py:

PyTorch implementation of the Transformer. Implemented as per the original white paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### train.py:

Allows us to train a new model from scratch. If a model already exists then allows us to resume training and saves the model.
Also performs inference at the end on a sample example from the validation set.
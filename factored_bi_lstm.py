#!/usr/bin/env python

import os
import optparse
from collections import OrderedDict

import numpy as np

from keras.preprocessing.sequence import pad_sequences

from keras.layers.embeddings import Embedding
#from keras.layers import Merge
#from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Merge, Dense, Dropout, Embedding, LSTM, Bidirectional
#from keras.layers.core import Dense, Activation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from keras.optimizers import SGD

from keras.models import load_model


# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-d", "--data", default="",
    help="Data set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-o", "--optimizer", default="sgd",
    help="Optimizer"
)
optparser.add_option(
    "-r", "--reload", default="",
    help="Reload the last saved model"
)
optparser.add_option(
    "-m", "--model", default="model",
    help="Name of a model to be saved"
)
opts = optparser.parse_args()[0]

# Parse parameters
parameters = OrderedDict()
parameters['data'] = opts.data
parameters['test'] = opts.test
parameters['dropout'] = opts.dropout
parameters['optimizer'] = opts.optimizer
parameters['reload'] = opts.reload
parameters['model'] = opts.model

print "Detail configuration:"
print "===================================================="
print "Training set: %s" % parameters['data']
print "Test set: %s" % parameters['test']
print "=>Format: word lemma word_cluster pos chunk ne"
print ""
print "Dropout: %f" % parameters['dropout']
print "Optimizer: %s" % parameters['optimizer']
print "Reload: %s" % parameters['reload']
print "Model: %s" % parameters['model']
print "===================================================="

# Check parameters validity
assert os.path.isfile(opts.data)
assert os.path.isfile(opts.test)
assert 0. <= parameters['dropout'] < 1.0

def score_func(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

def split_data(X, y, length=0):
    return X[0: length - 1], y[0: length - 1], X[length:], y[length:]

def read_data(filename):
    raw = open(filename, 'r').readlines()
    all_x = []
    point = []
    for line in raw:
        stripped_line = line.strip().split(' ')
        point.append(stripped_line)
        if line == '\n':
            all_x.append(point[:-1])
            point = []
    all_x = all_x[:-1]
    lengths = [len(x) for x in all_x]
    print 'Input sequence length range: ', max(lengths), min(lengths)
    short_x = [x for x in all_x if len(x) < 64]
    return short_x

def encode_label(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result

short_x_train = read_data(parameters['data'])
short_x_test = read_data(parameters['test'])
train_size = len(short_x_train)
test_size = len(short_x_test)
short_x = short_x_train + short_x_test

X = [[c[0] for c in x] for x in short_x]
y = [[c[5] for c in y] for y in short_x]

def encode(X, y):
    all_text = [c for x in X for c in x]
    
    words = list(set(all_text))
    word2ind = {word: index for index, word in enumerate(words)}
    ind2word = {index: word for index, word in enumerate(words)}
    labels = list(set([c for x in y for c in x]))
    label2ind = {label: (index + 1) for index, label in enumerate(labels)}
    ind2label = {(index + 1): label for index, label in enumerate(labels)}
    print 'Vocabulary size:', len(word2ind), len(label2ind)
    
    maxlen = max([len(x) for x in X])
    print 'Maximum sequence length:', maxlen
    
    
    X_enc = [[word2ind[c] for c in x] for x in X]
    max_label = max(label2ind.values()) + 1
    y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
    y_enc = [[encode_label(c, max_label) for c in ey] for ey in y_enc]
    
    X_enc = pad_sequences(X_enc, maxlen=maxlen)
    y_enc = pad_sequences(y_enc, maxlen=maxlen)
    max_features = len(word2ind)
    out_size = len(label2ind) + 1
    
    X_enc_train = X_enc[0:train_size - 1]
    y_enc_train = y_enc[0:train_size - 1]
    X_enc_test = X_enc[train_size:]
    y_enc_test = y_enc[train_size:]

    return X_enc_train, y_enc_train, X_enc_test, y_enc_test, max_features, out_size, maxlen

# Feature word
X_enc_train, y_enc_train, X_enc_test, y_enc_test, max_features, out_size, maxlen = encode(X, y)

X_train, y_train, X_dev, y_dev = split_data(X_enc_train, y_enc_train, train_size - 300)
X_test, y_test = X_enc_test, y_enc_test

MAX_FEATURES = max_features
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 100
OUT_SIZE = out_size
MAX_LEN = maxlen
BATCH_SIZE = 32
NUM_EPOCHS = 50

print ""
print "Model configuration:"
print "===================================================="
print "MAX_FEATURES: %d" % MAX_FEATURES
print "EMBEDDING_SIZE: %d" % EMBEDDING_SIZE
print "HIDDEN_SIZE: %d" % HIDDEN_SIZE
print "OUT_SIZE: %d" % OUT_SIZE
print "BATCH_SIZE: %d" % BATCH_SIZE
print "EPOCH: %d" % NUM_EPOCHS
print "===================================================="

# Define model
model = Sequential()
model.add(Embedding(MAX_FEATURES, EMBEDDING_SIZE, input_length=MAX_LEN))
model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
model.add(Dense(OUT_SIZE, activation='softmax'))

# Compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

if parameters['reload'] != "":
    print "Reloading model: %s" % parameters['reload']
    model = load_model(parameters['reload'])
# Fit the model with data
model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, validation_data=(X_dev, y_dev), verbose=1)

# The loss value and metrics values for the model in test mode
score = model.evaluate(X_dev, y_dev, batch_size=BATCH_SIZE)
print('Raw test score:', score)

print "======================================"
print "DEV set:"
pr = model.predict_classes(X_dev)
yh = y_dev.argmax(2)
fyh, fpr = score_func(yh, pr)
print 'Accuracy: %f' % accuracy_score(fyh, fpr)
print 'Pricision Recall F1:'
print precision_recall_fscore_support(fyh, fpr, average='weighted')
print "======================================"

print "======================================"
print "TEST set:"
pr = model.predict_classes(X_test)
yh = y_test.argmax(2)
fyh, fpr = score_func(yh, pr)
print 'Accuracy: %f' % accuracy_score(fyh, fpr)
print 'Pricision Recall F1:'
print precision_recall_fscore_support(fyh, fpr, average='weighted')
print "======================================"

if parameters['model'] != "":
    print "Saving model: %s" % parameters['model']
    model.save(parameters['model'])

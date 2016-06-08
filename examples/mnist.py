# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import numpy as np
import sys

from ipwxlearn import glue
from ipwxlearn.datasets import mnist
from ipwxlearn.glue import G
from ipwxlearn.utils import training, predicting, tempdir

BATCH_SIZE = 64
TARGET_NUM = 10


def load_data():
    cache_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], '../data'))
    train_X, train_y, test_X, test_y = mnist.read_data_sets(cache_dir=cache_dir, floatX=glue.config.floatX)

    train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
    test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

    # split train-test set.
    indices = np.arange(train_X.shape[0])
    np.random.shuffle(indices)
    valid_size = int(train_X.shape[0] * 0.1)
    train_idx, valid_idx = indices[:-valid_size], indices[-valid_size:]
    return (train_X[train_idx], train_y[train_idx]), (train_X[valid_idx], train_y[valid_idx]), \
           (test_X, test_y)


(train_X, train_y), (valid_X, valid_y), (test_X, test_y) = load_data()


# build the simple convolutional network.
graph = G.Graph()
with graph.as_default():
    if glue.config.backend == 'theano':
        # Lasagne conv layer does not support to change input shape at :method:`get_output`.
        # Thus we have to keep the training shape and testing shape identical.
        train_input_shape = (None, 28, 28, 1)
    else:
        train_input_shape = (BATCH_SIZE, 28, 28, 1)
    test_input_shape = (None,) + train_input_shape[1:]

    train_input = G.make_placeholder('trainX', shape=train_input_shape, dtype=glue.config.floatX)
    train_label = G.make_placeholder('trainY', shape=train_input_shape[:1], dtype=np.int32)
    test_input = G.make_placeholder('testX', shape=test_input_shape, dtype=glue.config.floatX)
    test_label = G.make_placeholder('testY', shape=test_input_shape[:1], dtype=np.int32)

    # compose the network
    input_layer = G.layers.InputLayer(train_input, shape=train_input_shape)
    network = G.layers.Conv2DInputLayer(input_layer)

    network = G.layers.Conv2DLayer('conv1', network, num_filters=32, filter_size=(5, 5))
    network = G.layers.MaxPool2DLayer('pool1', network, pool_size=(2, 2))

    network = G.layers.Conv2DLayer('conv2', network, num_filters=32, filter_size=(5, 5))
    network = G.layers.MaxPool2DLayer('pool2', network, pool_size=(2, 2))

    network = G.layers.DenseLayer(
        'hidden1',
        G.layers.DropoutLayer('dropout1', network, p=.5),
        num_units=256
    )
    network = G.layers.SoftmaxLayer(
        'softmax',
        G.layers.DropoutLayer('dropout2', network, p=.5),
        num_units=TARGET_NUM
    )

    # derivate the predictions and loss
    train_output, train_loss = G.layers.get_output_with_sparse_softmax_crossentropy(network, train_label)
    train_loss = G.op.mean(train_loss)

    test_output, test_loss = G.layers.get_output_with_sparse_softmax_crossentropy(
        network,
        test_label,
        inputs={input_layer: test_input},   # We use this to override the training input.
        deterministic=True,                 # Disable dropout on testing.
    )
    test_loss = G.op.mean(test_loss)
    test_predict = G.op.argmax(test_output, axis=1)

    # gather summaries
    var_summary = G.summary.merge_summary(G.summary.collect_variable_summaries())
    train_loss_summary = G.summary.scalar_summary('training_loss', train_loss)
    valid_loss_summary = G.summary.scalar_summary('validation_loss', test_loss)

    # Create update expressions for training.
    params = G.layers.get_all_params(network, trainable=True)
    updates = G.updates.adam(train_loss, params)

    train_fn = G.make_function(
        inputs=[train_input, train_label],
        outputs=[train_loss, train_loss_summary],
        updates=updates
    )
    valid_fn = G.make_function(inputs=[test_input, test_label], outputs=[test_loss, valid_loss_summary])
    test_fn = G.make_function(inputs=[test_input], outputs=test_predict)

# train the Network.
with G.Session(graph) as session:
    with tempdir.TemporaryDirectory() as logdir:
        print('Summary log directory: %s' % logdir)
        writer = G.summary.SummaryWriter(logdir, delete_exist=True)
        monitors = [
            training.ValidationMonitor(valid_fn, (valid_X, valid_y), params=params, log_file=sys.stdout,
                                       steps=100, summary_writer=writer),
            training.SummaryMonitor(writer, var_summary, steps=50)
        ]
        max_steps = 10 * len(train_X) // BATCH_SIZE
        training.run_steps(G, train_fn, (train_X, train_y), monitor=monitors, batch_size=BATCH_SIZE,
                           max_steps=max_steps, summary_writer=writer)

        # After training, we compute and print the test error.
        test_predicts = predicting.collect_batch_predict(test_fn, test_X)
        print('Test error: %.2f %%' % (float(np.mean(test_predicts != test_y)) * 100.0))

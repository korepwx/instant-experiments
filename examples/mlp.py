# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import sys

import numpy as np

from ipwxlearn import glue, datasets
from ipwxlearn.glue import G
from ipwxlearn.utils import training

BATCH_SIZE = 64
TARGET_NUM = 10

(train_X, train_y), (test_X, test_y) = datasets.mnist.load_mnist(flatten_to_vectors=True, dtype=glue.config.floatX)
(train_X, train_y), (valid_X, valid_y) = datasets.utils.split_train_valid((train_X, train_y), valid_portion=0.1)

# build the simple neural network.
graph = G.Graph()
with graph.as_default():
    # build the input layers and placeholders
    input_layer, input_var = G.layers.make_input('X', train_X, dtype=glue.config.floatX)
    label_var = G.make_placeholder_for('y', train_y, dtype=np.int32)

    # compose the network
    network = G.layers.DropoutLayer('dropout0', input_layer, p=.2)
    network = G.layers.DenseLayer('hidden1', network, num_units=128)
    network = G.layers.DropoutLayer('dropout1', network, p=.5)
    network = G.layers.DenseLayer('hidden2', network, num_units=32)
    network = G.layers.DropoutLayer('dropout2', network, p=.5)

    network = G.layers.SoftmaxLayer('softmax', network, num_units=TARGET_NUM)

    # derivate the predictions and loss
    train_output, train_loss = G.layers.get_output_with_sparse_softmax_crossentropy(network, label_var)
    train_loss = G.op.mean(train_loss)

    test_output, test_loss = G.layers.get_output_with_sparse_softmax_crossentropy(
        network,
        label_var,
        deterministic=True,                 # Disable dropout on testing.
    )
    test_loss = G.op.mean(test_loss)

    # gather summaries
    var_summary = G.summary.merge_summary(G.summary.collect_variable_summaries())
    train_loss_summary = G.summary.scalar_summary('training_loss', train_loss)
    valid_loss_summary = G.summary.scalar_summary('validation_loss', test_loss)

    # Create update expressions for training.
    params = G.layers.get_all_params(network, trainable=True)
    updates = G.updates.adam(train_loss, params)

    train_fn = G.make_function(
        inputs=[input_var, label_var],
        outputs=[train_loss, train_loss_summary],
        updates=updates
    )
    valid_fn = G.make_function(inputs=[input_var, label_var], outputs=test_loss)

# train the Network.
with G.Session(graph) as session:
    logdir = os.path.join(os.path.split(__file__)[0], 'logs/mlp')
    print('Summary log directory: %s' % logdir)
    writer = G.summary.SummaryWriter(logdir, delete_exist=True)
    monitors = [
        training.ValidationMonitor(valid_fn, (valid_X, valid_y), params=params, log_file=sys.stdout,
                                   steps=100, validation_batch=256, summary_writer=writer),
        training.SummaryMonitor(writer, var_summary, steps=50)
    ]
    max_steps = 10 * len(train_X) // BATCH_SIZE
    training.run_steps(G, train_fn, (train_X, train_y), monitor=monitors, batch_size=BATCH_SIZE,
                       max_steps=max_steps, summary_writer=writer)

    # After training, we compute and print the test error.
    from ipwxlearn import models
    clf = models.Classifier(input_var, network)
    test_predicts = clf.predict(test_X)
    print('Test error: %.2f %%' % (float(np.mean(test_predicts != test_y)) * 100.0))

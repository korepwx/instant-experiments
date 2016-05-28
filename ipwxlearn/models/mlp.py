# -*- coding: utf-8 -*-
import os
import time

import numpy as np

from ipwxlearn import glue
from ipwxlearn.datasets import mnist
from ipwxlearn.glue import G
from ipwxlearn.utils import dataflow


def load_data():
    cache_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], '../../data'))
    train_X, train_y, test_X, test_y = mnist.read_data_sets(cache_dir=cache_dir, floatX=glue.config.floatX)

    # split train-test set.
    indices = np.arange(train_X.shape[0])
    np.random.shuffle(indices)
    valid_size = int(train_X.shape[0] * 0.1)
    train_idx, valid_idx = indices[:-valid_size], indices[-valid_size:]
    return (train_X[train_idx], train_y[train_idx]), (train_X[valid_idx], train_y[valid_idx]), \
           (test_X, test_y)

if __name__ == '__main__':
    (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = load_data()

    # now execute the algorithm.
    graph = G.Graph()
    with G.Session(graph) as sess:
        target_num = 10
        train_input_shape = (32, 784)
        test_input_shape = (None, ) + train_input_shape[1:]

        train_input = G.make_placeholder('trainX', shape=train_input_shape, dtype=glue.config.floatX)
        train_label = G.make_placeholder('trainY', shape=train_input_shape[:1], dtype=np.int32)
        test_input = G.make_placeholder('testX', shape=test_input_shape, dtype=glue.config.floatX)
        test_label = G.make_placeholder('testY', shape=test_input_shape[:1], dtype=np.int32)

        # compose the network
        input = G.layers.InputLayer(train_input, shape=train_input_shape)
        dropout0 = G.layers.DropoutLayer('dropout0', input, p=0.2)
        hidden1 = G.layers.DenseLayer('hidden1', dropout0, num_units=128)
        dropout1 = G.layers.DropoutLayer('dropout1', hidden1, p=0.5)
        hidden2 = G.layers.DenseLayer('hidden2', dropout1, num_units=32)
        dropout2 = G.layers.DropoutLayer('dropout2', hidden1, p=0.5)
        softmax = G.layers.SoftmaxLayer('softmax', dropout2, num_units=target_num)

        # derivate the predictions and loss
        train_output = G.layers.get_output(softmax)
        train_loss = G.op.mean(G.objectives.sparse_categorical_crossentropy(train_output, train_label))

        test_output = G.layers.get_output(softmax, inputs={input: test_input}, deterministic=True)
        test_loss = G.op.sum(G.objectives.sparse_categorical_crossentropy(test_output, test_label))
        test_predict = G.op.argmax(test_output, axis=1)
        test_acc = G.op.sum(G.op.neq(test_predict, test_label))

        # Create update expressions for training.
        params = G.layers.get_all_params(softmax, trainable=True)
        updates = G.updates.adam(train_loss, params)

        train_fn = G.make_function(inputs=[train_input, train_label], outputs=train_loss, updates=updates)
        valid_fn = G.make_function(inputs=[test_input, test_label], outputs=[test_loss, test_acc])
        test_fn = G.make_function(inputs=[test_input], outputs=test_predict)

        # finally, launch the training loop.
        print('Start training ...')
        for epoch in range(10):
            # do a full pass over the training data
            train_batches = 0
            train_loss = 0
            start_time = time.time()
            for train_batch_X, train_batch_y in \
                    dataflow.iterate_training_batches([train_X, train_y],
                                                      batch_size=train_input_shape[0],
                                                      shuffle=True):
                train_loss += train_fn(train_batch_X, train_batch_y)
                train_batches += 1
            train_loss = train_loss / train_batches

            # add a full pass over the validation data
            valid_loss = 0
            valid_error = 0
            for valid_batch_X, valid_batch_y in \
                    dataflow.iterate_testing_batches([valid_X, valid_y], batch_size=256):
                loss, error = valid_fn(valid_batch_X, valid_batch_y)
                valid_loss += loss
                valid_error += error
            valid_loss /= len(test_X)
            valid_error /= len(test_y)

            # print the performance metric for this epoch.
            print('Epoch %s took %.3fs' % (epoch+1, time.time() - start_time))
            print('  training loss:\t\t%.6f' % train_loss)
            print('  validation loss:\t\t%.6f' % valid_loss)
            print('  validation error:\t\t%.2f %%' % (valid_error * 100.0))

        # After training, we compute and print the test error.
        test_predicts = []
        for test_batch_X in dataflow.iterate_testing_batches(test_X, batch_size=256):
            test_predicts.append(test_fn(test_batch_X))
        test_predicts = np.concatenate(test_predicts, axis=0).astype(np.int32)
        print('Final results:')
        print('  test error:\t\t%.2f %%' % (float(np.mean(test_predicts != test_y)) * 100.0))

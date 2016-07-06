# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import shutil

import numpy as np

from ipwxlearn import glue, datasets, models
from ipwxlearn.glue import G
from ipwxlearn.training.trainers import LossTrainer

BATCH_SIZE = 64
TARGET_NUM = 10

(train_X, train_y), (test_X, test_y) = datasets.mnist.load_mnist(dtype=glue.config.floatX)

# build the simple convolutional network.
graph = G.Graph()
with graph.as_default():
    # build the input layers and placeholders
    input_layer, input_var = G.layers.make_input('X', train_X, dtype=glue.config.floatX)
    label_var = G.make_placeholder_for('y', train_y, dtype=np.int32)

    # compose the network
    network = G.layers.Conv2DInputLayer(input_layer)

    network = G.layers.Conv2DLayer('conv1', network, num_filters=32, filter_size=(5, 5))
    network = G.layers.MaxPool2DLayer('pool1', network, pool_size=(2, 2))
    network = G.layers.Conv2DLayer('conv2', network, num_filters=32, filter_size=(5, 5))
    network = G.layers.MaxPool2DLayer('pool2', network, pool_size=(2, 2))
    network = G.layers.DropoutLayer('dropout0', network, p=.5)

    network = models.MLP('mlp', network, layer_units=[256], dropout=.5)

    lr = models.LogisticRegression('logistic', network, target_num=TARGET_NUM)

    # create the trainer.
    trainer = LossTrainer(
        validation_split=0.1,
        validation_steps=100,
        max_epoch=10
    )
    trainer.set_summary('logs/convnet', summary_steps=100)
    trainer.set_model(lr, input_var, label_var)

# train the Network.
with G.Session(graph) as session:
    # train the Network.
    if os.path.isdir(trainer.summary_dir):
        shutil.rmtree(trainer.summary_dir)
    print('Summary log directory: %s' % trainer.summary_dir)
    trainer.fit(train_X, train_y)

    # After training, we compute and print the test error.
    clf = models.wrappers.Classifier(lr, input_var)
    test_predicts = clf.predict(test_X)
    print('Test error: %.2f %%' % (float(np.mean(test_predicts != test_y)) * 100.0))

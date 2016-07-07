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

(train_X, train_y), (test_X, test_y) = datasets.mnist.load_mnist(flatten_to_vectors=True, dtype=glue.config.floatX)

# build the simple neural network.
graph = G.Graph()
with graph.as_default():
    # build the input layers and placeholders
    input_layer, input_var = G.layers.make_input('X', train_X, dtype=glue.config.floatX)
    label_var = G.make_placeholder_for('y', train_y, dtype=np.int32)

    # compose the network
    mlp = models.MLP('encoder', input_layer, layer_units=[128, 32], nonlinearity=G.nonlinearities.rectify)
    dae = models.DenoisingAutoEncoder(
        encoder=mlp,
        decoder=mlp.get_decoder(name='decoder'),
        noise=models.noise.DropoutNoise(p=.3),
        metric=models.metrics.SquareError()
    )

    # create the trainer.
    trainer = LossTrainer(
        validation_split=0.1,
        validation_steps=100,
        max_epoch=10
    )
    trainer.set_summary('logs/dae', summary_steps=100)
    trainer.set_model(dae, input_var)

with G.Session(graph) as session:
    # train the Network.
    if os.path.isdir(trainer.summary_dir):
        shutil.rmtree(trainer.summary_dir)
    print('Summary log directory: %s' % trainer.summary_dir)
    trainer.fit(train_X)

    # After training, we compute and print the test error.
    clf = models.wrappers.Transformer(G.layers.get_output(dae, use_decoder=True), input_var)
    test_predicts = clf.predict(test_X)
    print('MSE: %.6f' % np.mean((test_predicts - test_X) ** 2))

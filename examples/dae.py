# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import shutil

import numpy as np

from ipwxlearn import glue, datasets, models, visualize
from ipwxlearn.glue import G
from ipwxlearn.training.trainers import LossTrainer

BATCH_SIZE = 64
TARGET_NUM = 10

(train_X, train_y), (test_X, test_y) = datasets.mnist.load_mnist(dtype=glue.config.floatX)

# build the denoising auto encoder.
graph = G.Graph()
with graph.as_default():
    # build the input layers and placeholders
    input_layer, input_var = G.layers.make_input('X', train_X, dtype=glue.config.floatX)
    label_var = G.make_placeholder_for('y', train_y, dtype=np.int32)

    # compose the network
    mlp = models.MLP('encoder', input_layer, layer_units=[128, 32], nonlinearity=G.nonlinearities.rectify)
    dae = models.DenoisingAutoEncoder(
        encoder=mlp,
        decoder=mlp.build_decoder(name='decoder', tie_weights=True),
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

with G.Session(graph):
    # train the Network.
    if os.path.isdir(trainer.summary_dir):
        shutil.rmtree(trainer.summary_dir)
    print('Summary log directory: %s' % trainer.summary_dir)
    trainer.fit(train_X)

    # After training, we compute and print the test error.
    clf = models.wrappers.Transformer(G.layers.get_output(dae), input_var)
    test_predicts = clf.predict(test_X)
    print('MSE: %.6f' % np.mean((test_predicts - test_X) ** 2))

    # Save the original images and the reconstructed images to file.
    cols = int(np.ceil(np.sqrt(len(test_X))))
    visualize.save_image(visualize.grid_arrange_images(test_X, cols),
                         os.path.join(trainer.summary_dir, 'test.png'))
    visualize.save_image(visualize.grid_arrange_images(test_predicts, cols),
                         os.path.join(trainer.summary_dir, 'test-re.png'))

# add a logistic regression upon the denoising auto encoder and do the classification.
with graph.as_default():
    # Because the mlp in the auto-encoder does not have dropout, which makes it
    # vulnerable to overfitting, we thus build another mlp with the same weights
    # and biases, but with additional dropout layers on the input.
    mlp2 = models.MLP('mlp', G.layers.DropoutLayer('dropout0', input_layer, p=.2),
                      layer_units=mlp.layer_units, nonlinearity=mlp.nonlinearity,
                      W=mlp.layer_weights, b=mlp.layer_biases)
    lr = models.LogisticRegression('logistic', mlp2, target_num=TARGET_NUM)
    trainer.set_summary('logs/dae-lr', summary_steps=100)

    # set the loss function that should be trained.
    # if you want to fit the parameters in the MLP, you may try this:
    #
    #   with G.layers.with_param_tags(mlp2, trainable=False):
    #       trainer.set_model(lr, input_var, label_var)
    trainer.set_model(lr, input_var, label_var)

with G.Session(graph):
    # train the logistic regression as well as fine-tuning the auto-encoder.
    # you may also try to fix parameters of the auto-encoder, and only train the remaining network.
    if os.path.isdir(trainer.summary_dir):
        shutil.rmtree(trainer.summary_dir)
    print('Summary log directory: %s' % trainer.summary_dir)
    trainer.fit(train_X, train_y)

    # After training, we compute and print the test error.
    clf = models.wrappers.Classifier(lr, input_var)
    test_predicts = clf.predict(test_X)
    print('Test error: %.2f %%' % (float(np.mean(test_predicts != test_y)) * 100.0))

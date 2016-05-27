# -*- coding: utf-8 -*-
import numpy as np

from ipwxlearn import glue
from ipwxlearn.glue import G


if __name__ == '__main__':
    graph = G.Graph()
    with graph.as_default():
        input_shape = (None, 784)
        target_num = 10
        input_var = G.make_placeholder('X', shape=input_shape, dtype=glue.config.floatX)
        label_var = G.make_placeholder('labels', shape=(None,), dtype=np.int32)

        # compose the network
        input = G.layers.InputLayer(input_var, shape=input_shape)
        hidden1 = G.layers.DenseLayer('hidden1', input, num_units=128)
        hidden2 = G.layers.DenseLayer('hidden2', hidden1, num_units=32)
        softmax = G.layers.SoftmaxLayer('softmax', hidden2, num_units=target_num)

        # attach another network
        hidden3 = G.layers.DenseLayer('hidden3', input, num_units=800)
        softmax2 = G.layers.SoftmaxLayer('softmax2', hidden3, num_units=target_num)

        # derivate the predictions and loss
        train_output = G.layers.get_output(softmax)
        loss = G.op.mean(G.objectives.sparse_categorical_crossentropy(train_output, label_var))

        # Create update expressions for training.
        params = G.layers.get_all_params(softmax, trainable=True)
        updates = G.updates.adam(loss, params)

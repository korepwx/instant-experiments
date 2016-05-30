# -*- coding: utf-8 -*-
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from ipwxlearn import glue
from ipwxlearn.glue import G


class SoftmaxUnitTest(unittest.TestCase):

    @staticmethod
    def make_softmax_data(n=10000, dim=10, target_num=2, dtype=np.float64):
        if target_num > 2 or glue.config.backend != 'tensorflow':
            W = (np.random.random([dim, target_num]) - 0.5).astype(dtype)
            b = (np.random.random([target_num]) - 0.5).astype(dtype)
            X = ((np.random.random([n, dim]) - 0.5) * 10.0).astype(dtype)
            y = np.argmax(np.dot(X, W) + b, axis=1)
            return (W, b), (X, y)
        else:
            W = (np.random.random([dim, 1]) - 0.5).astype(dtype)
            b = (np.random.random([1]) - 0.5).astype(dtype)
            X = ((np.random.random([n, dim]) - 0.5) * 10.0).astype(dtype)
            logits = (np.dot(X, W) + b).reshape([X.shape[0]])
            y = (1.0 / (1 + np.exp(-logits)) <= 0.5).astype(np.int32)
            return (W, b), (X, y)

    def test_binary_predicting(self):
        """Test binary softmax classifier."""
        target_num = 2
        (W, b), (X, y) = self.make_softmax_data(target_num=target_num, dtype=glue.config.floatX)

        # When target_num == 2, LogisticRegression from scikit-learn uses sigmoid, but lasagne doesn't.
        # So we just test the consistency between scikit-learn LR with tensorflow backend.
        if glue.config.backend == 'tensorflow':
            lr = LogisticRegression().fit(X, y)
            lr.coef_ = -W.T
            lr.intercept_ = -b
            self.assertTrue(np.alltrue(lr.predict(X) == y))

        graph = G.Graph()
        with graph.as_default():
            input_var = G.make_placeholder('inputs', shape=(None, W.shape[0]), dtype=glue.config.floatX)
            input_layer = G.layers.InputLayer(input_var, shape=(None, W.shape[0]))
            softmax_layer = G.layers.SoftmaxLayer('softmax', input_layer, num_units=target_num, W=W, b=b)
            predict_prob = G.layers.get_output(softmax_layer)
            predict_label = G.op.argmax(predict_prob, axis=1)
            predict_fn = G.make_function(inputs=[input_var], outputs=[predict_prob, predict_label])

        with G.Session(graph):
            prob, predict = predict_fn(X)
            self.assertTrue(np.alltrue(predict == y))
            if glue.config.backend == 'tensorflow':
                self.assertTrue(np.allclose(lr.predict_proba(X), prob))

    def test_categorical_predicting(self):
        """Test categorical softmax classifier."""
        target_num = 5
        (W, b), (X, y) = self.make_softmax_data(target_num=target_num, dtype=glue.config.floatX)

        lr = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(X, y)
        lr.coef_ = W.T
        lr.intercept_ = b
        self.assertTrue(np.alltrue(lr.predict(X) == y))

        graph = G.Graph()
        with graph.as_default():
            input_var = G.make_placeholder('inputs', shape=(None, W.shape[0]), dtype=glue.config.floatX)
            input_layer = G.layers.InputLayer(input_var, shape=(None, W.shape[0]))
            softmax_layer = G.layers.SoftmaxLayer('softmax', input_layer, num_units=target_num, W=W, b=b)
            predict_prob = G.layers.get_output(softmax_layer)
            predict_label = G.op.argmax(predict_prob, axis=1)
            predict_fn = G.make_function(inputs=[input_var], outputs=[predict_prob, predict_label])

        with G.Session(graph):
            prob, predict = predict_fn(X)
            self.assertTrue(np.alltrue(predict == y))
            self.assertTrue(np.allclose(lr.predict_proba(X), prob))

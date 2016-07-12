# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin

from ipwxlearn.glue import G
from ipwxlearn.utils import predicting

__all__ = [
    'Classifier',
    'Regressor',
    'Transformer',
]


class BaseEstimator(object):
    """
    Model wrapper for estimator that produces some output on given input.

    This model wrapper does not provide the method to train the model.  Using trainers from
    :module:`ipwxlearn.trainers` to train the model.

    :param output: The output layer, or an output expression.
    :param input_var: The input placeholder.
    :param trainer: Trainer for this estimator.  If not specified, :method:`fit` will not work.
    :param predict_batch_size: If specified, will predict the output in batches.
    """

    def __init__(self, output, input_var, trainer=None, predict_batch_size=None):
        if isinstance(output, G.layers.Layer):
            self.output = G.layers.get_output(output, deterministic=True)
            self.output_layer = output
            self.graph = output.graph
        else:
            self.output = output
            self.output_layer = None
            self.graph = G.current_graph()
        self.input_var = input_var
        self.trainer = trainer
        self.predict_batch_size = predict_batch_size
        self.predict_fn = G.make_function(inputs=[input_var], outputs=self.output)

    def save(self, path):
        """
        Save the parameters of the model to external file.

        If an output expression is specified instead of a model, all the parameters in the graph
        will be saved to file.

        :param path: Path of the persistent file.
        """
        if self.output_layer is None:
            G.utils.save_graph_state(self.graph, path, persistent=True)
        else:
            params = G.layers.get_all_params(self.output_layer, persistent=True)
            G.utils.save_graph_state_by_vars(self.graph, path, params)

    def load(self, path):
        """
        Load the parameters of the model from external file.

        :param path: Path of the persistent file.
        """
        G.utils.restore_graph_state(self.graph, path)

    def _make_session(self):
        try:
            session = next(G.iter_sessions())
            if session.graph != self.graph:
                session = G.Session(self.graph)
        except StopIteration:
            session = G.Session(self.graph)
        return session

    def _do_predict(self, X):
        with self._make_session():
            if self.predict_batch_size is not None:
                ret = predicting.collect_batch_predict(self.predict_fn, X, batch_size=self.predict_batch_size,
                                                       mode='concat')
            else:
                ret = self.predict_fn(X)
        return ret

    def fit(self, X, y=None):
        """
        Train the estimator with given data.

        :param X: Input data.
        :param y: Target data, if the model is a supervised model.

        :return: self
        """
        with self._make_session():
            if self.trainer is None:
                raise ValueError('Trainer is not set.')
            self.trainer.fit(X, y)
        return self


class Classifier(BaseEstimator, ClassifierMixin):
    """
    Model wrapper for classifier.

    A classifier should fit on (X, y), where X should be an N-d tensor, and y be a 1-d integer vector with
    the same length in X's first dimension.
    """

    def predict(self, X):
        """
        Predict class labels for samples in X.

        :param X: An N-d tensors, as data points.
        :return: Returns a 1-D tensor, where each row represents the predicted class.
        """
        scores = self.predict_proba(X)
        if len(scores.shape) == 1:
            indices = (scores >= 0.5).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return indices

    def predict_proba(self, X):
        """
        Probability estimates.

        :param X: An N-d tensors, as data points.
        :return: Returns a 1-D tensor or 2-D tensor.
                 If 1-D tensor, it means the probability of being positive class.
                 If 2-D tensor, each row represents the probability of being each corresponding class.
        """
        return self._do_predict(X)

    def predict_log_proba(self, X):
        """
        Log of probability estimates

        :param X: An N-d tensors, as data points.
        :return: Returns a 1-D tensor or 2-D tensor.
                 If 1-D tensor, it means the log probability of being positive class.
                 If 2-D tensor, each row represents the log probability of being each corresponding class.
        """
        return np.log(self.predict_proba(X))


class Regressor(BaseEstimator, RegressorMixin):
    """
    Model wrapper for regressor.

    A regressor should fit on (X, y), where X should be an N-d tensor, and y be an N-d tensor with
    the same length in X's first dimension.
    """

    @staticmethod
    def squeeze_output(y):
        """Squeeze the second dimension if y is of shape (?, 1)."""
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.reshape([y.shape[0]])
        return y

    def predict(self, X):
        """
        Output estimates.

        :param X: Input data.
        :return: Estimated output.
        """
        return self.squeeze_output(self._do_predict(X))

    def score(self, X, y, sample_weight=None):
        return super(Regressor, self).score(X, self.squeeze_output(y), sample_weight=sample_weight)


class Transformer(BaseEstimator, TransformerMixin):
    """
    Model wrapper for transformer.

    A transformer should fit on X, where X should be an N-d tensor.
    """

    def predict(self, X):
        """
        Output estimates.

        :param X: Input data.
        :return: Estimated output.
        """
        return self._do_predict(X)

    def transform(self, X):
        return self.predict(X)

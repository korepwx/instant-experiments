# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import six
from sklearn.base import BaseEstimator as _BaseEstimator, ClassifierMixin as _ClassifierMixin

from ipwxlearn import glue
from ipwxlearn.glue import G
from ipwxlearn.glue.common.utils import get_graph_state, set_graph_state
from ipwxlearn.utils import predicting

__all__ = [
    'BaseEstimator',
    'BaseClassifier'
]


class BaseEstimator(_BaseEstimator):
    """
    Base class for all estimators.

    This class will drop all non-public fields (i.e., fields starting with underscore "_") on pickling.
    Subclasses must store all pickle-able parameters in public fields, while keeping transient objects
    in protected or private fields.
    """

    @property
    def graph(self):
        """
        Get the tensor computation graph for this model.
        :rtype: :class:`~ipwxlearn.glue.common.graph.BaseGraph`
        """
        return self._graph

    def __getstate__(self):
        return {
            'params': {k: v for k, v in six.iteritems(self.__dict__) if not k.startswith('_')},
            'state': get_graph_state(self.graph) if self.graph is not None else None
        }

    def __setstate__(self, state):
        for k, v in six.iteritems(state['params']):
            setattr(self, k, v)
        if state['state']:
            self._build_graph()
            set_graph_state(self.graph, state['state'])

    def _build_graph(self):
        """Derived classes should override this to build graph according to instance parameters."""
        raise NotImplementedError()


class BaseClassifier(BaseEstimator, _ClassifierMixin):
    """
    Base class for all classifiers.

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
        with G.Session(self.graph):
            return predicting.collect_batch_predict(
                self._predict_fn, X.astype(glue.config.floatX), batch_size=256, mode='concat')

    def predict_log_proba(self, X):
        """
        Log of probability estimates

        :param X: An N-d tensors, as data points.
        :return: Returns a 1-D tensor or 2-D tensor.
                 If 1-D tensor, it means the log probability of being positive class.
                 If 2-D tensor, each row represents the log probability of being each corresponding class.
        """
        return np.log(self.predict_proba(X))

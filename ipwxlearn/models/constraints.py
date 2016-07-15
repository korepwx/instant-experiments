# -*- coding: utf-8 -*-
import numpy as np

from ipwxlearn.glue import G
from ipwxlearn.utils.misc import maybe_iterable_to_list


class SupervisedModel(object):
    """Constraints for supervised models."""

    def _validate_target(self, target):
        if target is None:
            raise TypeError('"target" is not specified, but %s is a supervised model.' % self.__class__.__name__)


class UnsupervisedModel(object):
    """Constraints for unsupervised models."""

    def _validate_target(self, target):
        if target is not None:
            raise TypeError('"target" is not specified, but %s is an unsupervised model.' % self.__class__.__name__)


class ModelWithLoss(object):
    """
    Constraints for models equipped with a particular loss.
    """

    def get_loss_for(self, input, target=None, **kwargs):
        """
        Get the per-data loss for given input and target (if required).
        Returns a loss tensor with shape (input.shape[0],)
        """
        raise NotImplementedError()


class ModelSupportDecoding(object):
    """
    Constraints for models which supports decoding.
    """

    @staticmethod
    def transpose_initializers(initializers):
        """
        Transpose specified initializers.

        Returns transposed version of a numpy array or a backend variable, or just the original one
        if neither of these.

        :param initializers: An initializer, or a list of initializers.
        :return: The transposed initializers, returned in reversed order.
        """
        def transpose(x):
            if isinstance(x, np.ndarray):
                return x.T
            elif G.utils.is_variable(x):
                return G.op.transpose(x)
            else:
                return x

        initializers = maybe_iterable_to_list(initializers)
        if not isinstance(initializers, list):
            return transpose(initializers)
        return [transpose(i) for i in reversed(initializers)]

    def build_decoder_for(self, name, incoming, **kwargs):
        """
        Build the decoder of this model.

        :param name: Name for the decoder model.
        :param incoming: Input layer(s) for the decoder model.
        :param **kwargs: Additional arguments for creating the decoder model.
        :return: The decoder model.
        """
        raise NotImplementedError()

    def build_decoder(self, name, **kwargs):
        """
        Get the decoder of this model, using this model itself as the input to the decoder.

        :param name: Name for the decoder model.
        :param **kwargs: Additional arguments for creating the decoder model.
        :return: The decoder model.
        """
        return self.build_decoder_for(name=name, incoming=self, **kwargs)

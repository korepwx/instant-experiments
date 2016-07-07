# -*- coding: utf-8 -*-


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

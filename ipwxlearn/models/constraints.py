# -*- coding: utf-8 -*-


class SupervisedModel(object):
    """Constraints for supervised models."""


class UnsupervisedModel(object):
    """Constraints for unsupervised models."""


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
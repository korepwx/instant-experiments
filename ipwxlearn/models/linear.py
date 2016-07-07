# -*- coding: utf-8 -*-
from ipwxlearn import glue
from ipwxlearn.glue import G
from .base import BaseModel
from .constraints import ModelWithLoss, SupervisedModel

__all__ = [
    'LogisticRegression'
]


class LogisticRegression(BaseModel, ModelWithLoss, SupervisedModel):
    """
    Logistic regression model.

    A logistic regression maps the input to a k-dimensional multinomial distribution.

    :param name: Name of this MLP model.
    :param incoming: Input layer, or the shape of input.
    :param target_num: Number of targets of the multinomial distribution.
    :param W: Weight initializer for the dense layers.
    :param b: Bias initializer for the dense layers.
    """

    def __init__(self, name, incoming, target_num, W=G.init.XavierNormal(), b=G.init.Constant(0.0)):
        super(LogisticRegression, self).__init__(name=name, incoming=incoming)

        # if the target_num is only 2, we use sigmoid instead of softmax.
        if target_num < 2:
            raise ValueError('Logistic Regression can only be applied to target_num > 2.')
        elif target_num == 2:
            num_units = 1
        else:
            num_units = target_num

        # use a linear dense layer to produce the logits.
        self.logits = G.layers.DenseLayer(name, incoming, num_units=num_units, W=W, b=b, nonlinearity=None)
        self.target_num = target_num

    def get_output_shape_for(self, input_shape):
        return input_shape[0] + (self.target_num,)

    def get_output_for(self, input, **kwargs):
        """
        Returns a tensor of shape (input.shape[0], target_num), indicating the classification probability
        of each target.
        """
        logits = self.logits.get_output_for(input, **kwargs)
        if self.target_num == 2:
            output = G.op.sigmoid(logits)
            output = G.op.concat([1.0 - output, output], axis=1)
        else:
            output = G.op.softmax(logits)
        return output

    def get_loss_for(self, input, target=None, **kwargs):
        self._validate_target(target)
        logits = self.logits.get_output_for(input, **kwargs)
        if self.target_num == 2:
            logits = G.op.flatten(logits, ndim=1)
            loss = G.objectives.sigmoid_cross_entropy_with_logits(logits, G.op.cast(target, glue.config.floatX))
        else:
            loss = G.objectives.sparse_softmax_cross_entropy_with_logits(logits, target)
        return loss

    def get_params(self, **tags):
        return self.logits.get_params(**tags)

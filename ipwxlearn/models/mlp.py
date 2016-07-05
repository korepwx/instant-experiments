# -*- coding: utf-8 -*-
from ipwxlearn.glue import G
from ipwxlearn.models.chain import ChainModel

__all__ = [
    'MLP'
]


class MLP(ChainModel):
    """
    Multi-layer perceptron model.

    A multi-layer perceptron consists of several fully-connected layers, with an optimal
    non-linear activation function.

    :param name: Name of this MLP model.
    :param incoming: Input layer, or the shape of input.
    :param layer_units: A tuple/list of integers, representing the number of units of each layer.
    :param nonlinearity: Non-linear activation function for each layer.
    :param dropout: A float number as the dropout probability for the output of each layer.
    :param W: Weight initializer for the dense layers.
    :param b: Bias initializer for the dense layers.
    """

    def __init__(self, name, incoming, layer_units, nonlinearity=G.nonlinearities.rectify, dropout=None,
                 W=G.init.XavierNormal(), b=G.init.Constant(0.0)):
        if not layer_units:
            raise ValueError('At least one layer should be specified.')
        super(MLP, self).__init__(name, incoming)

        with G.name_scope(self.name_scope):
            network = self.input_layer or self.input_shape
            for i, n_out in enumerate(layer_units, 1):
                network = G.layers.DenseLayer('dense%d' % i, network, num_units=n_out, nonlinearity=nonlinearity,
                                              W=W, b=b)
                self.add_layer(network)
                if dropout:
                    network = G.layers.DropoutLayer('dropout%d' % i, network, p=dropout)
                    self.add_layer(network)

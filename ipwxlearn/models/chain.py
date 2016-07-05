# -*- coding: utf-8 -*-
from ipwxlearn.models import BaseModel

__all__ = [
    'ChainModel'
]


class ChainModel(BaseModel):
    """
    Chaining neural network model.

    A chaining network consists of several layers chained together, where the input flows through
    the whole chain to produce an output.
    """

    def __init__(self, name, incoming):
        super(ChainModel, self).__init__(name, incoming)

        #: Store all the layers in this feed forward network.
        self.layers = []

    @property
    def output_layer(self):
        """Get the output layer (last layer) of the feed forward network."""
        return self.layers[-1]

    def add_layer(self, layer):
        """Add a layer to this feed forward network."""
        self.layers.append(layer)

    def get_output_shape_for(self, input_shape):
        output_shape = input_shape
        for layer in self.layers:
            output_shape = layer.get_output_shape_for(output_shape)
        return output_shape

    def get_output_for(self, input, **kwargs):
        output = input
        for layer in self.layers:
            output = layer.get_output_for(output, **kwargs)
        return output

    def get_params(self, **tags):
        result = []
        for layer in self.layers:
            result.extend(layer.get_params(**tags))
        result.extend(super(ChainModel, self).get_params(**tags))
        return result
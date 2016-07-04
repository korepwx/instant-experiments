# -*- coding: utf-8 -*-
from ipwxlearn.glue import G

__all__ = [
    'BaseModel',
    'FeedForwardModel'
]


class BaseModel(G.layers.Layer):
    """
    Abstract interface for all neural network models.

    A typical neural network model consumes the input and produces output, thus it could
    also be seen as a huge Layer from a higher perspective.

    :param name: Name of this model, which would be used as the name scope for all layers
                 in this model.
    :param incoming: Input layer, or the shape of input.
    """

    def __init__(self, name, incoming):
        super(BaseModel, self).__init__(incoming, name=name)

        try:
            parent_ns = G.current_name_scope()
        except ValueError:
            self.name_scope = G.NameScope(name)
        else:
            self.name_scope = parent_ns.sub_scope(name)

    def save(self, path):
        """
        Save the parameters of this model to external file.

        :param path: Path of the persistent file.
        """
        G.utils.save_graph_state_by_vars(self.graph, path, self.get_params(persistent=True))

    def load(self, path):
        """
        Load the parameters of this model from external file.

        :param path: Path of the persistent file.
        """
        G.utils.restore_graph_state(self.graph, path)


class FeedForwardModel(BaseModel):
    """
    Feed forward neural network model.

    A feed forward network consists of several layers chained together, where the input flows through
    the whole chain to produce an output.
    """

    def __init__(self, name, incoming):
        super(FeedForwardModel, self).__init__(name, incoming)

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
        """Computes the output shape of this layer, given an input shape."""
        output_shape = input_shape
        for layer in self.layers:
            output_shape = layer.get_output_shape_for(output_shape)
        return output_shape

    def get_output_for(self, input, **kwargs):
        """
        Propagates the given input through this layer (and only this layer).

        :param input: TensorFlow expression that should be propagate through this layer.
        :return: TensorFlow expression as the output of this layer.
        """
        output = input
        for layer in self.layers:
            output = layer.get_output_for(output, **kwargs)
        return output

    def get_params(self, **tags):
        """
        Return a list of variables that parameterize the layer.

        :param **tags: Tags that filter the parameters.
        :return: List of variables that parameterize the layer.
        """
        result = []
        for layer in self.layers:
            result.extend(layer.get_params(**tags))
        result.extend(super(FeedForwardModel, self).get_params(**tags))
        return result

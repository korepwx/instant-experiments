# -*- coding: utf-8 -*-
from ipwxlearn.glue import G
from ipwxlearn.utils import misc

__all__ = [
    'BaseModel',
    'BaseChainModel',
]


class BaseModel(G.layers.Layer):
    """
    Abstract interface for all neural network models.

    A typical neural network model consumes the input and produces output, thus it could
    also be seen as a huge Layer from a higher perspective.

    :param name: Name of this model, which would be used as the name scope for all layers
                 in this model.  Some models may accept empty name.
    :param incoming: Input layer, or the shape of input.
    """

    def __init__(self, name, incoming):
        super(BaseModel, self).__init__(incoming, name=name or None)

        try:
            parent_ns = G.current_name_scope()
        except ValueError:
            self.name_scope = G.NameScope(name or None)     # name == None is feasible here.
        else:
            self.name_scope = parent_ns.sub_scope(name) if name else parent_ns

    def save(self, path, include_inputs=True):
        """
        Save the parameters of this model to external file.

        :param path: Path of the persistent file.
        :param include_inputs: Whether or not to include parameters from all ancestor layers?
                               (Default True)
        """
        if include_inputs:
            params = G.layers.get_all_params(self, persistent=True)
        else:
            params = self.get_params(persistent=True)
        G.utils.save_graph_state_by_vars(self.graph, path, params)

    def load(self, path):
        """
        Load the parameters of this model from external file.

        :param path: Path of the persistent file.
        """
        G.utils.restore_graph_state(self.graph, path)


class BaseChainModel(BaseModel):
    """
    Base class for all chaining neural network models.

    A chaining network consists of several layers chained together, where the input flows through
    the whole chain to produce an output.
    """

    def __init__(self, name, incoming):
        super(BaseChainModel, self).__init__(name, incoming)

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
        # :method:`G.layers.get_all_params` can recursively discover all the parameters
        # in whole network, with the help of :attr:`input_layer` attribute.  But this
        # chaining model acts as a whole from an outside viewer's point.  Thus we will
        # have to correctly discover all the parameters from hidden inputs.
        common_ignore_layers = []
        if isinstance(self, G.layers.MergeLayer):
            common_ignore_layers.extend([l for l in self.input_layers if l is not None])
        elif self.input_layer is not None:
            common_ignore_layers.append(self.input_layer)

        result = super(BaseChainModel, self).get_params(**tags)
        for i, layer in enumerate(self.layers):
            # when we are looking for
            treat_as_input = common_ignore_layers + self.layers[:i] + self.layers[i+1:]
            result.extend(G.layers.get_all_params(layer, treat_as_input=treat_as_input, **tags))

        return misc.unique(result)

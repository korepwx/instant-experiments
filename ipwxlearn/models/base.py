# -*- coding: utf-8 -*-
import contextlib

from ipwxlearn.glue import G

__all__ = [
    'BaseModel'
]


class BaseModel(G.layers.Layer):
    """
    Abstract interface for all neural network models.

    A typical neural network model consumes the input and produces output, thus it could
    also be seen as a huge Layer from a higher perspective.

    :param incoming: Input layer, or the shape of input.
    :param name: Name of this model, which would be used as the name scope for all layers
                 in this model.  Some models may accept empty name.
    """

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

    @contextlib.contextmanager
    def with_scope(self):
        """Open the name scope of this model."""
        with G.name_scope(self.name_scope) as scope:
            yield scope

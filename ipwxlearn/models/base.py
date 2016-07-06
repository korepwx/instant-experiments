# -*- coding: utf-8 -*-
from ipwxlearn.glue import G

__all__ = [
    'BaseModel'
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

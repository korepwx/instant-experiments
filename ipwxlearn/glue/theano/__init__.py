from ..common import graph as _common_graph
from ..common import scope as _common_scope

VariableTags = _common_graph.VariableTags
GlueVariable = _common_graph.VariableInfo
current_graph = _common_graph.current_graph
current_name_scope = _common_scope.current_name_scope
name_scope = _common_scope.name_scope

from .. import config as _config
config = _config
floatX = config.floatX

from . import graph, layers, utils
from .graph import Graph
from .utils import make_variable

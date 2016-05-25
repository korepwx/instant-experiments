from ..common import graph as _common_graph
from ..common import scope as _common_scope
from ..common import session as _common_session

VariableTags = _common_graph.VariableTags
VariableInfo = _common_graph.VariableInfo
current_graph = _common_graph.current_graph
current_name_scope = _common_scope.current_name_scope
name_scope = _common_scope.name_scope
current_session = _common_session.current_session

from .. import config as _config
config = _config
floatX = config.floatX

from . import graph, layers, session, utils
from .graph import Graph
from .session import Session
from .utils import make_variable

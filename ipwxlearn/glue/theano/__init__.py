from ..common import scope as _common_scope

VariableTags = _common_scope.VariableTags
GlueVariable = _common_scope.GlueVariable
current_graph = _common_scope.current_graph
current_name_scope = _common_scope.current_name_scope
name_scope = _common_scope.name_scope

from . import layers
from . import scope
from .scope import *

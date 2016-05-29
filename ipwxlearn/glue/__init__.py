# -*- coding: utf-8 -*-

from . import config
from .common.graph import current_graph
from .common.scope import current_name_scope
from .common.session import current_session


if config.backend == 'theano':
    from . import theano
    G = theano
elif config.backend == 'tensorflow':
    from . import tensorflow
    G = tensorflow
else:
    raise ValueError('Unknown backend %s.' % config.backend)

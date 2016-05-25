# -*- coding: utf-8 -*-

import os

from . import tensorflow, theano

_BACKENDS_ = {
    'tensorflow': tensorflow,
    'theano': theano
}

_backend_ = os.getenv('IPWX_LEARN_BACKEND', None)
if _backend_ is not None and _backend_ not in _BACKENDS_:
    raise ValueError('Unknown backend %s.' % repr(_backend_))
G = _BACKENDS_[_backend_] if _backend_ else theano

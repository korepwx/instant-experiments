# -*- coding: utf-8 -*-

from . import config
from . import tensorflow, theano

_BACKENDS_ = {
    'tensorflow': tensorflow,
    'theano': theano
}

G = _BACKENDS_[config.backend]

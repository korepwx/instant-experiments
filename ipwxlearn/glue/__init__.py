# -*- coding: utf-8 -*-

import os

from . import tensorflow, theano

_BACKENDS_ = {
    'tensorflow': tensorflow,
    'theano': theano
}

G = _BACKENDS_.get(os.getenv('IPWX_LEARN_BACKEND', None), theano)

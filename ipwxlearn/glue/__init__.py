# -*- coding: utf-8 -*-

from . import config


if config.backend == 'theano':
    from . import theano
    G = theano
elif config.backend == 'tensorflow':
    from . import tensorflow
    G = tensorflow
else:
    raise ValueError('Unknown backend %s.' % config.backend)

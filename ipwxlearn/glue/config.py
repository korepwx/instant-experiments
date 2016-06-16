# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import re
import warnings


# Read the config of default backend type.
def _read_backend_type():
    backend = os.environ.get('TENSOR_BACKEND', 'theano').lower()
    if backend == 'tensorflow':
        # TensorFlow would not perform correctly if GPU is selected by Theano.
        # So we unset the THEANO_FLAGS if TensorFlow is going to be used.
        theano_flags = ''.join([
            kv for kv in os.environ.get('THEANO_FLAGS', '').split(',')
            if not re.match(r'^device=gpu.*$', re.I)
        ])
        os.environ['THEANO_FLAGS'] = theano_flags
    return backend

backend = _read_backend_type()


# If we've got config of the float number type from environment variable,
# we may likely to change the config values in backend according to these.
def _read_floatX():
    import theano
    floatX = os.environ.get('TENSOR_FLOATX', None)
    if floatX is None:
        floatX = theano.config.floatX
    floatX = floatX.lower()
    if (floatX in ('32', 'float32')):
        floatX = 'float32'
    elif (floatX in ('64', 'float64')):
        if backend == 'tensorflow':
            warnings.warn('TensorFlow currently lack a good support for float64, downgrade to float32.')
            floatX = 'float32'
        else:
            floatX = 'float64'
    else:
        raise ValueError('Unknown float number type %s.' % repr(floatX))
    theano.config.floatX = floatX
    return floatX

floatX = _read_floatX()

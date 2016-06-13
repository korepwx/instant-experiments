# -*- coding: utf-8 -*-
from __future__ import absolute_import

from ipwxlearn.glue import G


ACTIVATIONS = {
    'relu': G.nonlinearities.rectify,
    'sigmoid': G.nonlinearities.sigmoid,
    'tanh': G.nonlinearities.tanh
}

#!/usr/bin/env python
import os
import sys

import six
import numpy as np

from ipwxlearn import glue
from ipwxlearn.datasets.utils import split_train_valid
from ipwxlearn.estimators import MLPRegressor

if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl

X = np.linspace(0, np.pi * 2, 1001, dtype=glue.config.floatX).reshape((-1, 1))
y = np.sin(X)
(train_X, train_y), (test_X, test_y) = split_train_valid((X, y), valid_portion=0.2)

model_path = sys.argv[1] if len(sys.argv) > 1 else None
if model_path and os.path.isfile(model_path):
    with open(model_path, 'rb') as f:
        reg = pkl.load(f)
else:
    reg = MLPRegressor(layers=[256, 64], max_epoch=400, activation='relu')
    reg.fit(train_X, train_y)
    if model_path:
        with open(model_path, 'wb') as f:
            pkl.dump(reg, f, protocol=pkl.HIGHEST_PROTOCOL)

predict = reg.predict(test_X)
print('MSE: %s' % np.mean((predict - test_y) ** 2))

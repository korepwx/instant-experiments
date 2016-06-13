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

X = (2 * np.pi * np.random.random((50000, 4))).astype(glue.config.floatX)
y = np.sum(np.sin(X), axis=1)
(train_X, train_y), (test_X, test_y) = split_train_valid((X, y), valid_portion=0.2)

model_path = sys.argv[1] if len(sys.argv) > 1 else None
if model_path and os.path.isfile(model_path):
    with open(model_path, 'rb') as f:
        reg = pkl.load(f)
else:
    reg = MLPRegressor(layers=[512, 256, 128, 64, 32], max_epoch=100, activation='relu')
    reg.fit(train_X, train_y, validation_steps=100)
    if model_path:
        with open(model_path, 'wb') as f:
            pkl.dump(reg, f, protocol=pkl.HIGHEST_PROTOCOL)

predict = reg.predict(test_X)
print('MSE: %s' % np.mean((predict - test_y) ** 2))
relative_error = np.mean(np.abs((predict - test_y) / (test_y + (test_y == 0.0) * 1e-7)))
print('Relative Error: %.2f%%' % (100.0 * relative_error))

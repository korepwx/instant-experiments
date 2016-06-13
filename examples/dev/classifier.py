#!/usr/bin/env python
import os
import sys

import six
import numpy as np
from ipwxlearn.datasets import mnist

from ipwxlearn.estimators import MLPClassifier

if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl

(train_X, train_y), (test_X, test_y) = mnist.load_mnist()

model_path = sys.argv[1] if len(sys.argv) > 1 else None
if model_path and os.path.isfile(model_path):
    with open(model_path, 'rb') as f:
        clf = pkl.load(f)
else:
    clf = MLPClassifier(layers=[128, 32], max_epoch=10)
    clf.fit(train_X, train_y)
    if model_path:
        with open(model_path, 'wb') as f:
            pkl.dump(clf, f, protocol=pkl.HIGHEST_PROTOCOL)

predict = clf.predict(test_X)
print('Error Rate: %s%%' % (100.0 * np.mean(predict != test_y)))

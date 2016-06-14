#!/usr/bin/env python

import os
import sys

import six
import numpy as np
from ipwxlearn import glue
from ipwxlearn.estimators import MLPRegressor


###############################################################################
# Generate data
X = np.linspace(0, np.pi * 2, 2000, dtype=glue.config.floatX)
y = (np.sin(X) * 10 + np.random.normal(size=X.shape)).astype(glue.config.floatX)
indices = np.arange(len(X))
np.random.shuffle(indices)
offset = int(X.shape[0] * 0.8)
X_train, y_train = X[indices[:offset]], y[indices[:offset]]
X_test, y_test = X[indices[offset:]], y[indices[offset:]]


###############################################################################
# Fit regression model
clf = MLPRegressor(layers=[800, 100], max_epoch=100, dropout=0.5, activation='relu', verbose=True)
clf.fit(X_train.reshape((-1, 1)), y_train)
y_pred = clf.predict(X_test.reshape((-1, 1)))
mse = np.mean((y_test - y_pred) ** 2)
print("MSE: %.4f" % mse)

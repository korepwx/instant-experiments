# -*- coding: utf-8 -*-
from __future__ import absolute_import

import unittest

import numpy as np

from ipwxlearn import glue
from ipwxlearn.glue import G


class OpTester(object):

    def __init__(self, owner, tensor_op, numpy_op=None,
                 seal_args=False, dtype=glue.config.floatX):
        self.owner = owner
        self.tensor_op = tensor_op
        self.numpy_op = numpy_op
        self.seal_args = seal_args
        self.dtype = dtype
        self.test_cases = []

    def convert_to_tensor(self, x):
        if G.__backend__ == 'tensorflow':
            import tensorflow as tf
            return tf.convert_to_tensor(x, dtype=self.dtype)
        elif G.__backend__ == 'theano':
            from theano import tensor as T
            return T.as_tensor_variable(np.asarray(x, dtype=self.dtype))

    def add(self, *args, **kwargs):
        self.test_cases.append((args, kwargs))

    def run(self):
        graph = G.Graph()
        with graph.as_default():
            r1, r2 = [], []
            for args, kwargs in self.test_cases:
                kwargs = kwargs.copy()
                if self.numpy_op is None:
                    r2.append(np.asarray(kwargs.pop('result'), dtype=np.int32))
                else:
                    if self.seal_args:
                        r2.append(self.numpy_op(args, **kwargs))
                    else:
                        r2.append(self.numpy_op(*args, **kwargs))
                tensor_args = [self.convert_to_tensor(a) for a in args]
                if self.seal_args:
                    r1.append(self.tensor_op(tensor_args, **kwargs))
                else:
                    r1.append(self.tensor_op(*tensor_args, **kwargs))
            compute = G.make_function(outputs=r1)
            with G.Session(graph):
                r1 = compute()
            for a, b, (args, kwargs) in zip(r1, r2, self.test_cases):
                self.owner.assertTrue(
                    a.shape == b.shape,
                    msg='%r != %r: args=%r, kwargs=%r' % (a, b, args, kwargs)
                )
                self.owner.assertTrue(
                    np.allclose(a, b),
                    msg='%r != %r: args=%r, kwargs=%r' % (a, b, args, kwargs)
                )


class OpTestCase(unittest.TestCase):

    def test_dot(self):
        """Test dot product."""
        x = [3, 4, 5]
        y = [[6, 7, 8], [9, 10, 11], [12, 13, 14]]
        t = OpTester(self, G.op.dot, np.dot, dtype=np.int32)
        t.add(2, 3)
        t.add(2, x)
        t.add(x, 2)
        t.add(x, x)
        t.add(2, y)
        t.add(y, 2)
        t.add(x, y)
        t.add(y, x)
        t.add(y, y)
        t.run()

    def test_concat(self):
        """Test tensor concatenate."""
        t = OpTester(self, G.op.concat, np.concatenate, seal_args=True, dtype=np.int32)
        t.add([1, 2], [3, 4], axis=0)
        t.add([1], [2, 3, 4], axis=0)
        t.add([[1, 2]], [[3, 4]], axis=0)
        t.add([[1, 2]], [[3, 4]], axis=1)
        t.run()

    def test_squeeze(self):
        """Test squeeze."""
        x = [[[1], [2], [3]]]
        t = OpTester(self, G.op.squeeze, dtype=np.int32)
        t.add(x, result=[1, 2, 3])
        t.add(x, dims=[0], result=[[1], [2], [3]])
        t.add(x, dims=[2], result=[[1, 2, 3]])
        t.add(x, dims=[-1], result=[[1, 2, 3]])
        t.add(x, dims=[0, 2], result=[1, 2, 3])
        t.add(x, dims=[0, -1], result=[1, 2, 3])
        t.run()

    def test_flatten(self):
        """Test flatten."""
        x = np.arange(16, dtype=np.int32).reshape([2, 4, 2])
        t = OpTester(self, G.op.flatten, dtype=np.int32)
        t.add(x, result=x.reshape([16]))
        t.add(x, ndim=2, result=x.reshape([2, 8]))
        t.add(x, ndim=3, result=x.reshape([2, 4, 2]))
        t.run()

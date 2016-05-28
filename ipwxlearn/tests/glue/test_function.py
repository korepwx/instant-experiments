# -*- coding: utf-8 -*-
import unittest

import numpy as np

from ipwxlearn.glue import G
from ipwxlearn.utils.misc import assert_raises_message


class FunctionTestCase(unittest.TestCase):

    def test_make_function(self):
        """Test make function."""
        graph = G.Graph()
        with graph.as_default():
            a = G.make_placeholder('a', shape=(), dtype=np.int32)
            b = G.make_placeholder('b', shape=(), dtype=np.int32)
            c = G.make_placeholder('c', shape=(), dtype=np.int32)
            fn = G.make_function(inputs=[a, b], outputs=(a + b + c), givens={c: np.array(1000, dtype=np.int32)})
            self.assertEqual(fn(1, 2), 1003)

    def test_args_check(self):
        """Test argument check when calling function."""

        graph = G.Graph()
        with graph.as_default():
            a = G.make_placeholder('a', shape=(), dtype=np.int32)
            b = G.make_placeholder('b', shape=(), dtype=np.int32)
            c = G.make_variable('c', shape=(), init=2, dtype=np.int32)

        with G.Session(graph):
            # test calling function with no argument.
            fn0 = G.make_function(outputs=c)
            self.assertEquals(fn0(), 2)
            with assert_raises_message(self, ValueError, 'Function only accepts unnamed arguments.'):
                fn0(a=2)
            with assert_raises_message(self, ValueError, 'Require 0 unnamed arguments, but got 1.'):
                fn0(1)

            # test calling function with one unnamed argument.
            fn1 = G.make_function(inputs=a, outputs=2 * a)
            self.assertEquals(fn1(2), 4)
            with assert_raises_message(self, ValueError, 'Function only accepts unnamed arguments.'):
                fn1(a=2)
            with assert_raises_message(self, ValueError, 'Require 1 unnamed arguments, but got 0.'):
                fn1()
            with assert_raises_message(self, ValueError, 'Require 1 unnamed arguments, but got 2.'):
                fn1(2, 3)

            # test calling function with two unnamed arguments.
            fn2 = G.make_function(inputs=[a, b], outputs=a + b)
            self.assertEquals(fn2(2, 3), 5)
            with assert_raises_message(self, ValueError, 'Function only accepts unnamed arguments.'):
                fn2(a=2, b=3)
            with assert_raises_message(self, ValueError, 'Function only accepts unnamed arguments.'):
                fn2(2, b=3)
            with assert_raises_message(self, ValueError, 'Require 2 unnamed arguments, but got 1.'):
                fn2(2)
            with assert_raises_message(self, ValueError, 'Require 2 unnamed arguments, but got 3.'):
                fn2(2, 3, 4)

            # test calling function with two named arguments.
            fn3 = G.make_function(inputs={'a': a, 'b': b}, outputs=a + b)
            self.assertEquals(fn3(a=3, b=4), 7)
            with assert_raises_message(self, ValueError, 'Function only accepts named arguments.'):
                fn3(3, 4)
            with assert_raises_message(self, ValueError, 'Function only accepts named arguments.'):
                fn3(3, b=4)
            with assert_raises_message(self, ValueError, 'Named argument b is required but not specified.'):
                fn3(a=3)
            with assert_raises_message(self, ValueError, 'Unexpected named argument c.'):
                fn3(a=3, b=4, c=5)

    def test_updates(self):
        """Test variable updates by function."""

        graph = G.Graph()
        with graph.as_default():
            a = G.make_placeholder('a', shape=(), dtype=np.int32)
            b = G.make_placeholder('b', shape=(), dtype=np.int32)
            c = G.make_variable('c', shape=(), init=2, dtype=np.int32, persistent=True)

        with G.Session(graph):
            updates = G.op.assign(c, a + b)
            fn = G.make_function(inputs=[a, b], outputs=c + a + b, updates=updates)
            self.assertEquals(fn(2, 3), 7, msg='Expressions in the function should use the old values of variables, '
                                               'not the updated ones.')
            self.assertEquals(G.get_variable_values(c), 5)

        with G.Session(graph):
            updates = [G.op.assign(c, a + b), G.op.assign(c, a * b)]
            fn = G.make_function(inputs=[a, b], updates=updates)
            fn(3, 7)
            self.assertEquals(G.get_variable_values(c), 21, msg='Should select the latest assignment to '
                                                                'a single variable.')

# -*- coding: utf-8 -*-
import unittest

import numpy as np

from ipwxlearn.glue import G


class ShapeUnitTest(unittest.TestCase):

    def test_slice(self):
        """Test the slice layer."""
        shape = (51, 53, 57)
        X = np.random.randint(low=0, high=2147483647, size=shape, dtype=np.int32)

        def build(indices, axis=-1):
            graph = G.Graph()
            with graph.as_default():
                input_shape = (None,) + shape[1:]
                input_var = G.make_placeholder('X', shape=input_shape, dtype=np.int32)
                input_layer = G.layers.InputLayer(input_var, shape=input_shape)
                slice_layer = G.layers.SliceLayer(input_layer, axis=axis, indices=indices)
                output = G.layers.get_output(slice_layer)
                get_output = G.make_function(inputs=[input_var], outputs=output)
            return graph, get_output

        def test(indices, axis=-1):
            graph, get_output = build(indices, axis)
            if axis < 0:
                axis += len(shape)

            if axis == 0:
                truth = X[indices]
            elif axis == 1:
                truth = X[:, indices]
            elif axis == 2:
                truth = X[:, :, indices]
            else:
                raise ValueError('"axis" out of range.')

            with G.Session(graph):
                output = get_output(X)

            self.assertEquals(truth.shape, output.shape,
                              msg="test(slice=%r, axis=%r): expected shape %r, got %r" %
                                  (indices, axis, truth.shape, output.shape))
            self.assertTrue(np.alltrue(truth == output),
                            msg="test(slice=%r, axis=%r): output mismatch" %
                                (indices, axis))

        for axis in (-3, -2, -1, 0, 1, 2):
            for indices in (0, 1, 31, -2, -1):
                test(indices=indices, axis=axis)

        for axis in (0, 1, 2):
            for indices in (slice(0, None), slice(1, None), slice(31, None), slice(-2, None), slice(-1, None),
                            slice(None, 0), slice(None, 1), slice(None, 31), slice(None, -2), slice(None, -1),
                            slice(0, -1), slice(-2, -1), slice(0, 31), slice(31, -2), slice(-31, 31)):
                test(indices=indices, axis=axis)

    def test_reshape(self):
        """Test the reshape layer."""
        x = np.arange(16, dtype=np.int32).reshape([2, 4, 2])
        graph = G.Graph()
        with graph.as_default():
            x_input, x_var = G.layers.make_input('x', x, dtype=np.int32)
            i_var = G.make_placeholder('i', shape=(), dtype=np.int32)
            j_var = G.make_placeholder('j', shape=(), dtype=np.int32)

            def test(shape, result=None, givens=None):
                if result is None:
                    result = x.reshape(shape)
                result = np.asarray(result, dtype=np.int32)
                layer = G.layers.ReshapeLayer(x_input, shape=shape)
                f = G.make_function(x_var, outputs=G.layers.get_output(layer), givens=givens)
                with G.Session(graph):
                    output = f(x)
                    self.assertTrue(np.all(output == result),
                                    msg='%r != %r: shape=%r' % (output, result, shape))

            def i32(i):
                return np.asarray(i, dtype=np.int32)

            # test various reshape operations.
            test([-1], result=x.reshape([-1]))
            test([4, 4])
            test([-1, 2, 2])
            test([-1, 4])
            test([8, -1])
            test([[0], [2], [1]], x.reshape([2, 2, 4]))
            test([2, [1], -1], x.reshape([2, 4, 2]))
            test([i_var, j_var], result=x.reshape([8, 2]), givens={i_var: i32(8), j_var: i32(-1)})
            test([i_var, [1], -1], result=x.reshape([2, 4, 2]), givens={i_var: i32(2)})

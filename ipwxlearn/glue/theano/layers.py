# -*- coding: utf-8 -*-
import lasagne
import six

from ipwxlearn.glue.theano import current_name_scope


def _get_layer_name(name):
    return current_name_scope().resolve_name(name)


def _collect_layer_params(layer):
    """Collect layer parameters as variables into the current name scope."""
    parent, name = layer.name.rsplit('.', 1)
    assert(parent == current_name_scope().full_name)

    with current_name_scope().get_scope(name) as layer_scope:
        for param in layer.params:
            parent2, name2 = param.name.rsplit('.', 1)
            assert(parent2 == layer_scope.full_name)
            layer_scope.add_variable(name2, param)

    return layer


def _layer_constructor(method):
    @six.wraps(method)
    def wrapper(name, *args, **kwargs):
        name = _get_layer_name(name)
        layer = method(name=name, *args, **kwargs)
        return _collect_layer_params(layer)
    return wrapper


@_layer_constructor
def inputs(name, shape, input_var):
    """
    Input layer.

    :param name: Name of this layer.
    :param shape: Shape of the input.
    :param input_var: Input variable for this input layer.
    """
    return lasagne.layers.InputLayer(shape=shape, input_var=input_var, name=name)


@_layer_constructor
def dropout(name, incoming, p=0.5, rescale=True):
    """
    Dropout layer.

    :param name: Name of this layer.
    :param incoming: Tuple (shape, var) or layer, as the input to this layer.
    :param p: Dropout probability.
    :param rescale: If true, the input is rescaled by 1 / (1-p) on training.
    """
    return lasagne.layers.DropoutLayer(incoming=incoming, p=p, rescale=rescale, name=name)


@_layer_constructor
def dense(name, incoming, num_units, W=None, b=None, activation=None):
    """
    A fully connected layer.

    :param name: Name of this layer.
    :param incoming: Tuple (shape, var) or layer, as the input to this layer.
    :param num_units: The number of units in this layer.
    :param W: Theano variable, numpy array, or initializer for the weight matrix.
    :param b: Theano variable, numpy array, or initializer for the bias vector.
    :param activation: Activation function that would be applied to this layer.
                       If None is provided, the layer will be linear.
    """
    return lasagne.layers.DenseLayer(incoming=incoming, num_units=num_units, W=W, b=b,
                                     nonlinearity=activation, name=name)

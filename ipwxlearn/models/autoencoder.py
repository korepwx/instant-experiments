# -*- coding: utf-8 -*-
from ipwxlearn.glue import G
from .base import BaseModel
from .constraints import ModelWithLoss, UnsupervisedModel
from .metrics import SquareError

__all__ = [
    'BaseAutoEncoder',
    'DenoisingAutoEncoder',
]


class BaseAutoEncoder(G.layers.CompoundLayer, BaseModel, ModelWithLoss, UnsupervisedModel):
    """
    Base class for auto-encoders.

    An auto-encoder consists of an encoder and a decoder, where the encoder transforms
    the network input to a certain representation (which often has fewer dimensions
    than the input), and the decoder aims to transform it back to input.

    This class is the base class for all kinds of auto-encoders.  It should be initialized
    with the output layers of the encoder and the decoder, plus the input layer to the whole
    encoder network.  Then this class will build an auto-encoder upon these objects, and
    take care of the training loss.

    :param encoder: The output layer of the encoder network.
    :param decoder: The output layer of the decoder network.
    :param metric: Error metric for the input and the reconstructed output.
                   See :module:`~ipwxlearn.models.metrics` for more error metrics.
    :param main_input: If more than one input exists to the encoder + decoder network,
                       you must specify the main input for auto-encoder to fit with.
    """

    def __init__(self, encoder, decoder, metric=SquareError(), main_input=None, name=None):
        super(BaseAutoEncoder, self).__init__(children=[encoder, decoder], name=name)

        if not self.input_layers:
            raise ValueError('No input is given to the auto-encoder network.')
        if len(self.input_layers) > 1 and main_input is None:
            raise ValueError('More than one input is given to the auto-encoder network, '
                             'but the main input is not specified.')
        if main_input not in self.input_layers:
            raise ValueError('The main input is not one of the inputs to the auto-encoder network.')

        self.encoder = encoder
        self.decoder = decoder
        self.metric = metric
        self.main_input = main_input
        self._main_input_index = self.input_layers.index(self.main_input)

    def get_output_for(self, inputs, use_decoder=False, **kwargs):
        """
        Get the output of this model.

        :param input: Input to the model.
        :param use_decoder: Make use of decoder when computing the output. (Default False).
        """
        ctx = G.layers.CompoundLayer.GetOutputContext(self, inputs=inputs, **kwargs)
        if use_decoder:
            return ctx.get_output(self.decoder)
        else:
            return ctx.get_output(self.encoder)

    def get_output_shape_for(self, input_shapes):
        ctx = G.layers.CompoundLayer.GetOutputContext(self, input_shapes=input_shapes)
        return ctx.get_output_shape(self.encoder)


class DenoisingAutoEncoder(BaseAutoEncoder):
    """
    Denoising auto-encoder.

    This class is a certain kind of auto-encoder, which adds noise to the input during
    training, in order to prevent the auto-encoder to learn an identity function.

    :param noise: The noise to be added to input on training.
                  See :module:`~ipwxlearn.models.noise` for more noise generators.
    """

    def __init__(self, encoder, decoder, noise, metric=SquareError(), name=None):
        super(DenoisingAutoEncoder, self).__init__(encoder=encoder, decoder=decoder, metric=metric, name=name)
        self.noise = noise

    def get_loss_for(self, inputs, target=None, **kwargs):
        self._validate_target(target)

        # make corrupted inputs
        idx = self._main_input_index
        corrupted_inputs = inputs.copy()
        corrupted_inputs[idx] = self.noise(inputs[idx])

        # get the reconstructed output
        output = self.get_output_for(corrupted_inputs, use_decoder=True, **kwargs)

        # finally compose the loss
        loss = G.op.mean(self.metric(output, inputs[idx]))
        return loss

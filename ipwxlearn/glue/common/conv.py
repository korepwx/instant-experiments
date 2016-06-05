# -*- coding: utf-8 -*-


class ConvPadType:
    #: No padding / a valid convolution.
    VALID = 'valid'

    #: Padding the input with half the filter size on both sides.
    #: When ``stride=1``, this results in an output size equal to the input size.
    SAME = 'same'

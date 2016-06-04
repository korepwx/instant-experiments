# -*- coding: utf-8 -*-


class ConvPadType:
    #: Padding the input with half the filter size on both sides.
    #: When ``stride=``, this results in an output size equal to the input size.
    SAME = 'same'

    #: No padding / a valid convolution.
    VALID = 'valid'

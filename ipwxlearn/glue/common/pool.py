# -*- coding: utf-8 -*-


class PoolPadType:
    #: No padding.
    NONE = 'none'

    #: Padding the input with half the filter size on both sides.
    #: This option would cause the output shape to be backend specific,
    #: but might have better performance than SAME padding.
    BACKEND = 'backend'

    #: Padding the input with half the filter size on both sides.
    #: The size of this padding type is deterministic across different backends.
    #: When ``stride=1``, this results in an output size equal to the input size.
    SAME = 'same'

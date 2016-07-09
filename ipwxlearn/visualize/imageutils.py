# -*- coding: utf-8 -*-
import numpy as np

__all__ = [
    'grid_arrange_images',
]


def grid_arrange_images(images, cols=None, rows=None):
    """
    Arrange the images in grid.
    The images will be arranged in the row-first order.

    :param images: 4D numpy array of shape (n_images, image_height, image_width, n_channels).
    :param cols: Columns of the grid.
    :param rows: Rows of the grid.  Ignored if the cols is specified.

    :return: 3D numpy array as the grid image.
    """
    if len(images.shape) != 4:
        raise ValueError('"images" must be a 4D tensor.')

    N = images.shape[0]
    n_channels = images.shape[3]
    image_shape = images.shape[1: 3]
    if cols:
        rows = (N + cols - 1) // cols
    else:
        cols = (N + rows - 1) // rows
    H, W = image_shape[0], image_shape[1]

    # the new canvas to hold all images.
    canvas = np.zeros(shape=(H * rows, W * cols, n_channels), dtype=images.dtype)

    # copy every image to the canvas
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            if idx < N:
                img = images[idx]
                for i in range(H):
                    canvas_row = row * H + i
                    canvas_start = col * W
                    canvas_end = (col + 1) * W
                    canvas[canvas_row, canvas_start:canvas_end] = img[i]

    return canvas

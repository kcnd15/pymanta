# utils.py

import numpy as np


def scale(
        Y,
        center=True,
        scale_flag=False):
    """
    scale {base}	R Documentation
    Scaling and Centering of Matrix-like Objects
    Description
    scale is generic function whose default method centers and/or scales the columns of a numeric matrix.

    Usage
    scale(x, center = TRUE, scale = TRUE)
    Arguments
    x	a numeric matrix(like object).
    center	either a logical value or numeric-alike vector of length equal to the number of columns of x,
            where ‘numeric-alike’ means that as.numeric(.) will be applied successfully if is.numeric(.) is not true.
    scale	either a logical value or a numeric-alike vector of length equal to the number of columns of x.
    :param Y:
    :param center:
    :param scale_flag:
    :return:
    """

    x = Y.copy()

    if center:
        x -= x.mean()

    if scale_flag and center:
        x /= x.std()

    elif scale_flag:
        x /= np.sqrt(x.pow(2).sum().div(x.count() - 1))

    return x

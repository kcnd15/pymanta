# utils.py

import numpy as np
import sys


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
        # axis=None: The default is to compute the mean of the flattened array.
        x_mean_columns = x.mean(axis=0)
        x -= x_mean_columns

    if scale_flag and center:
        # axis=None: The default is to compute the standard deviation of the flattened array.
        x_std_columns = x.std(axis=0)
        x /= x_std_columns

    elif scale_flag:
        x /= np.sqrt(x.pow(2).sum(axis=0).div(x.count() - 1))

    return x


def stop(message: str):
    print("Error:", message)
    sys.exit(1)


def match_arg(argument, valid_values: list, argument_name=None):
    """
    R function "match.arg" replacement
    :param argument:
    :param valid_values:
    :param argument_name:
    :return:
    """
    if argument in valid_values:
        return argument
    else:
        if argument_name is None:
            error_msg = f"Error: argument should be one of "
        else:
            error_msg = f"Error: {argument_name} should be one of "
        error_msg += f"{str(valid_values)}, but is '{argument}'"

        print(error_msg)
        sys.exit(2)


def model_matrix(
        object,
        data, # = environment(object),
        contrasts_arg = None, # NULL,
        xlev = None # NULL
    ):
    pass

def crossproduct(matrix):
    """
    calculate the crossproduct of a matrix T =  T'T
    :param matrix: matrix
    :return: crossproduct
    """

    cp = matrix.transpose().dot(matrix)
    return cp

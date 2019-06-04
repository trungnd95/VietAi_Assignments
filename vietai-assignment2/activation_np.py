"""activation_np.py
This file provides activation functions for the NN 
Author: Phuong Hoang
Editor: Trung Ng
Date: 02/06/2019
VietAi Ha Noi - Course 04 - 2019
"""

import numpy as np


def sigmoid(x):
    """sigmoid
    TODO:  - Doing by Trung Ng - 02/06/2019
           - State: Done
    Sigmoid function. Output = 1 / (1 + exp(-x)).
    :param x: input
    """
    if isinstance(x, np.ndarray):
        return 1 / (1 + np.exp(-x))
    else:
        raise TypeError("Type of input param should be numpy!")


def sigmoid_grad(a):
    """sigmoid_grad
    TODO: - Doing by Trung Ng - 02/06/2019
          - State: Done
    Compute gradient of sigmoid with respect to input. g'(x) = g(x)*(1-g(x))
    :param a: output of the sigmoid function
    """
    if isinstance(a, np.ndarray):
        return a * (1-a)
    else:
        raise TypeError("Type of input param should be numpy!")


def reLU(x):
    """reLU
    TODO: - Doing by Trung Ng - 02/06/2019
          - State: Done
    Rectified linear unit function. Output = max(0,x).
    :param x: input
    """
    if isinstance(x, np.ndarray):
        return x * (x > 0)
    else:
        raise TypeError("Type of input param should be numpy!")


def reLU_grad(a):
    """reLU_grad
    TODO: - Doing by Trung Ng - 02/06/2019
          - State: Done
    Compute gradient of ReLU with respect to input
    :param x: output of ReLU
    """
    if isinstance(a, np.ndarray):
        return 1. * (a > 0)
    else:
        raise TypeError("Type of input param should be numpy!")


def tanh(x):
    """tanh
    TODO: - Doing by Trung Ng - 02/06/2019
          - State: Done
    Tanh function.
    :param x: input
    """
    if isinstance(x, np.ndarray):
        return np.tanh(x)
    else:
        raise TypeError("Type of input param should be numpy!")


def tanh_grad(a):
    """tanh_grad
    TODO: - Doing by Trung Ng - 02/06/2019
          - State: Done
    Compute gradient for tanh w.r.t input
    :param a: output of tanh
    """
    if isinstance(a, np.ndarray):
        return 1. - a * a
    else:
        raise TypeError("Type of input param should be numpy!")


def softmax(x):
    """softmax
    TODO: - Doing by Trung Ng - 02/06/2019
          - State: Done
    Softmax function.
    :param x: input
    """
    if isinstance(x, np.ndarray):
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0) if e_x.ndim == 1 else e_x / e_x.sum(axis=1).reshape(x.shape[0], 1)
    else:
        raise TypeError("Input type should be numpy array")


def softmax_minus_max(x):
    """softmax_minus_max
    TODO: - Doing by Trung Ng - 02/06/2019
          - State: Done
    Stable softmax function.
    :param x: input
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)
        else:
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1).T
    else:
        raise TypeError("Input type should be numpy array")
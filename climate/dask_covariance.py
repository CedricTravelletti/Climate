""" Re-implementation of dask covariance so that stays float32.

"""
import warnings
import numpy as np

from dask.utils import apply, derived_from
from dask.array.core import (
    Array,
    asarray,
    concatenate,
)
from dask.array.routines import dot, array


@derived_from(np)
def cov(m, y=None, rowvar=1, bias=0, ddof=None):
    # This was copied almost verbatim from np.cov
    # See numpy license at https://github.com/numpy/numpy/blob/master/LICENSE.txt
    # or NUMPY_LICENSE.txt within this directory
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")

    # Handles complex arrays too
    m = asarray(m)
    if y is None:
        dtype = np.result_type(m, np.float32)
    else:
        y = asarray(y)
        dtype = np.result_type(m, y, np.float32)
    X = array(m, ndmin=2, dtype=dtype)

    if X.shape[0] == 1:
        rowvar = 1
    if rowvar:
        N = X.shape[1]
        axis = 0
    else:
        N = X.shape[0]
        axis = 1

    # check ddof
    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0
    fact = float(N - ddof)
    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning)
        fact = 0.0

    if y is not None:
        y = array(y, ndmin=2, dtype=dtype)
        X = concatenate((X, y), axis)

    X = X - X.mean(axis=1 - axis, keepdims=True)
    if not rowvar:
        return (dot(X.T, X.conj()) / fact).squeeze()
    else:
        return (dot(X, X.T.conj()) / fact).squeeze()

from math import radians, sin, cos, asin, sqrt, pi, atan, atan2, fabs
import numpy as np


def _poisson_covariance(dist, lambda0):
    """ Poisson covariance model on the sphere.

    Parameters
    ----------
    dist: float
        Great circle distance.
    lambda0: float
        Lengthscale parameter.

    """
    cov = (1 - lambda0**2) / (1 - 2*lambda0*np.cos(dist) + lambda0**2)**(3/2)
    return cov

poisson_covariance = np.vectorize(_poisson_covariance, excluded=['lambda0'])

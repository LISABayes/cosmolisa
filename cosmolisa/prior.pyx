from __future__ import division
from libc.math cimport log
cimport cython
from .cosmology cimport CosmologicalParameters

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double logprior_redshift_single_event(CosmologicalParameters omega, double event_redshift, double log_norm):
    """
    Prior function for a single GW event redshift.
    Loops over all possible hosts to accumulate the likelihood
    Parameters:
    ===============

    """
#    # p(z_gw|O H I)
    cdef double logP     = log(omega.UniformComovingVolumeDensity(event_redshift))-log_norm

    return logP

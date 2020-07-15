from __future__ import division
from libc.math cimport log
cimport cython
from .cosmology cimport CosmologicalParameters
import numpy as np

def logprior_redshift_single_event(CosmologicalParameters omega, double event_redshift, double zmax, double dl_cutoff):
    return _logprior_redshift_single_event(omega, event_redshift, zmax, dl_cutoff)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _logprior_redshift_single_event(CosmologicalParameters omega, double event_redshift, double zmax, double dl_cutoff):
    """
    Prior function for a single GW event redshift.
    Parameters:
    ===============

    """
    # p(z_gw|O H I)
    if dl_cutoff > 0:
        if omega.LuminosityDistance(zmax) >= dl_cutoff:
            return -np.inf
    log_norm = log(omega.IntegrateComovingVolumeDensity(zmax))
    return log(omega.UniformComovingVolumeDensity(event_redshift))-log_norm

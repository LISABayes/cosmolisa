cimport cython
from .cosmology cimport CosmologicalParameters

cdef double _logprior_redshift_single_event(CosmologicalParameters omega, double event_redshift, double zmax, double dl_cutoff)
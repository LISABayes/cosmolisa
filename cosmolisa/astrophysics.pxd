import cosmolisa.cosmology
cimport numpy as np
from cosmolisa.cosmology cimport CosmologicalParameters

ctypedef double (*model_pointer)(double, double[5]) nogil

cdef class PopulationModel:
    cdef public str density_model
    cdef double _integrated_rate_(self)
    cdef model_pointer density
    cdef public CosmologicalParameters omega
    cdef public double[5] params
    cdef public double zmin
    cdef public double zmax
    cdef public double _integrated_rate
    cdef public double _number_density(self, double z)
    cdef public double _pdf(self, double z)
    cdef public double _cdf(self, double z)
    cdef public double _integrate(self, double zmin, double zmax)

cdef double _madau_porciani_sfrd(const double z, const double[5] P) nogil

cdef double _madau_fragos_sfrd(const double z, const double[5] P) nogil

cdef double _powerlaw(const double z, const double[5] P) nogil
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport log, exp, sqrt, cos, fabs, sin, sinh, M_PI, \
    erf, erfc, HUGE_VAL, log1p
from scipy.optimize import newton
from cosmolisa.cosmology cimport CosmologicalParameters
from sys import exit


cdef class PopulationModel:
    """Class holding the population model. This is modelling
    the comoving merger density rate. To get actual estimates
    we need the cosmological volume elements and time delay. 
    """
    def __cinit__(self,
                  double r0,
                  double p1,
                  double p2,
                  double p3,
                  double p4,
                  CosmologicalParameters omega,
                  double zmin,
                  double zmax,
                  str density_model=None):

        self.zmin = zmin
        self.zmax = zmax
        self.params[0] = r0
        self.params[1] = p1
        self.params[2] = p2
        self.params[3] = p3
        self.params[4] = p4
        self.omega = omega
        self._integrated_rate = -1
        self.density_model = density_model

        if (self.density_model == 'madau-porciani'):
            self.density = &_madau_porciani_sfrd
        elif (self.density_model == 'madau-fragos'):
            self.density = &_madau_fragos_sfrd
        elif (self.density_model == 'powerlaw'):
            self.density = &_powerlaw
        else:
            self.density = NULL
            print("model {} unknown!".format(self.density_model))
            exit()
        
    def integrated_rate(self):
        return self._integrated_rate_()

    cdef double _integrated_rate_(self):
        if (self._integrated_rate == -1):
            self._integrated_rate = self._integrate(self.zmin, self.zmax)

        return self._integrated_rate

    def number_density(self, double z):
        return self._number_density(z)

    # This is SFRD * (1/(1+z)) * dV/dz.
    cdef double _number_density(self, double z):
        return (self.density(z, self.params)
                * self.omega._UniformComovingVolumeDensity(z))

    def pdf(self, double z):
        return self._pdf(z)

    cdef double _pdf(self, double z):
        return self._number_density(z) / self._integrated_rate_()

    def cdf(self, double z):
        return self._cdf(z)

    cdef double _cdf(self, double z):
        return self._integrate(self.zmin, z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    # Trapezoidal rule with uniform grid.
    cdef double _integrate(self, double zmin, double zmax):
        cdef unsigned int i = 0 
        cdef unsigned int n = 100
        cdef double h = (zmax - zmin)/n
        cdef double z = zmin + h
        cdef double result = 0.5 * (self._number_density(zmin)
                                    + self._number_density(zmax))

        for i in range(1, n):
            result += self._number_density(z)
            z += h

        return result * h

    def __call__(self, double z):
        return self._number_density(z)

cdef double _madau_porciani_sfrd(const double z,
                                    const double[5] P) nogil:
    """From <arXiv:astro-ph/0505181>.
    P[0]: r0
    P[1]: W
    P[2]: Q
    P[3]: R
    """        
    return P[0] * (1.0 + P[1]) * exp(P[2]*z)/(exp(P[3]*z) + P[1])

cdef double _madau_fragos_sfrd(const double z,
                                const double[5] P) nogil:
    """From <arXiv:1606.07887>.
    """
    return P[0] * ((1.0 + z)**P[1])/(1.0 + ((1.0 + z)/P[2])**P[3])

cdef double _powerlaw(const double z,
                        const double[5] P) nogil:
    return P[0] * (1.0 + z)**P[1]

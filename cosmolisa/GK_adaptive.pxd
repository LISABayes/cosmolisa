import cython
cimport cython

import numpy as np
cimport numpy as np

cdef class GKIntegrator:

    cdef public int dimension
    cdef public int minintervals
    cdef public int limit
    cdef public double tolerance
    cdef public np.ndarray y
    cdef public object integrand
    cdef public object args
    cdef public double I
    cdef public double err
    cdef public np.ndarray a
    cdef public np.ndarray b
    
    cdef (double, double) _integrate(self, object f, object args, np.ndarray a, np.ndarray b)
    cdef (double, double) _gausskronrod_integrate(self,
                                                  double a,
                                                  double b,
                                                  int dimension)
    cdef (double, double) _gausskronrod_integrate_adaptive(self, int dimension)

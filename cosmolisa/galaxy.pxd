cimport cosmolisa.cosmology
from cosmolisa.cosmology cimport CosmologicalParameters
cdef class GalaxyDistribution:
    cdef public double n0
    cdef public double logMstar0
    cdef public double alpha0
    cdef public double logMmin
    cdef public double logMmax
    cdef public double logMthreshold
    cdef public double sky_coverage
    cdef public double zmin
    cdef public double zmax
    cdef public double ramin
    cdef public double ramax
    cdef public double decmin
    cdef public double decmax
    cdef public CosmologicalParameters omega
    cdef public double _pmax
    cdef public double _normalisation
    cdef public double _detected_normalisation
    cdef public double _non_detected_normalisation
    cdef public int slope_model_choice
    cdef public int cutoff_model_choice
    cdef public int density_model_choice
    cdef public double logMstar_exponent
    cdef public double alpha_exponent
    cdef public double number_density_exponent

    cdef double _evaluate(self, double log10M, double z) nogil
    cdef double _evaluate_detected(self, double log10M, double z) nogil
    cdef double _evaluate_non_detected(self, double log10M, double z) nogil
    cdef double _get_number_of_galaxies(self, double zmin, double zmax, int selection) nogil
    cdef double _get_normalisation(self, double zmin, double zmax) nogil
    cdef double _get_detected_normalisation(self, double zmin, double zmax) nogil
    cdef double _get_non_detected_normalisation(self, double zmin, double zmax) nogil
    cdef double _pdf(self, double m, double z) nogil
    cdef double _pdf_detected(self, double m, double z) nogil
    cdef double _pdf_non_detected(self, double m, double z) nogil
    cdef double _get_pmax(self) nogil
    cdef tuple _sample(self, double zmin, double zmax, double ramin, double ramax, double decmin, double decmax, int selection)
    cdef double _loglikelihood(self, const double[::1] M, const double[::1] Z) nogil
    cdef double _luminosity_function(self, double M, double Z, int selection) nogil
    
cdef double _absolute_magnitude(double apparent_magnitude, double dl) nogil
cdef double _log_stirling_approx(double n) nogil

"""
# cython: profile=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log, exp, sqrt, cos, fabs, sin, sinh, M_PI, \
    erf, erfc, HUGE_VAL, log1p
from scipy.optimize import newton

from cosmolisa.cosmology cimport CosmologicalParameters, \
    _StarFormationDensity, _IntegrateRateWeightedComovingVolumeDensity
from cosmolisa.galaxy cimport GalaxyDistribution

cdef inline double log_add(double x, double y) nogil: 
    return x + log(1.0+exp(y-x)) if x >= y else y + log(1.0+exp(x-y))

def logLikelihood_single_event(const double[:,::1] hosts,
                               const double meandl,
                               const double sigmadl,
                               CosmologicalParameters omega,
                               const double event_redshift,
                               const double zmin=0.0,
                               const double zmax=1.0):
    """Likelihood function p( Di | O, M, I) for a single GW event
    of data Di assuming cosmological model M and parameters O.
    Following the formalism of <arXiv:2102.01708>.
    Loops over all possible hosts to accumulate the likelihood.
    Parameters:
    ===============
    hosts: :obj: 'numpy.array' with shape Nx4. The columns are
        redshift, redshift_error, angular_weight, magnitude
    meandl: :obj: 'numpy.double': mean of the luminosity distance dL
    sigmadl: :obj: 'numpy.double': standard deviation of dL
    omega: :obj: 'lal.CosmologicalParameter': cosmological parameter
        structure O
    event_redshift: :obj: 'numpy.double': redshift for the GW event
    zmin: :obj: 'numpy.double': minimum GW redshift
    zmax: :obj: 'numpy.double': maximum GW redshift
    """
    return _logLikelihood_single_event(hosts, meandl, sigmadl, omega,
                                       event_redshift, zmin=zmin, zmax=zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _logLikelihood_single_event(const double[:,::1] hosts,
                                        const double meandl,
                                        const double sigmadl,
                                        CosmologicalParameters omega,
                                        const double event_redshift,
                                        double zmin=0.0,
                                        double zmax=1.0) nogil:

    cdef unsigned int j
    cdef double dl
    cdef double logL_galaxy
    cdef double logL_detector
    cdef double sigma_z, score_z
    cdef double weak_lensing_error
    cdef unsigned int N = hosts.shape[0]
    cdef double logTwoPiByTwo = 0.5*log(2.0*M_PI)
    cdef double logL = -HUGE_VAL

    # Predict dL from the cosmology O and the redshift z_gw:
    # d(O, z_GW).
    dl = omega._LuminosityDistance(event_redshift)
    # sigma_WL and combined sigma entering the detector likelihood:
    # p(Di | dL, z_gw, M, I).
    weak_lensing_error = _sigma_weak_lensing(event_redshift, dl)
    cdef double SigmaSquared = sigmadl**2 + weak_lensing_error**2
    cdef double logSigmaByTwo = 0.5*log(SigmaSquared)
    # 1/sqrt{2pi*SigmaSquared}*exp(-0.5*(dL-d(O, z_GW))^2/SigmaSquared)
    logL_detector = (-0.5*(dl-meandl)*(dl-meandl)/SigmaSquared
                     - logSigmaByTwo - logTwoPiByTwo)

    # Sum over the observed-galaxy redshifts: p(z_GW | dL, O, M, I) =
    # sum_j^Ng (w_j/sqrt{2pi}*sig_z_j)*exp(-0.5*(z_j-z_GW)^2/sig_z_j^2)
    for j in range(N):
        # Estimate sig_z_j ~= (z_jobs-z_jcos) = (v_pec/c)*(1+z_j).
        sigma_z = hosts[j,1] * (1+hosts[j,0])
        # Compute the full single-galaxy term to be summed over Ng.
        score_z = (event_redshift - hosts[j,0])/sigma_z
        logL_galaxy = (log(hosts[j,2]) - log(sigma_z)
                       - 0.5*score_z*score_z - logTwoPiByTwo)
        logL = log_add(logL, logL_galaxy)
        
    # p(Di | d(O, z_GW), z_GW, O, M, I) * p(z_GW | dL, O, M, I)
    return logL_detector + logL

def sigma_weak_lensing(const double z, const double dl):
    return _sigma_weak_lensing(z, dl)

cdef inline double _sigma_weak_lensing(const double z, const double dl) nogil:
    """Weak lensing error. From <arXiv:1601.07112v3>,
    Eq. (7.3) corrected by a factor 0.5 
    to match <arXiv:1004.3988v2>.
    Parameters:
    ===============
    z: :obj:'numpy.double': redshift
    dl: :obj:'numpy.double': luminosity distance
    """
    return 0.5*0.066*dl*((1.0-(1.0+z)**(-0.25))/0.25)**1.8

def em_selection_function(double dl):
    return _em_selection_function(dl)

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double _em_selection_function(double dl) nogil:
    """Completeness function f(dL) currently used for plotting."""
    return (1.0-dl/12000.)/(1.0+(dl/3700.0)**7)**1.35

def logLikelihood_single_event_sel_fun(const double[:,::1] hosts,
                                       double meandl,
                                       double sigmadl,
                                       CosmologicalParameters omega,
                                       GalaxyDistribution gal, 
                                       double event_redshift,
                                       double zmin=0.0,
                                       double zmax=1.0):

    return _logLikelihood_single_event_sel_fun(hosts, meandl, sigmadl,
                                               omega, gal, event_redshift,
                                               zmin=zmin, zmax=zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _logLikelihood_single_event_sel_fun(const double[:,::1] hosts,
                                                double meandl,
                                                double sigmadl,
                                                CosmologicalParameters omega,
                                                GalaxyDistribution gal,
                                                double event_redshift,
                                                double zmin=0.0,
                                                double zmax=1.0) nogil:
    """Single-event likelihood function enforcing the 
    selection function.
    Parameters:
    ===============
    hosts: :obj: 'numpy.array' with shape Nx4. The columns are
        redshift, redshift_error, angular_weight, magnitude
    meandl: :obj: 'numpy.double': mean of the luminosity distance dL
    sigmadl: :obj: 'numpy.double': standard deviation of dL
    omega: :obj: 'lal.CosmologicalParameter': cosmological parameter
        structure O
    gal: :obj: 'galaxy.GalaxyDistribution'. Galaxy distribution 
        function 
    event_redshift: :obj: 'numpy.double': redshift for the GW event
    zmin: :obj: 'numpy.double': minimum GW redshift
    zmax: :obj: 'numpy.double': maximum GW redshift
    """
    cdef double logL = -HUGE_VAL
    cdef double p_out_cat = -HUGE_VAL
    logL = _logLikelihood_single_event(hosts, meandl, sigmadl, omega, 
                                       event_redshift, zmin, zmax)
    p_out_cat = (gal._get_non_detected_normalisation(zmin, zmax)
                 /gal._get_normalisation(zmin, zmax))

    return logL + log1p(-p_out_cat)


cpdef double find_redshift(CosmologicalParameters omega, double dl):
    return newton(objective, 1.0, args=(omega,dl))

cdef double objective(double z, CosmologicalParameters omega, double dl):
    return dl - omega._LuminosityDistance(z)

def integrated_rate(double r0,
                    double W,
                    double R,
                    double Q,
                    CosmologicalParameters omega,
                    double zmin=0.0,
                    double zmax=1.0):
    return _integrated_rate(r0, W, R, Q, omega, zmin, zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _integrated_rate(const double r0,
                             const double W,
                             const double R,
                             const double Q,
                             CosmologicalParameters omega,
                             double zmin,
                             double zmax) nogil:
    return _IntegrateRateWeightedComovingVolumeDensity(r0, W, R, Q,
                                                       omega, zmin, zmax)


def logLikelihood_single_event_rate_only(CosmologicalParameters O,
                                         double z,
                                         double r0,
                                         double W,
                                         double R,
                                         double Q,
                                         double N):
    return _logLikelihood_single_event_rate_only(O, z, r0, W, Q, R, N)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _logLikelihood_single_event_rate_only(CosmologicalParameters O,
                                                  double z,
                                                  double r0,
                                                  double W,
                                                  double R,
                                                  double Q,
                                                  double N) nogil:
    return (log(_StarFormationDensity(z, r0, W, R, Q))
            + log(O._UniformComovingVolumeDensity(z)) - log(N))


##########################################################
#
#               Selection probability functions
#
##########################################################

def gw_selection_probability_sfr(const double zmin,
                                 const double zmax,
                                 const double r0,
                                 const double W,
                                 const double R,
                                 const double Q,
                                 const double SNR_threshold,
                                 CosmologicalParameters omega):
    return _gw_selection_probability_sfr(zmin, zmax, r0, W, R, Q,
                                         SNR_threshold, omega)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _gw_selection_probability_sfr(const double zmin,
                                          const double zmax,
                                          const double r0,
                                          const double W,
                                          const double R,
                                          const double Q,
                                          const double SNR_threshold,
                                          CosmologicalParameters omega) nogil:

    cdef int i
    cdef int N = 64
    cdef double I = 0.0
    cdef double dz = (zmax-zmin)/N
    cdef double z  = zmin
    for i in range(N):
        I += _gw_selection_probability_integrand_sfr(z, r0, W, R, Q, 
                                                     SNR_threshold, omega)
        z += dz
    return I*dz

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _gw_selection_probability_integrand_sfr(
        const double z,
        const double r0,
        const double W,
        const double R,
        const double Q,
        const double SNR_threshold,
        CosmologicalParameters omega) nogil:

    cdef double dl = omega._LuminosityDistance(z)
    cdef double sigma = _distance_error_vs_snr(SNR_threshold)
    cdef double sigma_total = sqrt(_sigma_weak_lensing(z, dl)**2 + sigma**2)
    # the following is the distance threshold assuming 
    # a simple scaling law for the SNR
    cdef double Dthreshold = _threshold_distance(SNR_threshold)
    cdef double A = (_StarFormationDensity(z, r0, W, R, Q)
                     * omega._UniformComovingVolumeDensity(z))
    cdef double denominator = sqrt(2.0)*sigma_total
    cdef double integrand = 0.5*A*(erf(dl/denominator)
                                   - erf((dl-Dthreshold)/denominator))
    return integrand
    
def snr_vs_distance(double d):
    return _snr_vs_distance(d)

cdef inline double _snr_vs_distance(double d) nogil:
    """From a log-linear regression on M106."""
    return 23299.606754*d**(-0.741036)

def distance_error_vs_snr(double snr):
    return _distance_error_vs_snr(snr)
    
cdef inline double _distance_error_vs_snr(double snr) nogil:
    """From a log-linear regression on M106."""
    return 23912.196795*snr**(-1.424880)

def threshold_distance(double SNR_threshold):
    return _threshold_distance(SNR_threshold)
    
cdef inline double _threshold_distance(double SNR_threshold) nogil:
    # D0       = 1748.50, SNR0     = 87
    return (SNR_threshold/23299.606754)**(-1./0.741036)
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _log_stirling_approx(double n) nogil:
    return n*log(n)-n if n > 0 else 0
    
def log_stirling_approx(double n):
    return _log_stirling_approx(n)

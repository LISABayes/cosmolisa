"""
# cython: profile=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
from __future__ import division
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from libc.math cimport log,exp,sqrt,cos,fabs,sin,sinh,M_PI,erf,erfc,HUGE_VAL,log1p
cimport cython
from scipy.special import logsumexp
from scipy.optimize import newton
from scipy.integrate import quad
from cosmolisa.cosmology cimport CosmologicalParameters, _StarFormationDensity, _IntegrateRateWeightedComovingVolumeDensity
from cosmolisa.galaxy cimport GalaxyDistribution
from libc.math cimport isfinite

cdef inline double log_add(double x, double y) nogil: return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))

def logLikelihood_single_event(const double[:,::1] hosts,
                               const double meandl,
                               const double sigma,
                               CosmologicalParameters omega,
                               const double event_redshift,
                               const double zmin = 0.0,
                               const double zmax = 1.0):
    """
    Likelihood function for a single GW event.
    Loops over all possible hosts to accumulate the likelihood
    Parameters:
    ===============
    hosts: :obj:'numpy.array' with shape Nx3. The columns are redshift, redshift_error, angular_weight
    meandl: :obj: 'numpy.double': mean of the DL marginal likelihood
    sigma: :obj:'numpy.double': standard deviation of the DL marginal likelihood
    omega: :obj:'lal.CosmologicalParameter': cosmological parameter structure
    event_redshift: :obj:'numpy.double': redshift for the the GW event
    em_selection :obj:'numpy.int': apply em selection function. optional. default = 0
    zmin: :obj:'numpy.double': minimum redshift
    zmax: :obj:'numpy.double': maximum redshift
    """
    return _logLikelihood_single_event(hosts, meandl, sigma, omega, event_redshift, zmin = zmin, zmax = zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _logLikelihood_single_event(const double[:,::1] hosts,
                                        const double meandl,
                                        const double sigma,
                                        CosmologicalParameters omega,
                                        const double event_redshift,
                                        double zmin = 0.0,
                                        double zmax = 1.0) nogil:

    cdef unsigned int i
    cdef double dl
    cdef double logL_galaxy
    cdef double sigma_z, score_z
    cdef double weak_lensing_error
    cdef unsigned int N           = hosts.shape[0]
    cdef double logTwoPiByTwo     = 0.5*log(2.0*M_PI)
    cdef double logL              = -HUGE_VAL
    cdef double logLn             = -HUGE_VAL
    cdef double logp_detection    = 0.0
    cdef double logp_nondetection = 0.0

#    # p(z_gw|O H I) #currently implemented in prior.pyx
#    cdef double log_norm = log(omega.IntegrateComovingVolumeDensity(zmax))
#    cdef double logP     = log(omega.UniformComovingVolumeDensity(event_redshift))-log_norm

    # Predict dl from the cosmology O and the redshift z_gw
    dl = omega._LuminosityDistance(event_redshift)

    # Factors multiplying exp(-0.5*((dL-d(zgw,O))/sig_dL)^2) in p(Di | dL z_gw H I)
    weak_lensing_error            = _sigma_weak_lensing(event_redshift, dl)
    cdef double SigmaSquared      = sigma**2+weak_lensing_error**2
    cdef double logSigmaByTwo     = 0.5*log(SigmaSquared)
#    cdef double[:,::1] hosts_view = hosts #this is a pointer to the data of the array hosts to remove the numpy overhead

    # p(G| dL z_gw O H I): sum over the observed-galaxy redshifts:
    # sum_i^Ng w_i*exp(-0.5*(z_i-zgw)^2/sig_z_i^2)
    for i in range(N):
        sigma_z     = hosts[i,1]*(1+hosts[i,0])
        score_z     = (event_redshift-hosts[i,0])/sigma_z
        logL_galaxy = -0.5*score_z*score_z+log(hosts[i,2])-log(sigma_z)-logTwoPiByTwo
        logL        = log_add(logL,logL_galaxy)
        
    # p(Di |...)*(p(G|...)+p(barG|...))*p(z_gw |...)
    return -0.5*(dl-meandl)*(dl-meandl)/SigmaSquared-logTwoPiByTwo-logSigmaByTwo+logL

def sigma_weak_lensing(const double z, const double dl):
    return _sigma_weak_lensing(z, dl)

cdef inline double _sigma_weak_lensing(const double z, const double dl) nogil:
    """
    Weak lensing error. From <arXiv:1601.07112v3>
    Parameters:
    ===============
    z: :obj:'numpy.double': redshift
    dl: :obj:'numpy.double': luminosity distance
    """
    return 0.066*dl*((1.0-(1.0+z)**(-0.25))/0.25)**1.8

def em_selection_function(double dl):
    return _em_selection_function(dl)

# Completeness function f(dL) currently available in the code
@cython.cdivision(True)
@cython.boundscheck(False)
cdef double _em_selection_function(double dl) nogil:
    return (1.0-dl/12000.)/(1.0+(dl/3700.0)**7)**1.35


def logLikelihood_single_event_sel_fun(const double[:,::1] hosts, double meandl, double sigmadl, CosmologicalParameters omega, GalaxyDistribution gal, double event_redshift, int approx_int = 0, double zmin = 0.0, double zmax = 1.0):
    return _logLikelihood_single_event_sel_fun(hosts, meandl, sigmadl, omega, gal, event_redshift, approx_int = approx_int, zmin = zmin, zmax = zmax)

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
                                                int approx_int = 0,
                                                double zmin = 0.0,
                                                double zmax = 1.0) nogil:
    """
    Single-event likelihood function enforcing the selection function.

    Parameters:
    ===============
    hosts:            :obj: 'numpy.array'.               Shape Nx3. The columns are redshift, redshift_error, angular_weight
    meandl:           :obj: 'numpy.double'.              Mean of the DL marginal likelihood
    sigmadl:          :obj: 'numpy.double'.              Standard deviation of the DL marginal likelihood
    omega:            :obj: 'lal.CosmologicalParameter'. Cosmological parameter structure
    event_redshift:   :obj: 'numpy.double'.              Redshift of the GW event
    approx_int:       :obj: 'numpy.int'.                 Flag to choose whether or not to approximate the in-catalogue integral
    zmin, zmax        :obj: 'numpy.double'.              GW event min,max redshift
    """
    cdef double logL      = -HUGE_VAL
    cdef double p_out_cat = -HUGE_VAL
    logL                  = _logLikelihood_single_event(hosts, meandl, sigmadl, omega, event_redshift, zmin, zmax)
    p_out_cat             = gal._get_non_detected_normalisation(zmin, zmax)/gal._get_normalisation(zmin, zmax)

    return logL+log1p(-p_out_cat)

#################
## UNUSED CODE ##
#################

@cython.cdivision(True)
@cython.boundscheck(False)
cpdef double em_selection_function_number_density(double dl):
    return (1.0)/(1.0+(dl/3700.0)**7)**1.35

cpdef double em_selection_function_normalisation(double zmin, double zmax, CosmologicalParameters omega, int N = 1):
    cdef double tmp
    cdef int i      = 0
    cdef double z   = zmin, dz = (zmax-zmin)/100.
    cdef double res = -HUGE_VAL
    for i in range(0,100):
        dl  = omega.LuminosityDistance(z)
        tmp = N*(log(1.0-_em_selection_function(dl))+log(omega._ComovingVolumeElement(z)))#
        res = log_add(res,tmp)
        z  += dz
    return res+log(dz)

cpdef double find_redshift(CosmologicalParameters omega, double dl):
    return newton(objective,1.0,args=(omega,dl))

cdef double objective(double z, CosmologicalParameters omega, double dl):
    return dl - omega._LuminosityDistance(z)

def integrated_rate(double r0, double W, double R, double Q, CosmologicalParameters omega, double zmin = 0.0, double zmax = 1.0):
    return _integrated_rate(r0, W, R, Q, omega, zmin, zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _integrated_rate(const double r0, const double W, const double R, const double Q, CosmologicalParameters omega, double zmin, double zmax) nogil:
    return _IntegrateRateWeightedComovingVolumeDensity(r0, W, R, Q, omega, zmin, zmax)


def logLikelihood_single_event_rate_only(CosmologicalParameters O, double z, double r0, double W, double R, double Q, double N):
    return _logLikelihood_single_event_rate_only(O, z, r0, W, Q, R, N)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _logLikelihood_single_event_rate_only(CosmologicalParameters O, double z, double r0, double W, double R, double Q, double N) nogil:
    return log(_StarFormationDensity(z, r0, W, R, Q))+log(O._UniformComovingVolumeDensity(z))-log(N)


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
                                 CosmologicalParameters omega,
                                 const double norm):
    return _gw_selection_probability_sfr(zmin, zmax, r0, W, R, Q, SNR_threshold, omega, norm)

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
                                          CosmologicalParameters omega,
                                          double norm) nogil:

    cdef int i
    cdef int N = 64
    cdef double I = 0.0
    cdef double dz = (zmax-zmin)/N
    cdef double z  = zmin
    for i in range(N):
        I += _gw_selection_probability_integrand_sfr(z, r0, W, R, Q, SNR_threshold, omega)
        z += dz
    return I*dz/norm

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _gw_selection_probability_integrand_sfr(const double z,
                                                    const double r0,
                                                    const double W,
                                                    const double R,
                                                    const double Q,
                                                    const double SNR_threshold,
                                                    CosmologicalParameters omega) nogil:

    cdef double dl                = omega._LuminosityDistance(z)
    cdef double sigma             = _distance_error_vs_snr(SNR_threshold)
    cdef double sigma_total       = sqrt(_sigma_weak_lensing(z, dl)**2+sigma**2)
    # this is the distance threshold assuming a simple scaling law for the SNR
    cdef double Dthreshold        = _threshold_distance(SNR_threshold)
    cdef double A                 = _StarFormationDensity(z, r0, W, R, Q)*omega._UniformComovingVolumeDensity(z)
    cdef double denominator       = sqrt(2.0)*sigma_total
    cdef double integrand         = 0.5*A*(erf(dl/denominator)-erf((dl-Dthreshold)/denominator))
    return integrand
    
def snr_vs_distance(double d):
    return _snr_vs_distance(d)

cdef inline double _snr_vs_distance(double d) nogil:
    """ from a log-linear regression on M106 """
    return 23299.606754*d**(-0.741036)

def distance_error_vs_snr(double snr):
    return _distance_error_vs_snr(snr)
    
cdef inline double _distance_error_vs_snr(double snr) nogil:
    """ from a log-linear regression on M106 """
    return 23912.196795*snr**(-1.424880 )

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

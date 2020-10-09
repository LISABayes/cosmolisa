from __future__ import division
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from libc.math cimport log,exp,sqrt,cos,fabs,sin,sinh,M_PI,erf,erfc,HUGE_VAL
cimport cython
from scipy.special import logsumexp
from scipy.optimize import newton
from scipy.integrate import quad
from .cosmology cimport CosmologicalParameters, _StarFormationDensity, _IntegrateRateWeightedComovingVolumeDensity
from libc.math cimport isfinite

cdef inline double log_add(double x, double y) nogil: return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))

def logLikelihood_single_event(const double[:,::1] hosts, double meandl, double sigma, CosmologicalParameters omega, double event_redshift, int em_selection = 0, double zmin = 0.0, double zmax = 1.0):
    return _logLikelihood_single_event(hosts, meandl, sigma, omega, event_redshift, em_selection = em_selection, zmin = zmin, zmax = zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _logLikelihood_single_event(const double[:,::1] hosts, double meandl, double sigma, CosmologicalParameters omega, double event_redshift, int em_selection = 0, double zmin = 0.0, double zmax = 1.0):
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
    """
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

    if (em_selection == 1):
        # Define the 'completeness function' as a weight f(dL),
        # entering the probability p(G| dL z_gw O H I) that the event
        # is located in a detected galaxy and add it to p(G| dL z_gw O H I)
        logp_detection    = log(_em_selection_function(dl))
        logL             += logp_detection
        # Compute the probability p(notG| dL z_gw O H I) that the event
        # is located in a non-detected galaxy as 1-p(G| dL z_gw O H I)
        logp_nondetection = logsumexp([0.0,logp_detection], b = [1,-1])
        logLn             = logp_nondetection

    # p(Di |...)*(p(G|...)+p(barG|...))*p(z_gw |...)
    return (-0.5*(dl-meandl)*(dl-meandl)/SigmaSquared-logTwoPiByTwo-logSigmaByTwo)+logL#+log_add(logL,logLn)#+logP

def sigma_weak_lensing(double z, double dl):
    return _sigma_weak_lensing(z, dl)

cdef inline double _sigma_weak_lensing(double z, double dl) nogil:
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


def logLikelihood_single_event_sel_fun(const double[:,::1] hosts, double meandl, double sigmadl, CosmologicalParameters omega, double event_redshift, int approx_int = 0, double zmin = 0.0, double zmax = 1.0):
    return _logLikelihood_single_event_sel_fun(hosts, meandl, sigmadl, omega, event_redshift, approx_int = approx_int, zmin = zmin, zmax = zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _logLikelihood_single_event_sel_fun(const double[:,::1] hosts, double meandl, double sigmadl, CosmologicalParameters omega, double event_redshift, int approx_int = 0, double zmin = 0.0, double zmax = 1.0):
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
    cdef double dl, logL_galaxy, sigma_z, score_z, weak_lensing_error
    cdef unsigned int i
    cdef unsigned int N           = hosts.shape[0]
    cdef double logTwoPiByTwo     = 0.5*log(2.0*M_PI)
    cdef double log_in_cat        = -HUGE_VAL
    cdef double log_out_cat       = -HUGE_VAL
    cdef double logp_detection    = 0.0
    cdef double logp_nondetection = 0.0

    # p(Di | dL z_gw O H I)
    dl = omega._LuminosityDistance(event_redshift)

    weak_lensing_error            = _sigma_weak_lensing(event_redshift, dl)
    cdef double SigmaSquared      = sigmadl**2 + weak_lensing_error**2
    cdef double logSigmaByTwo     = 0.5*log(SigmaSquared)
    log_dL = -0.5*(dl-meandl)*(dl-meandl)/SigmaSquared - logTwoPiByTwo - logSigmaByTwo

    # p(z_gw | O H I) = in-catalogue term + out-catalog term

    # in-catalogue term
    # sum_i^Ng w_i*exp(-0.5*(z_i-zgw)^2/sig_z_i^2)
    for i in range(N):
        sigma_z     = hosts[i,1]*(1+hosts[i,0]) 
        score_z     = (event_redshift - hosts[i,0])/sigma_z
        logL_galaxy = -0.5*score_z*score_z - log(sigma_z) - logTwoPiByTwo + log(hosts[i,2])
        log_in_cat  = log_add(log_in_cat, logL_galaxy)

    # IntegrateComovingVolume (without 1+z factor): https://lscsoft.docs.ligo.org/lalsuite/lal/_l_a_l_cosmology_calculator_8h.html#a2803b9093568dcba15ca8921b2bece79
    cdef double log_Vc_norm = log(omega._IntegrateComovingVolume(zmax))

    if (approx_int == 1):
        logp_detection = log(_em_selection_function(dl))
    else:
        logp_detection = log(quad(_integrand, 0, zmax, args=(omega))[0]) - log_Vc_norm
    log_in_cat    += logp_detection
    
    # out-catalogue term
    cdef double log_Vc      = log(omega._ComovingVolumeElement(event_redshift))
    log_com_vol             = log_Vc - log_Vc_norm
    logp_nondetection       = logsumexp([0.0, log(_em_selection_function(dl))], b = [1,-1])
    log_out_cat             = logp_nondetection + log_com_vol

    # p(Di | dL z_gw O H I) * p(z_gw | O H I)
    # if not(isfinite(log_out_cat)):
    #     return -np.inf
    # else:
    return log_dL + log_add(log_in_cat, log_out_cat)

# ComovingVolumeElement (without 1+z factor): https://lscsoft.docs.ligo.org/lalsuite/lal/_l_a_l_cosmology_calculator_8h.html#a846d4df0118b0687b9b31f777679b5d3
cpdef double _integrand(double z, CosmologicalParameters omega):
    cdef double dl = omega._LuminosityDistance(z)
    return _em_selection_function(dl)*omega._ComovingVolumeElement(z)



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

def likelihood_normalisation(double zmin, double zmax, const double[:,::1] hosts, double sigma, double SNR, double SNR_threshold, CosmologicalParameters omega):
    return _likelihood_normalisation(zmin, zmax, hosts, sigma, SNR, SNR_threshold, omega)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _likelihood_normalisation(double zmin, double zmax, const double[:,::1] hosts, double sigma, double SNR, double SNR_threshold, CosmologicalParameters omega):
    cdef int i
    cdef int N = 10
    cdef double normalisation = 0.0
    cdef double dz = (zmax-zmin)/N
    cdef double z  = zmin
    for i in range(N):
        normalisation += _likelihood_normalisation_integrand(z, hosts, sigma, SNR, SNR_threshold, omega)*dz
        z += dz
    return normalisation

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _likelihood_normalisation_integrand(double event_redshift, const double[:,::1] hosts, double sigma, double SNR, double SNR_threshold, CosmologicalParameters omega) nogil:
    
    cdef unsigned int i
    cdef unsigned int N           = hosts.shape[0]
    cdef double dl                = omega._LuminosityDistance(event_redshift)
    cdef double erfc_arg          = dl*(SNR_threshold/SNR-1.0)/(sqrt(2.)*sigma)
    cdef double integrand         = 0.0
    cdef double sigma_z, score_z
    for i in range(N):
        sigma_z     = hosts[i,1]*(1+hosts[i,0])
        score_z     = (event_redshift - hosts[i,0])/sigma_z
        integrand  += hosts[i,2]*exp(-0.5*score_z*score_z)/sqrt(2.0*M_PI*sigma_z*sigma_z)
    integrand *= sqrt(M_PI/2.0)*sigma*erfc(erfc_arg)
    return integrand

def logLikelihood_single_event_snr_threshold(const double[:,::1] hosts, double meandl, double sigma, double SNR, CosmologicalParameters omega, double event_redshift, int em_selection = 0, double zmin = 0.0, double zmax = 1.0, double z_threshold = 1.0, double SNR_threshold = 20.0):
    return _logLikelihood_single_event(hosts, meandl, sigma, omega, event_redshift, em_selection = em_selection, zmin = zmin, zmax = zmax)-log(_likelihood_normalisation(0.0, z_threshold, hosts,  sigma, SNR, SNR_threshold, omega))

def logLikelihood_single_event_sfr(const double[:,::1] hosts, double meandl, double sigma, CosmologicalParameters omega, double event_redshift, double r0, double W, double Q, double R, double normalisation, int em_selection = 0, double zmin = 0.0, double zmax = 1.0):
    return _logLikelihood_single_event(hosts, meandl, sigma, omega, event_redshift, em_selection = em_selection, zmin = zmin, zmax = zmax)+log(_StarFormationDensity(event_redshift, r0, W, R, Q))-log(normalisation)

def integrated_rate(double r0, double W, double Q, double R, CosmologicalParameters omega, double zmin = 0.0, double zmax = 1.0):
    return _integrated_rate(r0, W, Q, R, omega, zmin, zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _integrated_rate(const double r0, const double W, const double Q, const double R, CosmologicalParameters omega, double zmin, double zmax) nogil:
    return _IntegrateRateWeightedComovingVolumeDensity(r0, W, Q, R, omega, zmin, zmax)


def logLikelihood_single_event_sfr_non_det(CosmologicalParameters O, double dl_non_det, double z_non_det, double r0, double W, double Q, double R, double zmin = 0.0, double zmax = 1.0):
    return _logLikelihood_single_event_sfr_non_det(O, dl_non_det, z_non_det, r0, W, Q, R, zmin, zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _logLikelihood_single_event_sfr_non_det(CosmologicalParameters O, double dl_non_det, double z_non_det, double r0, double W, double Q, double R, double zmin = 0.0, double zmax = 1.0) nogil:
    
#    cdef double dl = O._LuminosityDistance(z_non_det)
#    cdef double logL = -0.5*(dl-dl_non_det)*(dl-dl_non_det)
    return log(_StarFormationDensity(z_non_det, r0, W, R, Q))+log(O._UniformComovingVolumeDensity(z_non_det))


##########################################################
#
#               Selection probability functions
#
##########################################################

def gw_selection_probability_sfr(double zmin, double zmax, double r0, double W, double R, double Q, double Dmean, double sigma, double SNR, double SNR_threshold, CosmologicalParameters omega, double norm):
    return _gw_selection_probability_sfr(zmin, zmax, r0, W, R, Q, Dmean, sigma, SNR, SNR_threshold, omega)/norm

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _gw_selection_probability_sfr(double zmin, double zmax, double r0, double W, double R, double Q, double Dmean, double sigma, double SNR, double SNR_threshold, CosmologicalParameters omega) nogil:
    
    cdef int i
    cdef int N = 32
    cdef double normalisation = 0.0
    cdef double dz = (zmax-zmin)/N
    cdef double z  = zmin
    for i in range(N):
        normalisation += _gw_selection_probability_integrand_sfr(z, r0, W, R, Q, Dmean, sigma, SNR, SNR_threshold, omega)
        z += dz
    return normalisation*dz


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _gw_selection_probability_integrand_sfr(double z, double r0, double W, double R, double Q, double Dmean, double sigma, double SNR, double SNR_threshold, CosmologicalParameters omega) nogil:

    cdef double dl                = omega._LuminosityDistance(z)
    cdef double sigma_total       = sqrt(_sigma_weak_lensing(z, dl)**2+sigma**2)
    # this is the distance threshold assuming a simple scaling law for the SNR
    cdef double Dthreshold        = (SNR/SNR_threshold)*Dmean
    cdef double A                 = 0.5*_StarFormationDensity(z, r0, W, R, Q)*omega._UniformComovingVolumeDensity(z)
    cdef double denominator      = sqrt(2.0)*sigma_total
    cdef double integrand         = A*(erf(dl/denominator)-erf((dl-Dthreshold)/denominator))
    return integrand

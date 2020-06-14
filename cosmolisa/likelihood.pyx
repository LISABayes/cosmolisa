from __future__ import division
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from libc.math cimport log,exp,sqrt,cos,fabs,sin,sinh,M_PI
cimport cython
from scipy.special.cython_special cimport erfc, hyp2f1
from scipy.special import logsumexp
from scipy.optimize import newton
from .cosmology cimport CosmologicalParameters

cdef inline double log_add(double x, double y) nogil: return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))

def logLikelihood_single_event(double[:,::1] hosts, double meandl, double sigma, CosmologicalParameters omega, double event_redshift, int em_selection = 0, double zmin = 0.0, double zmax = 1.0):
    return _logLikelihood_single_event(hosts, meandl, sigma, omega, event_redshift, em_selection = em_selection, zmin = zmin, zmax = zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _logLikelihood_single_event(double[:,::1] hosts, double meandl, double sigma, CosmologicalParameters omega, double event_redshift, int em_selection = 0, double zmin = 0.0, double zmax = 1.0):
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
    cdef double logL              = -np.inf
    cdef double logLn             = -np.inf
    cdef double logp_detection    = 0.0
    cdef double logp_nondetection = 0.0

#    # p(z_gw|O H I)
#    cdef double log_norm = log(omega.IntegrateComovingVolumeDensity(zmax))
#    cdef double logP     = log(omega.UniformComovingVolumeDensity(event_redshift))-log_norm

    # Predict dl from the cosmology O and the redshift z_gw
    dl = omega.LuminosityDistance(event_redshift)

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
        logp_detection    = log(em_selection_function(dl))
        logL             += logp_detection
        # Compute the probability p(notG| dL z_gw O H I) that the event
        # is located in a non-detected galaxy as 1-p(G| dL z_gw O H I)
        logp_nondetection = logsumexp([0.0,logp_detection], b = [1,-1])
        logLn             = logp_nondetection

    # p(Di |...)*(p(G|...)+p(barG|...))*p(z_gw |...)
    return (-0.5*(dl-meandl)*(dl-meandl)/SigmaSquared-logTwoPiByTwo-logSigmaByTwo)+logL#+log_add(logL,logLn)#+logP

def sigma_weak_lensing(double z, double dl):
    return _sigma_weak_lensing(z, dl)

cdef inline double _sigma_weak_lensing(double z, double dl):
    """
    Weak lensing error. From <arXiv:1601.07112v3>
    Parameters:
    ===============
    z: :obj:'numpy.double': redshift
    dl: :obj:'numpy.double': luminosity distance
    """
    return 0.066*dl*((1.0-(1.0+z)**(-0.25))/0.25)**1.8

# Completeness function f(dL) currently available in the code
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef double em_selection_function(double dl):
    return (1.0-dl/12000.)/(1.0+(dl/3700.0)**7)**1.35

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
    cdef double res = -np.inf
    for i in range(0,100):
        dl  = omega.LuminosityDistance(z)
        tmp = N*(log(1.0-em_selection_function(dl))+log(omega.ComovingVolumeElement(z)))#
        res = log_add(res,tmp)
        z  += dz
    return res+log(dz)

cpdef double find_redshift(CosmologicalParameters omega, double dl):
    return newton(objective,1.0,args=(omega,dl))

cdef double objective(double z, CosmologicalParameters omega, double dl):
    return dl - omega.LuminosityDistance(z)

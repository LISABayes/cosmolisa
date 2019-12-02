from __future__ import division
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from libc.math cimport log,exp,sqrt,cos,fabs,sin,sinh
cimport cython
from scipy.integrate import quad
from scipy.special.cython_special cimport erfc, hyp2f1
from scipy.special import logsumexp
from scipy.optimize import newton

cdef inline double log_add(double x, double y): return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))
cdef inline double linear_density(double x, double a, double b): return a+log(x)*b

@cython.cdivision(True)
@cython.boundscheck(False)
cpdef double logLikelihood_single_event(ndarray[double, ndim=2] hosts, double meandl, double sigma, double Vc, object omega, double event_redshift, int em_selection = 0, double zmin = 0.0, double zmax = 1.0):
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
    cdef unsigned int N = hosts.shape[0]
    cdef double logTwoPiByTwo = 0.5*log(2.0*np.pi)
    cdef double logL_galaxy
    cdef double dl
    cdef double score_z, sigma_z
    cdef double logL = -np.inf
    cdef double weak_lensing_error
    cdef double number_lost = 0.0
    cdef ndarray[double, ndim = 1] redshifts = np.linspace(zmin,zmax,11)
    cdef ndarray local_number
    local_number, _ = np.histogram(hosts[:,0], bins=redshifts)
    
    # estimate the number of galaxies that have been missed
#    for i in range(10):
#        number_lost += local_number[i]*(1.0-em_selection_function(omega.LuminosityDistance(redshifts[i])))
#        print('i:',i,'n:',number_lost,'ln:',local_number[i])
#    exit()

    # predict dl from the cosmology and the redshift
    dl = omega.LuminosityDistance(event_redshift)
    
    # compute the probability p(G|O) that the event is located in a detected galaxy
    cdef double logp_detection = log(em_selection_function(dl))
    # compute the probability p(notG|O) that the event is located in a non-detected galaxy as 1-p(G|O)
#    cdef double logp_nondetection = logsumexp([0.0,logp_detection], b = [1,-1])

#    # guestimate the number of unseen galaxies
#    cdef int Nn = np.maximum(0,int(number_lost))
#    cdef int Ntot = Nn+N
    
    # compute the weak lensing error
    weak_lensing_error = sigma_weak_lensing(event_redshift, dl)
    
    # sum over the observed galaxies
    # p(d|O,zgw,G)p(zgw|O,G) = exp(-0.5*((d-d(zgw,O))/sig_d)^2)*\sum_g w_g*exp(-0.5*(z_g-zgw)^2/sig_z_g^2)
    for i in range(N):

        sigma_z = hosts[i,1]*(1+hosts[i,0])
        score_z = (event_redshift-hosts[i,0])/sigma_z
        logL_galaxy = -0.5*score_z*score_z+log(hosts[i,2])-log(sigma_z)-logTwoPiByTwo#+log(em_selection_function(omega.LuminosityDistance(hosts[i,0])))
        logL = log_add(logL,logL_galaxy)

#    # add the probability that the GW was in a seen galaxy, multiply by p(G|O)
#    logL += logp_detection
#
    cdef double logLn = -np.inf
#    # sum over the unobserved galaxies, assuming they all have redshift = zgw
##     p(d|O,zgw,notG)p(zgw|O,notG) = exp(-0.5*((d-d(zgw,O))/sig_d)^2)*\sum_g (1/Nn)*exp(-0.5*(zgw-zgw)^2/sig_zgw^2)
#    if em_selection == 1:
##        logLn = -0.5*(dl-meandl)*(dl-meandl)/SigmaSquared-logTwoPiByTwo-logSigmaByTwo# + log(Nn)
#        # multiply by p(notG|O)
#        logLn = logp_nondetection
        
#    print('logl = ', logL,'logl n = ',logLn, 'N = ',N,'number_lost',number_lost,'sum of logs = ',log_add(logL,logLn))
#    exit()
    cdef double SigmaSquared = sigma**2+weak_lensing_error**2
    cdef double logSigmaByTwo = 0.5*log(sigma**2+weak_lensing_error**2)
    return (-0.5*(dl-meandl)*(dl-meandl)/SigmaSquared-logTwoPiByTwo-logSigmaByTwo)+log_add(logL,logLn)
    
cdef double sigma_weak_lensing(double z, double dl):
    """
    Weak lensing error. From <REF>
    Parameters:
    ===============
    z: :obj:'numpy.double': redshift
    dl: :obj:'numpy.double': luminosity distance
    """
    return 0.066*dl*((1.0-(1.0+z)**(-0.25))/0.25)**1.8

@cython.cdivision(True)
@cython.boundscheck(False)
cpdef double em_selection_function(double dl):
    return (1.0-dl/12000.)/(1.0+(dl/3700.0)**7)**1.35

@cython.cdivision(True)
@cython.boundscheck(False)
cpdef double em_selection_function_number_density(double dl):
    return (1.0)/(1.0+(dl/3700.0)**7)**1.35

cpdef double em_selection_function_normalisation(double zmin, double zmax, object omega, int N = 1):
    cdef int i = 0
    cdef double z = zmin, dz = (zmax-zmin)/100.
    cdef double res = -np.inf
    cdef double tmp
    for i in range(0,100):
        dl = omega.LuminosityDistance(z)
        tmp = N*(log(1.0-em_selection_function(dl))+log(omega.ComovingVolumeElement(z)))#
        res = log_add(res,tmp)
        z   += dz
    return res+log(dz)

cdef double find_redshift(object omega, double dl):
    return newton(objective,1.0,args=(omega,dl))

cdef double objective(double z, object omega, double dl):
    return dl - omega.LuminosityDistance(z)

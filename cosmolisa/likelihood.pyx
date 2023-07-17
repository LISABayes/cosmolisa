"""
# cython: profile=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log, exp, sqrt, cos, fabs, sin, sinh, M_PI, \
    erf, erfc, HUGE_VAL, log1p, M_SQRT1_2, M_2_SQRTPI
from scipy.optimize import newton

from cosmolisa.cosmology cimport CosmologicalParameters
from cosmolisa.astrophysics cimport PopulationModel
from cosmolisa.galaxy cimport GalaxyDistribution

from cosmolisa.GK_adaptive cimport GKIntegrator # import integrator

cdef inline double log_add(double x, double y) nogil: 
    return x + log(1.0+exp(y-x)) if x >= y else y + log(1.0+exp(x-y))

###########################################################
# Block to compute the integral using the trapezoidal rule
###########################################################
def lk_dark_single_event_trap(const double[:,::1] hosts,
                            const double meandl,
                            const double sigmadl,
                            CosmologicalParameters omega,
                            str model,
                            const double zmin,
                            const double zmax):
    return _lk_dark_single_event_trap(hosts, meandl, sigmadl, omega,
                                      model, zmin, zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _lk_dark_single_event_trap(const double[:,::1] hosts,
                            const double meandl,
                            const double sigmadl,
                            CosmologicalParameters omega,
                            str model,
                            const double zmin,
                            const double zmax):

    cdef int i
    cdef int N = 100
    cdef double dz = (zmax-zmin)/N
    cdef double z  = zmin + dz
    cdef double I = (0.5
        * (_lk_dark_single_event_integrand_trap(zmin, hosts, meandl,
                                                sigmadl, omega, model)
        + _lk_dark_single_event_integrand_trap(zmax, hosts, meandl,
                                               sigmadl, omega, model)))
    for i in range(1, N):
        I += _lk_dark_single_event_integrand_trap(z, hosts, meandl,
                                                  sigmadl, omega, model)
        z += dz
    return I*dz

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _lk_dark_single_event_integrand_trap(const double event_redshift,
                                        const double[:,::1] hosts,
                                        const double meandl,
                                        const double sigmadl,
                                        CosmologicalParameters omega,
                                        str model) nogil:

    cdef unsigned int j
    cdef double dl
    cdef double L_gal = 0.0
    cdef double L_detector = 0.0
    cdef double sigma_z, score_z
    cdef unsigned int N = hosts.shape[0]
    cdef double OneSqrtTwoPi = M_SQRT1_2*0.5*M_2_SQRTPI
    cdef double L_galaxy = 0.0
    if ('Xi0' in model) or ('n1' in model):
        dl = omega._LuminosityDistance_Xi0_n1(event_redshift)
    elif ('b' in model) or ('n2' in model):
        dl = omega._LuminosityDistance_b_n2(event_redshift)
    else:     
        dl = omega._LuminosityDistance(event_redshift)
    cdef double weak_lensing_error = _sigma_weak_lensing(event_redshift, dl)
    cdef double SigmaSquared = sigmadl**2 + weak_lensing_error**2
    cdef double SigmaNorm = OneSqrtTwoPi * 1/sqrt(SigmaSquared)

    # 1/sqrt{2pi*SigmaSquared}*exp(-0.5*(dL-d(O, z_GW))^2/SigmaSquared)
    L_detector = (SigmaNorm * exp(-0.5*(dl-meandl)*(dl-meandl)
                  / SigmaSquared))

    # sum_j^Ng (w_j/sqrt{2pi}*sig_z_j)*exp(-0.5*(z_j-z_GW)^2/sig_z_j^2)
    for j in range(N):
        # Estimate sig_z_j ~= (z_jobs-z_jcos) = (v_pec/c)*(1+z_j).
        sigma_z = hosts[j,1] * (1 + hosts[j,0])
        # Compute the full single-galaxy term to be summed over Ng.
        score_z = (event_redshift - hosts[j,0])/sigma_z
        L_gal = (OneSqrtTwoPi * (1/sigma_z) * hosts[j,2]
                 * exp(-0.5*score_z*score_z))
        L_galaxy += L_gal
    
    # p(Di | d(O, z_GW), z_GW, O, M, I) * p(z_GW | dL, O, M, I)
    return L_detector * L_galaxy

###############################################################
# Block to compute the integral using the Gauss-Kronrod method
###############################################################
def lk_dark_single_event(const double[:,::1] hosts,
                            const double meandl,
                            const double sigmadl,
                            CosmologicalParameters omega,
                            const double zmin,
                            const double zmax):
    return _lk_dark_single_event(hosts, meandl, sigmadl, omega,
                                    zmin, zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _lk_dark_single_event(const double[:,::1] hosts,
                            const double meandl,
                            const double sigmadl,
                            CosmologicalParameters omega,
                            const double zmin,
                            const double zmax):

    cdef int limit = 100
    cdef int minintervals = 2
    cdef double tol = 1e-5
    cdef GKIntegrator Integrator = GKIntegrator(limit, minintervals, tol)

    cdef double z
    cdef tuple args = (hosts, meandl, sigmadl, omega)
    cdef np.ndarray a = np.array([zmin])
    cdef np.ndarray b = np.array([zmax])
    cdef (double, double) result = Integrator.integrate(_lk_dark_single_event_integrand,
                                         args, a, b)
    return result[0]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _lk_dark_single_event_integrand(const double event_redshift,
                                            tuple args):
    cdef unsigned int j
    cdef double dl
    cdef double L_gal = 0.0
    cdef double L_detector = 0.0
    cdef double sigma_z, score_z
    cdef double weak_lensing_error
    cdef unsigned int N = args[0].shape[0]
    cdef double OneSqrtTwoPi = M_SQRT1_2*0.5*M_2_SQRTPI
    cdef double L_galaxy = 0.0
    cdef CosmologicalParameters O = args[3]
    dl = O._LuminosityDistance(event_redshift)
    weak_lensing_error = _sigma_weak_lensing(event_redshift, dl)
    cdef double SigmaSquared = args[2]**2 + weak_lensing_error**2
    cdef double SigmaNorm = OneSqrtTwoPi * 1/sqrt(SigmaSquared)
    # 1/sqrt{2pi*SigmaSquared}*exp(-0.5*(dL-d(O, z_GW))^2/SigmaSquared)
    L_detector = (SigmaNorm * exp(-0.5*(dl-args[1])*(dl-args[1])
                  / SigmaSquared))

    # Sum over the observed-galaxy redshifts: p(z_GW | dL, O, M, I) =
    # sum_j^Ng (w_j/sqrt{2pi}*sig_z_j)*exp(-0.5*(z_j-z_GW)^2/sig_z_j^2)
    for j in range(N):
        # Estimate sig_z_j ~= (z_jobs-z_jcos) = (v_pec/c)*(1+z_j).
        sigma_z = args[0][j,1] * (1 + args[0][j,0])
        # Compute the full single-galaxy term to be summed over Ng.
        score_z = (event_redshift - args[0][j,0])/sigma_z
        L_gal = (OneSqrtTwoPi * (1/sigma_z) * args[0][j,2]
                 * exp(-0.5*score_z*score_z))
        L_galaxy += L_gal
    # p(Di | d(O, z_GW), z_GW, O, M, I) * p(z_GW | dL, O, M, I)
    return L_detector * L_galaxy


def loglk_bright_single_event(const double[:,::1] hosts,
                              const double meandl,
                              const double sigmadl,
                              CosmologicalParameters omega,
                              const double event_redshift,
                              const double zmin=0.0,
                              const double zmax=1.0):
    """Likelihood function p( Di | O, M, I) for a single bright GW 
    event of data Di assuming cosmological model M and parameters O.
    Following the formalism of <arXiv:2102.01708>.
    Use EM host data to compute the likelihood.
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
    return _loglk_bright_single_event(hosts, meandl, sigmadl, omega,
                                      event_redshift, zmin=zmin, zmax=zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _loglk_bright_single_event(const double[:,::1] hosts,
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

    # Use EM counterpart data: p(z_GW | dL, O, M, I) =
    # (1/sqrt{2pi}*sig_z_EM)*exp(-0.5*(z_EM-z_GW)^2/sig_z_EM^2)
    # Read sig_z_EM from EM data.
    sigma_z = hosts[0,1]
    # Compute the full single-galaxy term to be summed over Ng.
    score_z = (event_redshift - hosts[0,0])/sigma_z
    logL_galaxy = (- log(sigma_z)
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


##########################################################
#                                                        #
#                       Corrections                      #
#                                                        #
##########################################################


def em_selection_function(double dl):
    return _em_selection_function(dl)

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double _em_selection_function(double dl) nogil:
    """Completeness function f(dL) currently used for plotting."""
    return (1.0-dl/12000.)/(1.0+(dl/3700.0)**7)**1.35

# def logLikelihood_single_event_sel_fun(const double[:,::1] hosts,
#                                        double meandl,
#                                        double sigmadl,
#                                        CosmologicalParameters omega,
#                                        GalaxyDistribution gal, 
#                                        double event_redshift,
#                                        double zmin=0.0,
#                                        double zmax=1.0):

#     return _logLikelihood_single_event_sel_fun(hosts, meandl, sigmadl,
#                                                omega, gal, event_redshift,
#                                                zmin=zmin, zmax=zmax)

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cdef double _logLikelihood_single_event_sel_fun(const double[:,::1] hosts,
#                                                 double meandl,
#                                                 double sigmadl,
#                                                 CosmologicalParameters omega,
#                                                 GalaxyDistribution gal,
#                                                 double event_redshift,
#                                                 double zmin=0.0,
#                                                 double zmax=1.0) nogil:
#     """Single-event likelihood function enforcing the 
#     selection function.
    # Parameters:
    # ===============
    # hosts: :obj: 'numpy.array' with shape Nx4. The columns are
    #     redshift, redshift_error, angular_weight, magnitude
    # meandl: :obj: 'numpy.double': mean of the luminosity distance dL
    # sigmadl: :obj: 'numpy.double': standard deviation of dL
    # omega: :obj: 'lal.CosmologicalParameter': cosmological parameter
    #     structure O
    # gal: :obj: 'galaxy.GalaxyDistribution'. Galaxy distribution 
    #     function 
    # event_redshift: :obj: 'numpy.double': redshift for the GW event
    # zmin: :obj: 'numpy.double': minimum GW redshift
    # zmax: :obj: 'numpy.double': maximum GW redshift
    # """
    # cdef double logL = -HUGE_VAL
    # cdef double p_out_cat = -HUGE_VAL
    # logL = _lk_dark_single_event(hosts, meandl, sigmadl, omega, 
    #                                    event_redshift, zmin, zmax)
    # p_out_cat = (gal._get_non_detected_normalisation(zmin, zmax)
    #              /gal._get_normalisation(zmin, zmax))

    # return logL + log1p(-p_out_cat)


cpdef double find_redshift(CosmologicalParameters omega, double dl):
    return newton(objective, 1.0, args=(omega,dl))

cdef double objective(double z, CosmologicalParameters omega, double dl):
    return dl - omega._LuminosityDistance(z)

def logLikelihood_single_event_rate_only(double z,
                                         PopulationModel PopMod,
                                         double N):
    return _logLikelihood_single_event_rate_only(z, PopMod, N)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _logLikelihood_single_event_rate_only(double z,
                                                  PopulationModel PopMod,
                                                  double N):
    return (log(PopMod._pdf(z)) - log(N))


def number_of_detectable_gw(PopulationModel PopMod,
                            const double SNR_threshold,
                            dict corr_const):
    return _number_of_detectable_gw(PopMod, SNR_threshold, corr_const)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _number_of_detectable_gw(PopulationModel PopMod,
                                     const double SNR_threshold,
                                     dict corr_const
                                     ):

    cdef int i
    cdef int N = 64
    cdef double zmin = PopMod.zmin
    cdef double zmax = PopMod.zmax
    cdef double dz = (zmax-zmin)/N
    cdef double z  = zmin + dz
    cdef double I = (0.5
        * (_number_of_detectable_gw_integrand(zmin, PopMod,
                                              SNR_threshold, corr_const)
        + _number_of_detectable_gw_integrand(zmax, PopMod,
                                             SNR_threshold, corr_const)))
    for i in range(1, N):
        I += _number_of_detectable_gw_integrand(z, PopMod,
                                                SNR_threshold, corr_const)
        z += dz
    return I*dz

# The integrand is (dR/dz) * p(rho_up | z), where the second factor is
# the probability that a GW with redshift z is detectable. 
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _number_of_detectable_gw_integrand(
        const double z,
        PopulationModel PopMod,
        const double SNR_threshold,
        dict corr_const):

    cdef double rho_dl_const = corr_const["rho_dl_const"]
    cdef double rho_dl_exp = corr_const["rho_dl_exp"]
    cdef double sigma_rho_const = corr_const["sigma_rho_const"]
    cdef double sigma_rho_exp = corr_const["sigma_rho_exp"]

    cdef double dl = PopMod.omega._LuminosityDistance(z)
    cdef double sigmadl = _distance_error_vs_snr(SNR_threshold,
                                                 sigma_rho_const,
                                                 sigma_rho_exp)
    cdef double sigma_total = sqrt(_sigma_weak_lensing(z, dl)**2 + sigmadl**2)
    # The following quantity is the distance threshold
    # assuming a simple scaling law for the SNR.
    cdef double D_threshold = _threshold_distance(SNR_threshold,
                                                  rho_dl_const,
                                                  rho_dl_exp)
    cdef double dRdz = (PopMod._number_density(z))
    cdef double denominator = sqrt(2.0) * sigma_total
    cdef double integrand = 0.5 * dRdz * (erf(dl/denominator)
                                   - erf((dl-D_threshold)/denominator))
    return integrand
    
def snr_vs_distance(double d,
                    double rho_dl_const,
                    double rho_dl_exp):
    return _snr_vs_distance(d, rho_dl_const, rho_dl_exp)

cdef inline double _snr_vs_distance(double d,
                                    double rho_dl_const,
                                    double rho_dl_exp) nogil:
    return rho_dl_const * d**(rho_dl_exp)

def distance_error_vs_snr(double snr,
                          double sigma_rho_const,
                          double sigma_rho_exp):
    return _distance_error_vs_snr(snr, sigma_rho_const, sigma_rho_exp)
    
cdef inline double _distance_error_vs_snr(double snr,
                                          double sigma_rho_const,
                                          double sigma_rho_exp) nogil:
    return sigma_rho_const * snr**(sigma_rho_exp)

def threshold_distance(double SNR_threshold,
                       double rho_dl_const,
                       double rho_dl_exp):
    return _threshold_distance(SNR_threshold, rho_dl_const, rho_dl_exp)
    
cdef inline double _threshold_distance(double SNR_threshold,
                                       double rho_dl_const,
                                       double rho_dl_exp) nogil:
    return (SNR_threshold/rho_dl_const)**(1./rho_dl_exp)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _log_stirling_approx(double n) nogil:
    return n*log(n)-n if n > 0 else 0
    
def log_stirling_approx(double n):
    return _log_stirling_approx(n)

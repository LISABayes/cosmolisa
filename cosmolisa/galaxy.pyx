from __future__ import division
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from libc.math cimport pow,log,log10,isfinite,exp,sqrt,cos,fabs,sin,sinh,M_PI,erf,erfc,HUGE_VAL
cimport cython

from scipy.integrate import quad, dblquad
from cosmolisa.cosmology cimport CosmologicalParameters

from scipy import LowLevelCallable

ctypedef double (*model_pointer)(double, double, double)

cdef double p4log10 = 0.4*log(10.0)
cdef double ln10 = log(10.0)

cdef class GalaxyMassDistribution:
    """
    Returns a Galaxy distribution function in mass and redshift
    To obtain a pure Schechter function, set zmin = zmax = 0.0 and always evaluate the
    distribution at z = 0

    Parameters
    ----------

    """
    cdef public double logMstar0
    cdef public double alpha0
    cdef public double phistar0
    cdef public double logMmin
    cdef public double logMmax
    cdef public double logMthreshold
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
    cdef public double phistar_exponent
#    cdef public slope_model alpha
#    cdef public cutoff_model logMstar
#    cdef public density_model phistar
    
    def __cinit__(self,
                  CosmologicalParameters omega,
                  double phistar0,
                  double phistar_exponent,
                  double logMstar0,
                  double logMstar_exponent,
                  double alpha0,
                  double alpha_exponent,
                  double logMmin,
                  double logMmax,
                  double zmin,
                  double zmax,
                  double ramin,
                  double ramax,
                  double decmin,
                  double decmax,
                  double logMthreshold,
                  int slope_model_choice,
                  int cutoff_model_choice,
                  int density_model_choice):
        
        self.omega                          = omega
        self.phistar0                       = phistar0
        self.phistar_exponent               = phistar_exponent
        self.logMstar0                      = logMstar0
        self.logMstar_exponent              = logMstar_exponent
        self.alpha0                         = alpha0
        self.alpha_exponent                 = alpha_exponent
        self.logMmin                        = logMmin
        self.logMmax                        = logMmax
        self.zmin                           = zmin
        self.zmax                           = zmax
        self.logMthreshold                  = logMthreshold
        self.slope_model_choice             = slope_model_choice
        self.cutoff_model_choice            = cutoff_model_choice
        self.density_model_choice           = density_model_choice
#        self.alpha                          = slope_model(self.alpha0, self.alpha_exponent, self.slope_model_choice)
#        self.logMstar                       = cutoff_model(self.logMstar0, self.logMstar_exponent, self.cutoff_model_choice)
#        self.phistar                        = density_model(self.phistar0, self.phistar_exponent, self.density_model_choice)
        self._pmax                          = -1
        self._normalisation                 = -1
        self._detected_normalisation        = -1
        self._non_detected_normalisation    = -1

    def __call__(self, double logM, double z, int selection):
        if selection == 0:
            return self._evaluate(logM, z)
        elif selection == 1:
            return self._evaluate_detected(logM, z)
        elif selection == 2:
            return self._evaluate_non_detected(logM, z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _evaluate(self, double log10M, double z) nogil:
        """
        Eq. 10 in https://arxiv.org/pdf/1604.00008.pdf
        """
        cdef double logMs = 0.0, As = 0.0, phistar = 0.0
        if self.density_model_choice == 0:
            phistar = self.phistar0
        elif self.density_model_choice == 1:
            phistar = _powerlaw(self.phistar0, self.phistar_exponent, z)
            
        if self.cutoff_model_choice == 0:
            logMs = self.logMstar0
        elif self.cutoff_model_choice == 1:
            logMs = _powerlaw(self.logMstar0, self.logMstar_exponent, z)

        if self.slope_model_choice == 0:
            As = self.alpha0
        elif self.slope_model_choice == 1:
            As = _powerlaw(self.alpha0, self.alpha_exponent, z)
        
        cdef double log_diff = log10M-logMs
        cdef double p10logdiff = pow(10.0,log_diff)
        
        return ln10*phistar*exp(-p10logdiff)*p10logdiff**(As+1)*self.omega._ComovingVolumeElement(z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _evaluate_detected(self, double log10M, double z) nogil:
        """
        Eq. 10 in https://arxiv.org/pdf/1604.00008.pdf
        """
        cdef double logMs = 0.0, As = 0.0, phistar = 0.0
        if self.density_model_choice == 0:
            phistar = self.phistar0
        elif self.density_model_choice == 1:
            phistar = _powerlaw(self.phistar0, self.phistar_exponent, z)
            
        if self.cutoff_model_choice == 0:
            logMs = self.logMstar0
        elif self.cutoff_model_choice == 1:
            logMs = _powerlaw(self.logMstar0, self.logMstar_exponent, z)

        if self.slope_model_choice == 0:
            As = self.alpha0
        elif self.slope_model_choice == 1:
            As = _powerlaw(self.alpha0, self.alpha_exponent, z)
        
        cdef double log_diff = log10M-logMs
        cdef double p10logdiff = pow(10.0,log_diff)
        
        if log10M < self.logMthreshold:
            return 0
        else:
            return ln10*phistar*exp(-p10logdiff)*p10logdiff**(As+1)*self.omega._ComovingVolumeElement(z)
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _evaluate_non_detected(self, double log10M, double z) nogil:
        """
        Eq. 10 in https://arxiv.org/pdf/1604.00008.pdf
        """
        cdef double logMs = 0.0, As = 0.0, phistar = 0.0
        if self.density_model_choice == 0:
            phistar = self.phistar0
        elif self.density_model_choice == 1:
            phistar = _powerlaw(self.phistar0, self.phistar_exponent, z)
            
        if self.cutoff_model_choice == 0:
            logMs = self.logMstar0
        elif self.cutoff_model_choice == 1:
            logMs = _powerlaw(self.logMstar0, self.logMstar_exponent, z)

        if self.slope_model_choice == 0:
            As = self.alpha0
        elif self.slope_model_choice == 1:
            As = _powerlaw(self.alpha0, self.alpha_exponent, z)
        
        cdef double log_diff = log10M-logMs
        cdef double p10logdiff = pow(10.0,log_diff)
        
        if log10M > self.logMthreshold:
            return 0
        else:
            return ln10*phistar*exp(-p10logdiff)*p10logdiff**(As+1)*self.omega._ComovingVolumeElement(z)

#    @cython.boundscheck(False)
#    @cython.wraparound(False)
#    @cython.nonecheck(False)
#    @cython.cdivision(True)
#    cdef double _get_norm(self, int selection = 0):
#
#        return dblquad(self,
#                       self.zmin,
#                       self.zmax,
#                       lambda x: self.logMmin,
#                       lambda x: self.logMmax, args=(selection,))[0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_norm(self, int selection = 0) nogil:
        
        cdef unsigned int i, j
        cdef unsigned int N = 100
        cdef double result = 0.0
        cdef double dz = (self.zmax-self.zmin)/N
        cdef double dm = (self.logMmax-self.logMmin)/N
        cdef double m, z = self.zmin
        for i in range(N):
            
            m = self.logMmin
            for j in range(N):
                if selection == 0: result += self._evaluate(m, z)
                elif selection == 1: result += self._evaluate_detected(m, z)
                elif selection == 2: result += self._evaluate_non_detected(m, z)
                m += dm
            z += dz
            
        return result*dz*dm#dblquad(self, self.zmin, self.zmax, lambda x: self.mmin, lambda x: self.mmax, args=(sel,))[0]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_normalisation(self) nogil:
        if self._normalisation == -1:
            self._normalisation = self._get_norm(0)
        return self._normalisation
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_detected_normalisation(self) nogil:
        if self._detected_normalisation == -1:
            self._detected_normalisation = self._get_norm(1)
        return self._detected_normalisation

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_non_detected_normalisation(self) nogil:
        if self._non_detected_normalisation == -1:
            self._non_detected_normalisation = self._get_norm(2)
        return self._non_detected_normalisation

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _pdf(self, double m, double z) nogil:
        return self._evaluate(m, z)/self._get_normalisation()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _pdf_detected(self, double m, double z) nogil:
        return self._evaluate_detected(m, z)/self._get_detected_normalisation()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _pdf_non_detected(self, double m, double z) nogil:
        return self._evaluate_non_detected(m, z)/self._get_non_detected_normalisation()

    def pdf(self, double logm, double z, int selection = 0):
        if selection == 0:
            return self._pdf(logm, z)
        elif selection == 1:
            return self._pdf_detected(logm, z)
        elif selection == 2:
            return self._pdf_non_detected(logm, z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_pmax(self):

        cdef int i, n = 100
        cdef double m, dm, p, z, dz
        if self._pmax == -1:
            m  = self.Mmin
            dm = (self.Mmax-self.Mmin)/n
            dz = (self.zmax-self.zmin)/n
            for i in range(100):
                z = self.zmin
                for j in range(100):
                    p = self._pdf(m, z)
                    z += dz
                    if p > self._pmax:
                        self._pmax = p
                m += dm
        print(self._pmax)
        return self._pmax

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef tuple _sample(self):

        cdef int i = 0
        cdef double test, prob

        while True:
            test = self._get_pmax() * np.random.uniform(0,1)
            M    = np.random.uniform(self.logMmin,self.logMmax)
            Z    = np.random.uniform(self.zmin,self.zmax)
            prob = self._pdf(M,Z)
            if (test < prob): break
        return M, Z, prob

    def sample(self, int N):
        return np.array([self._sample() for _ in range(N)])
    
    def loglikelihood(self, const double[:,::1] data):
        return self._loglikelihood(data)
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _loglikelihood(self, const double[:,::1] data):
        """
        Eq. 10 in https://arxiv.org/pdf/0805.2946.pdf
        we are using the stirling approximation for the log factorial
        """
        cdef unsigned int i
        cdef double Ntot = self._get_normalisation()
        cdef unsigned int Ndet = data.shape[0]
        cdef double logL = log_stirling_approx(Ntot)-log_stirling_approx(Ndet)-log_stirling_approx(Ntot-Ndet)

        for i in range(Ndet):
            logL += log(self._pdf(data[i,0], data[i,1]))

        logL += (Ntot-Ndet)*log(self._get_non_detected_normalisation()/Ntot)
        return logL
        

cdef class SchechterFunctionMagnitude:
    """
    Returns a Schechter magnitude function for a given set of parameters

    Parameters
    ----------
    Mstar_obs : observed characteristic magnitude used to define
                Mstar = Mstar_obs + 5.*np.log10(h)
    alpha : observed characteristic slope.
    phistar : density (can be set to unity)
    """
    cdef public double Mstar
    cdef public double phistar
    cdef public double alpha
    cdef public double mmin
    cdef public double mmax
    cdef double _norm
    cdef double _pmax
    
    def __cinit__(self, double hubble_parameter, double Mstar, double alpha, double mmin, double mmax, double phistar=1.):
        self.Mstar      = _scaled_magnitude(hubble_parameter, Mstar)
        self.phistar    = _scaled_number_density(hubble_parameter, phistar)
        self.alpha      = alpha
        self.mmin       = mmin
        self.mmax       = mmax
        self._norm       = -1
        self._pmax       = -1
        
    def __call__(self, double m):
        return self._evaluate(m)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _evaluate(self, double m):
        return p4log10*self.phistar \
               * pow(10.0, -0.4*(self.alpha+1.0)*(m-self.Mstar)) \
               * exp(-pow(10, -0.4*(m-self.Mstar)))
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_norm(self):

        if self._norm == -1:

            self._norm = quad(self, self.mmin, self.mmax)[0]
            # self.norm = (gammainc(self.alpha+2, hibound)-gammainc(self.alpha+2, lowbound))*gamma(self.alpha+2)
        return self._norm

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _pdf(self, double m):
        return self._evaluate(m)/self._get_norm()
    
    def pdf(self, double m):
        return self._pdf(m)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_pmax(self):
        
        cdef int i, n = 100
        cdef double m, dm, p
        if self._pmax == -1:
            m  = self.mmin
            dm = (self.mmax-self.mmin)/n
            for i in range(100):
                p = self._pdf(m)
                m += dm
                if p > self._pmax:
                    self._pmax = p
        return self._pmax
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _sample(self):
        
        cdef int i = 0
        cdef double test, prob
        
        while True:
            test = self._get_pmax() * np.random.uniform(0,1)
            M    = np.random.uniform(self.mmin,self.mmax)
            prob = self._pdf(M)
            if (test < prob): break
        return M
    
    def sample(self, int N):
        return np.array([self._sample() for _ in range(N)])
    
cdef inline double _scaled_magnitude(double h, double M_obs):
    """
    Given an observed absolute magnitude, returns absolute magnitude
    """
    return M_obs + 5.*log10(h)

cdef inline double _scaled_number_density(double h, double phistar):
    """
    Given an observed absolute magnitude, returns absolute magnitude
    """
    return phistar*h*h*h

cdef inline double _selection_function(double M, double Mth):
    if M < Mth: return 1
    return 0

def selection_function(double M, double Mth):
    return _selection_function(M, Mth)

cdef class GalaxyDistributionLog:
    """
    Returns a Comoving Volume element weighted Schechter magnitude function for a given set of parameters
    p(z,  M|\Omega) dV/dz dz d\alpha d\delta dM. Allows for evolving Mstar and alpha

    Parameters
    ----------

    """
    cdef public CosmologicalParameters omega
    cdef public double Mstar0
    cdef public double Mstar_exponent
    cdef public double alpha0
    cdef public double alpha_exponent
    cdef public double phistar
    cdef public double mmin
    cdef public double mmax
    cdef public double zmin
    cdef public double zmax
    cdef public double ra_min
    cdef public double ra_max
    cdef public double dec_min
    cdef public double dec_max
    cdef public double _normalisation
    cdef public double _detected_normalisation
    cdef public double _non_detected_normalisation
    cdef public double _pmax
    cdef public int slope_model_choice
    cdef public int cutoff_model_choice
    cdef public slope_model alpha
    cdef public cutoff_model Mstar
    cdef public double apparent_magnitude_threshold
    
    def __cinit__(self,
                  CosmologicalParameters omega,
                  double Mstar0,
                  double Mstar_exponent,
                  double alpha0,
                  double alpha_exponent,
                  double mmin,
                  double mmax,
                  double zmin,
                  double zmax,
                  double ra_min,
                  double ra_max,
                  double dec_min,
                  double dec_max,
                  double apparent_magnitude_threshold,
                  int slope_model_choice,
                  int cutoff_model_choice):
        
        self.omega        = omega
        self.alpha0       = alpha0
        self.Mstar0       = Mstar0
        self.alpha_exponent = alpha_exponent
        self.Mstar_exponent = Mstar_exponent
        self.slope_model_choice  = slope_model_choice
        self.cutoff_model_choice = cutoff_model_choice
        self.alpha      = slope_model(self.alpha0, self.alpha_exponent, self.slope_model_choice)
        self.Mstar      = cutoff_model(self.Mstar0, self.Mstar_exponent, self.cutoff_model_choice)
        self.mmin       = mmin
        self.mmax       = mmax
        self.zmin       = zmin
        self.zmax       = zmax
        self.ra_min     = ra_min
        self.ra_max     = ra_max
        self.dec_min    = dec_min
        self.dec_max    = dec_max
        self.apparent_magnitude_threshold = apparent_magnitude_threshold
        self._normalisation                 = -1
        self._detected_normalisation        = -1
        self._non_detected_normalisation    = -1
        self._pmax      = -1
        self.phistar    = -1
        
    def __call__(self, double m, double z, int sel = 0):
        if sel == 0:
            return self._evaluate(m, z)
        elif sel == 1:
            return self._evaluate_detected(m, z)
        elif sel == 2:
            return self._evaluate_non_detected(m, z)
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _evaluate(self, double m, double z):
        cdef double Mstar = self.Mstar(z)
        return p4log10 \
                * pow(10.0, -0.4*(self.alpha(z)+1.0)*(m-Mstar)) \
                * exp(-pow(10, -0.4*(m-Mstar))) \
                * self.omega._ComovingVolumeElement(z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _evaluate_detected(self, double m, double z):
        cdef double Mstar = self.Mstar(z)
        cdef double Mth = _absolute_magnitude(self.apparent_magnitude_threshold,self.omega._LuminosityDistance(z))
        if m < Mth:
            return p4log10 \
                    * pow(10.0, -0.4*(self.alpha(z)+1.0)*(m-Mstar)) \
                    * exp(-pow(10, -0.4*(m-Mstar))) \
                    * self.omega._ComovingVolumeElement(z)
        else:
            return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _evaluate_non_detected(self, double m, double z):
        cdef double Mstar = self.Mstar(z)
        cdef double Mth = _absolute_magnitude(self.apparent_magnitude_threshold,self.omega._LuminosityDistance(z))
        if m > Mth:
            return p4log10 \
                    * pow(10.0, -0.4*(self.alpha(z)+1.0)*(m-Mstar)) \
                    * exp(-pow(10, -0.4*(m-Mstar))) \
                    * self.omega._ComovingVolumeElement(z)
        else:
            return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_norm(self, int sel = 0):
        
        cdef unsigned int i, j
        cdef unsigned int N = 100
        cdef double result = 0.0
        cdef double dz = (self.zmax-self.zmin)/N
        cdef double dm = (self.mmax-self.mmin)/N
        cdef double m, z = self.zmin
        for i in range(N):
            
            m = self.mmin
            for j in range(N):
                result += self(m, z, sel)*dz*dm
                m += dm
            z += dz
            
        return result#dblquad(self, self.zmin, self.zmax, lambda x: self.mmin, lambda x: self.mmax, args=(sel,))[0]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_normalisation(self):
        if self._normalisation == -1:
            self._normalisation = self._get_norm(0)
        return self._normalisation
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_detected_normalisation(self):
        if self._detected_normalisation == -1:
            self._detected_normalisation = self._get_norm(1)
        return self._detected_normalisation

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_non_detected_normalisation(self):
        if self._non_detected_normalisation == -1:
            self._non_detected_normalisation = self._get_norm(2)
        return self._non_detected_normalisation

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _pdf(self, double m, double z):
        return self._evaluate(m, z)/self._get_normalisation()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _pdf_detected(self, double m, double z):
        return self._evaluate_detected(m, z)/self._get_detected_normalisation()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _pdf_non_detected(self, double m, double z):
        return self._evaluate_non_detected(m, z)/self._get_non_detected_normalisation()

    def pdf(self, double m, double z, int sel = 0):
        if sel == 0:
            return self._pdf(m, z)
        elif sel == 1:
            return self._pdf_detected(m, z)
        elif sel == 2:
            return self._pdf_non_detected(m, z)
    
#    @cython.boundscheck(False)
#    @cython.wraparound(False)
#    @cython.nonecheck(False)
#    @cython.cdivision(True)
#    cdef double _get_pmax(self):
#
#        cdef int i, n = 100
#        cdef double m, dm, p, z, dz, sel = 0
#        if self._pmax == -1:
#            m  = self.mmin
#            dm = (self.mmax-self.mmin)/n
#            dz = (self.zmax-self.zmin)/n
#            for i in range(100):
#                z = self.zmin
#                for j in range(100):
#                    p = self._pdf(m, z,  sel)
#                    z += dz
#                    if p > self._pmax:
#                        self._pmax = p
#                m += dm
#        return self._pmax

#    @cython.boundscheck(False)
#    @cython.wraparound(False)
#    @cython.nonecheck(False)
#    @cython.cdivision(True)
#    cdef tuple _sample(self):
#        #FIXME: not working, probabilities are too small
#        cdef double test, prob
#
#        while True:
#            test = self._get_pmax() * np.random.uniform(0,1)
#            Z    = np.random.uniform(self.zmin,self.zmax)
#            M    = np.random.uniform(self.mmin,_absolute_magnitude(self.apparent_magnitude_threshold,self.omega._LuminosityDistance(Z)))
#            prob = self._pdf(M, Z)
#            if (test < prob): break
#        return M, Z, prob
#
#    def sample(self, int N):
#        return np.array([self._sample() for _ in range(N)])
    
    def loglikelihood(self, double[:,::1] data):
        return self._loglikelihood(data)
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _loglikelihood(self, double[:,::1] data):
        """
        Eq. 10 in https://arxiv.org/pdf/0805.2946.pdf
        we are using the stirling approximation for the log factorial
        """
        cdef unsigned int i
        cdef double Ntot = self._get_normalisation()
        cdef unsigned int Ndet = data.shape[0]
        cdef double logL = log_stirling_approx(Ntot)-log_stirling_approx(Ndet)-log_stirling_approx(Ntot-Ndet)

        for i in range(Ndet):
            logL += log(self._pdf(data[i,0], data[i,1]))

        logL += (Ntot-Ndet)*log(self._get_non_detected_normalisation()/Ntot)
        
        return logL

cdef class slope_model:

    cdef public int model_choice
    cdef public double alpha0
    cdef public double evolution
    cdef model_pointer model
    
    def __cinit__(self, double alpha0, double evolution, int model_choice):
        
        self.alpha0         = alpha0
        self.evolution      = evolution
        self.model_choice   = model_choice
        
        if self.model_choice == 0:
            self.model = &_constant
        elif self.model_choice == 1:
            self.model = &_powerlaw
        else:
            print("model %d not supported!"%self.model_choice)
            exit()
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    def __call__(self, double z):
        return self.model(self.alpha0, self.evolution, z)

cdef class cutoff_model:

    cdef public int model_choice
    cdef public double M0
    cdef public double evolution
    cdef model_pointer model
    
    def __cinit__(self, double M0, double evolution, int model_choice):
        
        self.M0             = M0
        self.evolution      = evolution
        self.model_choice   = model_choice
        
        if self.model_choice == 0:
            self.model = &_constant
        elif self.model_choice == 1:
            self.model = &_powerlaw
        else:
            print("model %d not supported!"%self.model_choice)
            exit()
        
    def __call__(self, double z):
        return self.model(self.M0, self.evolution, z)

cdef class density_model:

    cdef public int model_choice
    cdef public double n0
    cdef public double evolution
    cdef model_pointer model
    
    def __cinit__(self, double n0, double evolution, int model_choice):
        
        self.n0             = n0
        self.evolution      = evolution
        self.model_choice   = model_choice
        
        if self.model_choice == 0:
            self.model = &_constant
        elif self.model_choice == 1:
            self.model = &_powerlaw
        else:
            print("model %d not supported!"%self.model_choice)
            exit()
        
    def __call__(self, double z):
        return self.model(self.n0, self.evolution, z)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _constant(double a, double b, double z) nogil:
    return a

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _powerlaw(double a, double b, double z) nogil:
    return a*pow(1.0+z,b)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _mstar_evolution(double mu, double zstar, double eta, double z):
    """
    Eq (3) in https://arxiv.org/pdf/1201.6365.pdf
    """
    cdef double ratio = (1+z)/(1+zstar)
    return mu*pow(ratio,eta)*exp(-ratio)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _phistar_evolution(double theta, double gamma, double beta, double z):
    """
    Eq (4) in https://arxiv.org/pdf/1201.6365.pdf
    """
    return theta*exp(gamma/pow(1+z,beta))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _absolute_magnitude(double apparent_magnitude, double dl) nogil:
    return apparent_magnitude - 5.0*log10(1e5*dl)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double log_stirling_approx(double n) nogil:
    return n*log(n)-n

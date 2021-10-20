"""
# cython: profile=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
from __future__ import division
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from libc.math cimport pow,log,log10,isfinite,exp,sqrt,cos,fabs,sin,sinh,M_PI,erf,erfc,HUGE_VAL,log1p
cimport cython
from scipy.integrate import quad
from scipy.special.cython_special cimport gammaln
from cosmolisa.cosmology cimport CosmologicalParameters
from scipy.optimize import newton

ctypedef double (*model_pointer)(double, double, double)

cdef double p4log10 = 0.4*log(10.0)
cdef double ln10 = log(10.0)

cdef class GalaxyDistribution:
    """
    Returns a Galaxy distribution function in magnitude and redshift
    To obtain a pure Schechter function, set zmin = zmax = 0.0 and always evaluate the
    distribution at z = 0

    Parameters
    ----------

    """
    
    def __cinit__(self,
                  CosmologicalParameters omega,
                  double n0,
                  double number_density_exponent,
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
                  double sky_coverage,
                  int slope_model_choice,
                  int cutoff_model_choice,
                  int density_model_choice):
        
        self.omega                          = omega
        self.n0                             = n0
        self.number_density_exponent        = number_density_exponent
        self.logMstar0                      = logMstar0
        self.logMstar_exponent              = logMstar_exponent
        self.alpha0                         = alpha0
        self.alpha_exponent                 = alpha_exponent
        self.logMmin                        = logMmin
        self.logMmax                        = logMmax
        self.zmin                           = zmin
        self.zmax                           = zmax
        self.ramin                          = ramin
        self.ramax                          = ramax
        self.decmin                         = decmin
        self.decmax                         = decmax
        self.logMthreshold                  = logMthreshold
        self.sky_coverage                   = sky_coverage
        self.slope_model_choice             = slope_model_choice
        self.cutoff_model_choice            = cutoff_model_choice
        self.density_model_choice           = density_model_choice
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
            phistar = self.n0
        elif self.density_model_choice == 1:
            phistar = _powerlaw(self.n0, self.number_density_exponent, z)
            
        if self.cutoff_model_choice == 0:
            logMs = self.logMstar0
        elif self.cutoff_model_choice == 1:
            logMs = _powerlaw(self.logMstar0, self.logMstar_exponent, z)

        if self.slope_model_choice == 0:
            As = self.alpha0
        elif self.slope_model_choice == 1:
            As = _powerlaw(self.alpha0, self.alpha_exponent, z)
        
        cdef double log_diff = log10M-logMs
        cdef double p10logdiff = pow(10.0,-0.4*log_diff)
        
        return ln10*phistar*exp(-p10logdiff)*p10logdiff**(-0.4*(As+1))*self.omega._ComovingVolumeElement(z)

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
            phistar = self.n0
        elif self.density_model_choice == 1:
            phistar = _powerlaw(self.n0, self.number_density_exponent, z)
            
        if self.cutoff_model_choice == 0:
            logMs = self.logMstar0
        elif self.cutoff_model_choice == 1:
            logMs = _powerlaw(self.logMstar0, self.logMstar_exponent, z)

        if self.slope_model_choice == 0:
            As = self.alpha0
        elif self.slope_model_choice == 1:
            As = _powerlaw(self.alpha0, self.alpha_exponent, z)
        
        cdef double log_diff = log10M-logMs
        cdef double p10logdiff = pow(10.0,-0.4*log_diff)
        
        if log10M > _absolute_magnitude(self.logMthreshold, self.omega._LuminosityDistance(z)):
            return 0
        else:
            return ln10*phistar*exp(-p10logdiff)*p10logdiff**(-0.4*(As+1))*self.omega._ComovingVolumeElement(z)
            
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
            phistar = self.n0
        elif self.density_model_choice == 1:
            phistar = _powerlaw(self.n0, self.number_density_exponent, z)
            
        if self.cutoff_model_choice == 0:
            logMs = self.logMstar0
        elif self.cutoff_model_choice == 1:
            logMs = _powerlaw(self.logMstar0, self.logMstar_exponent, z)

        if self.slope_model_choice == 0:
            As = self.alpha0
        elif self.slope_model_choice == 1:
            As = _powerlaw(self.alpha0, self.alpha_exponent, z)
        
        cdef double log_diff = log10M-logMs
        cdef double p10logdiff = pow(10.0,-0.4*log_diff)
        
        if log10M < _absolute_magnitude(self.logMthreshold, self.omega._LuminosityDistance(z)):
            return 0
        else:
            return ln10*phistar*exp(-p10logdiff)*p10logdiff**(-0.4*(As+1))*self.omega._ComovingVolumeElement(z)

    def get_number_of_galaxies(self, double zmin, double zmax, int selection):
        if zmin == -1: zmin = self.zmin
        if zmax == -1: zmax = self.zmax
        return self._get_number_of_galaxies(zmin, zmax, selection)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_number_of_galaxies(self, double zmin, double zmax, int selection) nogil:
        
        cdef unsigned int i, j
        cdef unsigned int N = 32
        cdef double result = 0.0
        cdef double dz = (zmax-zmin)/N
        cdef double dm = (self.logMmax-self.logMmin)/N
        cdef double m, z = zmin
        for i in range(N):
            
            m = self.logMmin
            for j in range(N):
                if selection == 0: result += self._evaluate(m, z)
                elif selection == 1: result += self._evaluate_detected(m, z)
                elif selection == 2: result += self._evaluate_non_detected(m, z)
                m += dm
            z += dz
            
        return (self.sky_coverage/(4*M_PI))*result*dz*dm

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_normalisation(self, double zmin, double zmax) nogil:
        if self._normalisation == -1:
            self._normalisation = self._get_number_of_galaxies(zmin, zmax, 0)
        return self._normalisation
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_detected_normalisation(self, double zmin, double zmax) nogil:
        if self._detected_normalisation == -1:
            self._detected_normalisation = self._get_number_of_galaxies(zmin, zmax, 1)
        return self._detected_normalisation

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _get_non_detected_normalisation(self, double zmin, double zmax) nogil:
        if self._non_detected_normalisation == -1:
            self._non_detected_normalisation = self._get_number_of_galaxies(zmin, zmax, 2)
        return self._non_detected_normalisation

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _pdf(self, double m, double z) nogil:
        return self._evaluate(m, z)/self._get_normalisation(self.zmin, self.zmax)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _pdf_detected(self, double m, double z) nogil:
        return self._evaluate_detected(m, z)/self._get_detected_normalisation(self.zmin, self.zmax)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _pdf_non_detected(self, double m, double z) nogil:
        return self._evaluate_non_detected(m, z)/self._get_non_detected_normalisation(self.zmin, self.zmax)

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
    cdef double _get_pmax(self) nogil:

        cdef int i, n = 100
        cdef double m, dm, p, z, dz
        if self._pmax == -1:
            m  = self.logMmin
            dm = (self.logMmax-self.logMmin)/n
            dz = (self.zmax-self.zmin)/n
            for i in range(100):
                z = self.zmin
                for j in range(100):
                    p = self._pdf(m, z)
                    z += dz
                    if p > self._pmax:
                        self._pmax = p
                m += dm

        return self._pmax

    def sample_correlated(self, int N, double zmin, double zmax, double ramin, double ramax, double decmin, double decmax, int selection = 0):
        return self._sample_correlated(N, zmin, zmax, ramin, ramax, decmin, decmax, selection = selection)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef np.ndarray[double, mode="c",ndim=2] _sample_correlated(self, int N, double zmin, double zmax, double ramin, double ramax, double decmin, double decmax, int selection):
        """
        we want to sample a set of galaxies following the empirically measured correlation function
        as well as the global distribution of galaxies
        
        we start by noting that given a galaxy, the probability of finding another one within
        a volume dV is given by
        
        dP = n(1+eps(r))dV = n(1+eps(r))(dV/dz) dz
        
        if we enlarge the discourse to include the magnitude, we can write
        
        dP = n(1+eps(r))(dV/dz)\phi(M,z) dz dM
        
        with r =  r(dz)[and potentially dM]
        """
        cdef int i = 0
        cdef int j = 0
        cdef double test, prob
        cdef np.ndarray[double, mode="c",ndim=1] comoving_distance = np.zeros(N, dtype=np.double)
        cdef np.ndarray[double, mode="c",ndim=1] relative_distance
        cdef np.ndarray[double, mode="c",ndim=1] x1, x2
        cdef np.ndarray[double, mode="c",ndim=2] out = np.zeros((N,5), dtype=np.double)
        cdef double M, Z, RA, DEC
        cdef double D, corr = 0
        # begin by sampling a redshift, a magnitude and a sky position within the box
        # note that we are going to ignore correlations in the luminosities
        # that are there in the universe
        cdef np.ndarray[double, mode="c",ndim=1] r = np.linspace(0.0,1000,1000)
        cdef double corr_max = 0
        for j in range(1000):
            corr = _correlation_function(r[j], self.omega.h)
            if corr > corr_max: corr_max = corr
        
        while i<N:
            test = (1+corr_max)*self._get_pmax() * np.random.uniform(0,1)
            M    = np.random.uniform(self.logMmin,self.logMmax)
            Z    = np.random.uniform(zmin,zmax)
            RA   = np.random.uniform(ramin,ramax)
            DEC  = np.arcsin(np.sin(np.random.uniform(decmin,decmax)))
            comoving_distance[i] = self.omega._ComovingDistance(Z)

            if i > 0:
                relative_distance = np.zeros(i, dtype=np.double)
                x1 = np.array([comoving_distance[i],DEC,RA])

                for j in range(i):
                    
                    x2 = np.array([comoving_distance[j],out[j,3],out[j,2]])
                    relative_distance[j] = _distance_spherical(x1,x2)
 
                    
                D    = relative_distance.min()
                corr = _correlation_function(D, self.omega.h)
            prob = (1+corr)*self.pdf(M, Z, selection = selection)

            if (test < prob):
                out[i,1] = Z
                out[i,2] = RA
                out[i,3] = DEC #- np.pi/2.0
                out[i,0] = M
                i += 1

        return out
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef tuple _sample(self, double zmin, double zmax, double ramin, double ramax, double decmin, double decmax, int selection):

        cdef int i = 0
        cdef double test, prob

        while True:
            test = self._get_pmax() * np.random.uniform(0,1)
            M    = np.random.uniform(self.logMmin,self.logMmax)
            Z    = np.random.uniform(zmin,zmax)
            RA   = np.random.uniform(ramin,ramax)
            DEC  = np.arcsin(np.sin(np.random.uniform(decmin,decmax)))
            prob = self.pdf(M, Z, selection = selection)
            if (test < prob): break
        return M, Z, RA, DEC, prob

    def sample(self, double zmin, double zmax, double ramin, double ramax, double decmin, double decmax, int N, int selection = 0):
        return np.array([self._sample(zmin, zmax, ramin, ramax, decmin, decmax, selection = selection) for _ in range(N)])
    
    def loglikelihood(self, const double[::1] M, const double[::1] Z):
        return self._loglikelihood(M,Z)
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _loglikelihood(self, const double[::1] M, const double[::1] Z) nogil:
        """
        Eq. 10 in https://arxiv.org/pdf/0805.2946.pdf
        we are using the log gamma for the log factorial
        """
        cdef unsigned int i
        cdef double Ntot = self._get_normalisation(self.zmin, self.zmax)
        cdef unsigned int Ndet = Z.shape[0]
        cdef double logL = gammaln(Ntot+1)-gammaln(Ndet+1)-gammaln(Ntot-Ndet+1)
        for i in range(Ndet):
            logL += log(self._pdf(M[i], Z[i]))
        if (Ntot-Ndet) > 0.0:
            # see Eq.C8 in https://iopscience.iop.org/article/10.1088/0004-637X/786/1/57/pdf
            logL += (Ntot-Ndet)*log1p(-self._get_detected_normalisation(self.zmin, self.zmax)/Ntot)
        return logL
    
    def luminosity_function(self, double M, double Z, int selection = 0):
        return self._luminosity_function(M, Z, selection)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _luminosity_function(self, double M, double Z, int selection) nogil:
        cdef double N = self._get_normalisation(self.zmin, self.zmax)
        cdef double P = 0.0
        if selection == 0: P = self._pdf(M, Z)
        elif selection == 1: P = self._pdf_detected(M, Z)
        elif selection == 2: P = self._pdf_non_detected(M, Z)
        return N*P/self.omega._ComovingVolumeElement(Z)

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
        cdef double logL = _log_stirling_approx(Ntot)-_log_stirling_approx(Ndet)-_log_stirling_approx(Ntot-Ndet)

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
cdef inline double _apparent_magnitude(double absolute_magnitude, double dl) nogil:
    return absolute_magnitude + 5.0*log10(1e5*dl)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _log_stirling_approx(double n) nogil:
    return n*log(n)-n if n > 0 else 0
    
def log_stirling_approx(double n):
    return _log_stirling_approx(n)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _angular_correlation_function(double theta) nogil:
    """
    from https://academic.oup.com/mnras/article/432/3/1961/1746880
    """
    cdef double A = 10**(-2.12)
    cdef double one_minus_gamma  = -0.72
    return A*theta**one_minus_gamma

def angular_correlation_function(double theta):
    return _angular_correlation_function(theta)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _correlation_function(double r, double h) nogil: #r in Mpc
    """
    from https://academic.oup.com/mnras/article/432/3/1961/1746880
    """
    cdef double r0 = 5.5/h
    cdef double gamm  = 1.72
    if r < 1e-2:
        return 0
    return (r/r0)**(-gamm)

def correlation_function(r, h):
    return np.array([_correlation_function(ri, h) for ri in r])

cdef double find_redshift(CosmologicalParameters omega, double dc):
    return newton(objective,1.0,args=(omega,dc))

cdef double objective(double z, CosmologicalParameters omega, double dc) nogil:
    return dc - omega._ComovingDistance(z)

cdef cartesian_to_spherical(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arctan2(hxy, z)
    phi = np.arctan2(y, x)
    n1 = x ** 2 + y ** 2
    n2 = n1 + z ** 2
    return r, theta, phi

cdef spherical_to_cartesian(r, th, p):

    x = r * np.cos(p) * np.sin(th)
    y = r * np.sin(p) * np.sin(th)
    z = r * np.cos(th)

    return x, y, z

cdef double _distance_spherical(np.ndarray[double,mode="c",ndim=1] x1, np.ndarray[double,mode="c",ndim=1] x2):
    cdef double d1 = x1[0]**2+x2[0]**2
    cdef double d2 = -2.0*(x1[0]*x2[0])*cos(x1[1]-x2[1])-2.0*(x1[0]*x2[0])*sin(x1[1])*sin(x2[1])*(cos(x1[2]-x2[2])-1.0)
    return sqrt(d1+d2)

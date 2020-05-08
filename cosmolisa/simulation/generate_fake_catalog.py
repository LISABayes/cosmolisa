import numpy as np
import scipy.stats
import lal
import os
import sys

def redshift_rejection_sampling(min, max, p, pmax, norm):
    """
    Samples the cosmologically correct redshift
    distribution
    
    Parameters
    ===========
    min ::`obj`: `float`
    max ::`obj`: `float`
    omega ::`obj`: `lal.CosmologicalParameter`
    pmax ::`obj`: `float`
    norm ::`obj`: `float`
    
    Returns
    ===========
    z ::`obj`: `float`
    """
    while True:
        test = pmax * np.random.uniform(0,1)
        z = np.random.uniform(min,max)
        prob = lal.RateWeightedUniformComovingVolumeDensity(z,p)/norm
        if (test < prob): break
    return z
    
class EMRIDistribution(object):
    def __init__(self,
                 redshift_min = 0.0,
                 redshift_max = 1.0,
                 ra_min  = 0.0,
                 ra_max  = 2.0*np.pi,
                 dec_min = -np.pi/2.0,
                 dec_max = np.pi/2.0,
                 *args, **kwargs):
        
        self.D0      = 1419.20
        self.delta_D0= 0.01
        self.V0      = 0.1290e4
        self.SNR0    = 100.0
        self.ra_min  = ra_min
        self.ra_max  = ra_max
        self.dec_min = dec_min
        self.dec_max = dec_max
        self.z_min   = redshift_min
        self.z_max   = redshift_max
        
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    
        try: self.h               = getattr(self,'h')
        except: self.h            = 0.7
        try: self.omega_m         = getattr(self,'omega_m')
        except: self.omega_m      = 0.3
        try: self.omega_lambda    = getattr(self,'omega_lambda')
        except: self.omega_lambda = 0.7
        try: self.w0              = getattr(self,'w0')
        except: self.w0           = -1.0
        try: self.w1              = getattr(self,'w1')
        except: self.w1           = 0.0
        try: self.w2              = getattr(self,'w2')
        except: self.w2           = 0.0
        self.omega                = lal.CreateCosmologicalParameters(self.h,self.omega_m,self.omega_lambda,self.w0,self.w1,self.w2)
        try: self.r0              = getattr(self,'r0')
        except: self.r0           = 1.0
        try: self.W               = getattr(self,'W')
        except: self.W            = 0.0
        try: self.Q               = getattr(self,'Q')
        except: self.Q            = 0.0
        try: self.R               = getattr(self,'R')
        except: self.R            = 0.0
        self.rate                 = lal.CreateCosmologicalRateParameters(self.r0, self.W, self.Q, self.R)
        
        # create a rate and cosmology parameter foir the calculations
        self.p = lal.CreateCosmologicalParametersAndRate()
        # set it up to the values we want
        self.p.omega.h  = self.h
        self.p.omega.om = self.omega_m
        self.p.omega.ol = self.omega_lambda
        self.p.omega.ok = 1.0 - self.omega_m - self.omega_lambda
        self.p.omega.w0 = self.w0
        self.p.omega.w1 = self.w1
        self.p.omega.w2 = self.w2
        
        self.p.rate.r0  = self.r0
        self.p.rate.R   = self.R
        self.p.rate.W   = self.W
        self.p.rate.Q   = self.Q

        # now we are ready to sample the EMRI according to the cosmology and rate that we specified
            # find the maximum of the probability for efficiency
        zt        = np.linspace(0,self.z_max,1000)
        self.norm = lal.IntegrateRateWeightedComovingVolumeDensity(self.p, self.z_max)
        self.dist = lambda z: lal.RateWeightedUniformComovingVolumeDensity(z,self.p)/self.norm
        self.pmax = np.max([self.dist(zi) for zi in zt])
            
        self.ra_pdf     = scipy.stats.uniform(loc = self.ra_min, scale = self.ra_max-self.ra_min)
        # dec distributed as cos(dec) in [-np.pi/2, np.pi/2] implies sin(dec) uniformly distributed in [-1,1]
        self.sindec_min = np.sin(self.dec_min)
        self.sindec_max = np.sin(self.dec_max)
        self.sindec_pdf = scipy.stats.uniform(loc = self.sindec_min, scale = self.sindec_max-self.sindec_min)
        
    def get_sample(self, *args, **kwargs):
        ra    = self.ra_pdf.rvs()
        dec   = np.arcsin(self.sindec_pdf.rvs())
        z     = redshift_rejection_sampling(self.z_min, self.z_max, self.p, self.pmax, self.norm)
        d     = lal.LuminosityDistance(self.omega,z)
        return z,d,ra,dec
    
    def get_bare_catalog(self, T = 10, *args, **kwargs):
        N = np.random.poisson(self.norm*T)
        print("expected number of sources = ",N)
        self.samps = np.array([self.get_sample() for _ in range(N)])
        return self.samps
    
    def get_catalog(self, T = 10, SNR_threshold = 20, *args, **kwargs):
        if hasattr(self,'samps'):
            print('we already have generated the catalog of GWs, dressing it up with SNRs')
        else:
            self.samps = self.get_bare_catalog(T = T, *args, **kwargs)
        snrs = self.compute_SNR(self.samps[:,1])
        e_d  = self.credible_distance_error(snrs)
        Vc   = self.credible_volume(snrs)
        self.catalog = np.column_stack((self.samps,snrs,e_d,Vc))
        self.catalog = self.find_redshift_limits()
        (idx,) = np.where(snrs > SNR_threshold)
        print("Effective number of sources = ",len(idx))
        self.catalog = self.catalog[idx,:]
        return self.catalog
    
    def compute_SNR(self, distance):
        return self.SNR0*self.D0/distance
    
    def credible_volume(self, SNR):
        # see https://arxiv.org/pdf/1801.08009.pdf
        return self.V0 * (self.SNR0/SNR)**(-6)

    def credible_distance_error(self, SNR):
        # see https://arxiv.org/pdf/1801.08009.pdf
        return self.delta_D0 * (SNR/self.SNR0)**(-1)

    def find_redshift_limits(self,
                             h_w  = (0.5,1.0),
                             om_w = (0.04, 1.0),
                             w0_w = (-1,-1),
                             w1_w = (0,0)):
        from cosmolisa.likelihood import find_redshift
        import cosmolisa.cosmology as cs
        
        def limits(O, Dmin, Dmax):
            return find_redshift(O,Dmin), find_redshift(O,Dmax)
        
        redshift_min = np.zeros(self.catalog.shape[0])
        redshift_max = np.zeros(self.catalog.shape[0])
        redshift_fiducial = np.zeros(self.catalog.shape[0])
        fiducial_O = cs.CosmologicalParameters(0.73,0.25,0.75,-1.0,0.0)
        for k in range(self.catalog.shape[0]):
            sys.stderr.write("finding redshift limits for event {0} out of {1}\r".format(k+1,self.catalog.shape[0]))
            z_min = np.zeros(100)
            z_max = np.zeros(100)
            for i in range(100):
                h = np.random.uniform(h_w[0],h_w[1])
                om = np.random.uniform(om_w[0],om_w[1])
                ol = 1.0-om
                w0 = np.random.uniform(w0_w[0],w0_w[1])
                w1 = np.random.uniform(w1_w[0],w1_w[1])
                O = cs.CosmologicalParameters(h,om,ol,w0,w1)
                z_min[i], z_max[i] = limits(O, self.catalog[k,1]*(np.maximum(0.0,1.0-3*self.catalog[k,5])), self.catalog[k,1]*(1.0+3*self.catalog[k,5]))
            redshift_min[k] = z_min.min()
            redshift_max[k] = z_max.max()
            redshift_fiducial[k] = find_redshift(fiducial_O, self.catalog[k,1])
        sys.stderr.write("\n")
        self.catalog = np.column_stack((self.catalog,redshift_fiducial,redshift_min,redshift_max))
        return self.catalog
        
    def save_catalog_ids(self, folder):
        """
        The file ID.dat has a single row containing:
        1-event ID
        2-Luminosity distance (Mpc)
        3-relative error on luminosity distance (usually few %)
        4-rough estimate of comoving volume of the errorbox
        5-observed redshift of the true host
        6-minimum redshift assuming the *true cosmology*
        7-maximum redshift assuming the *true cosmology*
        8-fiducial redshift (i.e. the redshift corresponding to the measured distance in the true cosmology)
        9-minimum redshift adding the cosmological prior
        10-maximum redshift adding the cosmological prior
        11-SNR
        12-SNR at the true distance
        """
        os.system("mkdir -p {0}".format(folder))
        for i in range(self.catalog.shape[0]):
            os.system("mkdir -p {0}".format(os.path.join(folder,"EVENT_1{))
            np.savetxt(os.path.join(
        return
if __name__=="__main__":
    h  = 0.7
    om = 0.25
    ol = 0.75
    r0 = 1e-10 # in Mpc^{-3}yr^{-1}
    W  = 0.0
    R  = 0.0
    Q  = 0.0
    C = EMRIDistribution(redshift_max  = 6.0, r0 = r0, W = W, R = R, Q = Q)
    z  = np.linspace(C.z_min,C.z_max,1000)
    import matplotlib.pyplot as plt
    plt.hist(C.get_bare_catalog(T = 10)[:,0],bins=100,density=True,alpha=0.5)
    plt.hist(C.get_catalog(T = 10, SNR_threshold = 20)[:,0],bins=100,density=True,alpha=0.5)
    plt.plot(z, [C.dist(zi) for zi in z])
    plt.show()

import numpy as np
import scipy.stats
import lal
import os
import sys
import cosmolisa.cosmology as cs
import matplotlib.pyplot as plt

"""
    file ID.dat:
    col 1: ID number
    col 2: luminosity distance (Mpc)
    col 3: relative error on dL (delta{dL}/dL)
    col 4: corresponding comoving volume of the errorcube (Mpc^3), fottitene
    col 5: redshift of the host (true cosmological, not apparent)
    col 6: z_min assuming true cosmology
    col 7: z_max assuming true cosmology
    col 8: z_fiducial from measured dL assuming true cosmology
    col 9: z_min assuming cosmology prior
    col 10: z_max assuming cosmology prior
    col 11: theta offset of the host compared to lisa best sky location (in sigmas, i.e. theta-theta_best/sigma{theta})
    col 12: same for phi
    col 13: same for dL
    col 14: theta host (rad)
    col 15: phi host (rad)
    col 16: dL host (Mpc)
    col 17: SNR
    col 18: altro SNR
    
    ERRORBOX.dat
    col 1: luminosity distance of true host(Mpc)
    col 2: cosmological redshift of candidate
    col 3: observed redshift of candidate (with peculiar velocities)
    col 4: log10 stellar mass in solar masses
    col 5: relative probability of candidate (based on sky loc)
    col 6: theta candidate (rad)
    col 7: theta host (rad)
    col 8: (theta_cand-theta_host)/dtheta
    col 9: phi candidate (rad)
    col 10: phi host (rad)
    col 11: (phi_cand-phi_host)/dphi
    col 12: dL candidate (rad)
    col 13: dL host (rad)
    col 14: (dL_cand-dL_host)/ddL
"""
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

def galaxy_redshift_rejection_sampling(min, max, O, pmax, norm):
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
        prob = O.ComovingVolumeElement(z)/norm
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
                 *args, 
                 **kwargs):
        # M101 event 57
        self.D0       = 1748.50
        self.A0       = 0.000405736691211125 #in rads^2
        self.delta_D0 = 0.022
        self.V0       = 0.1358e5
        self.SNR0     = 87
        self.ra_min   = ra_min
        self.ra_max   = ra_max
        self.dec_min  = dec_min
        self.dec_max  = dec_max
        self.z_min    = redshift_min
        self.z_max    = redshift_max
        
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    
        try: self.h               = getattr(self,'h')
        except: self.h            = 0.73
        try: self.omega_m         = getattr(self,'omega_m')
        except: self.omega_m      = 0.25
        try: self.omega_lambda    = getattr(self,'omega_lambda')
        except: self.omega_lambda = 0.75
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
        
        self.fiducial_O = cs.CosmologicalParameters(self.h, self.omega_m, self.omega_lambda, self.w0, self.w1)
        # create a rate and cosmology parameter for the calculations
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
        
        self.galaxy_pmax = None
        self.galaxy_norm = None
        
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
    
    def compute_area(self, SNR):
        return self.A0 * (self.SNR0/SNR)**2
    
    def credible_volume(self, SNR):
        # see https://arxiv.org/pdf/1801.08009.pdf
        return self.V0 * (self.SNR0/SNR)**(6)

    def credible_distance_error(self, SNR):
        # see https://arxiv.org/pdf/1801.08009.pdf
        return self.delta_D0 * self.SNR0/SNR

    def find_redshift_limits(self,
                             h_w  = (0.5,1.0),
                             om_w = (0.04,0.5),
                             w0_w = (-1,-1),
                             w1_w = (0,0)):
        from cosmolisa.likelihood import find_redshift
        
        def limits(O, Dmin, Dmax):
            return find_redshift(O,Dmin), find_redshift(O,Dmax)
        
        redshift_min = np.zeros(self.catalog.shape[0])
        redshift_max = np.zeros(self.catalog.shape[0])
        redshift_fiducial = np.zeros(self.catalog.shape[0])
        
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
            redshift_fiducial[k] = find_redshift(self.fiducial_O, self.catalog[k,1])
        sys.stderr.write("\n")
        self.catalog = np.column_stack((self.catalog,redshift_fiducial,redshift_min,redshift_max))
        return self.catalog
    
    def generate_galaxies(self, i):
        self.n0 = 0.66 # Mpc^{-3}. Increase it to augment the # of possible hosts per event 
        if self.galaxy_norm is None:
            self.galaxy_norm = self.fiducial_O.ComovingVolume(self.z_max)
        if self.galaxy_pmax is None:
            zt    = np.linspace(0,self.z_max,1000)
            self.galaxy_pmax  = np.max([self.fiducial_O.ComovingVolumeElement(zi)/self.galaxy_norm for zi in zt])
        Vc = self.catalog[i,6]
        D  = self.catalog[i,1]
        dD = D*self.catalog[i,5]
        A  = self.compute_area(self.catalog[i,4])
        N_gal = np.random.poisson(A/(4.0*np.pi)*Vc*self.n0)
#        print("D = ",D,"dD = ",dD,"Vc = ",Vc, "A = ",A, "N = ", N_gal)
#        if N_gal > 10000:
#            return 0,0,0
        z_cosmo = [galaxy_redshift_rejection_sampling(self.catalog[i,8], self.catalog[i,9], self.fiducial_O, self.galaxy_pmax, self.galaxy_norm) for _ in range(N_gal-1)]
#        z_cosmo = []
        z_cosmo.append(self.catalog[i,0])
        z_cosmo = np.array(z_cosmo)
        z_obs   = z_cosmo #+ np.random.normal(0.0, 0.0015, size = z_cosmo.shape[0])
#        logM    = np.random.uniform(10, 13, size = z_cosmo.shape[0])
        dz      = np.ones(N_gal)*0.0015
        W       = np.random.uniform(0.0, 1.0, size = z_cosmo.shape[0])
        W      /= W.sum()
        return z_cosmo, z_obs, W
#        import matplotlib.pyplot as plt
#        plt.hist(z_cosmo, bins=np.linspace(self.catalog[i,8], self.catalog[i,9],100), density=True, alpha=0.5,facecolor='red')
#        plt.hist(z_obs, bins=np.linspace(self.catalog[i,8], self.catalog[i,9],100), density=True, alpha=0.5,facecolor='blue')
#        plt.axvline(self.catalog[i,0],ls='dashed')
#        plt.axvline(self.catalog[i,8],color='r')
#        plt.axvline(self.catalog[i,9],color='r')
#        plt.show()
        
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
            f = os.path.join(folder,"EVENT_1{:03d}".format(i+1))
            os.system("mkdir -p {0}".format(f))
            np.savetxt(os.path.join(f,"ID.dat"),np.column_stack((i+1,
                                                                self.catalog[i,1],
                                                                self.catalog[i,5],
                                                                self.catalog[i,6],
                                                                0.0,
                                                                0.0,
                                                                0.0,
                                                                self.catalog[i,0],
                                                                self.catalog[i,8],
                                                                self.catalog[i,9],
                                                                0,
                                                                0,
                                                                0,
                                                                0,
                                                                0,
                                                                0,
                                                                self.catalog[i,4],
                                                                self.catalog[i,4])),
                                                                fmt = '%d %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f',
                                                                delimiter =' ')
            z_cosmo,z_obs, W = self.generate_galaxies(i)
            """
            The file ERRORBOX.dat has all the info you need to run the inference code. Each row is a possible host within the errorbox. Columns are:
            1-best luminosity distance measured by LISA
            2-redshift of the host candidate (without peculiar velocity)
            3-redshift of the host candidate (with peculiar velocity)
            4-log_10 of the host candidate mass in solar masses
            5-probability of the host according to the multivariate gaussian including the prior on cosmology (all rows add to 1)
            6-theta of the host candidate
            7-best theta measured by LISA
            8-difference between the above two in units of LISA theta error
            9-phi of the host candidate
            10-best phi measured by LISA
            11-difference between the above two in units of LISA phi error
            12-luminosity distance of the host candidate (in the Millennium cosmology)
            13-best Dl measured by LISA
            14-difference between the above two in units of LISA Dl error
            """
            if np.all(z_cosmo) != 0 and np.all(z_obs) !=0 and np.all(W) !=0:
                N = len(z_cosmo)
                print("EVENT_1{:03d} at redshift {} has {} hosts".format(i+1,self.catalog[i,0],N))
                np.savetxt(os.path.join(f,"ERRORBOX.dat"),np.column_stack((self.catalog[i,1]*np.ones(N),
                                                                           z_cosmo,
                                                                           z_obs,
                                                                           np.zeros(N),
                                                                           W,
                                                                           np.zeros(N),
                                                                           np.zeros(N),
                                                                           np.zeros(N),
                                                                           np.zeros(N),
                                                                           np.zeros(N),
                                                                           np.zeros(N),
                                                                           np.zeros(N),
                                                                           np.zeros(N),
                                                                           np.zeros(N))))
            else:
                print("Too many hosts, EVENT_1{:03d} is unusable".format(i+1))
        return
        
        
        
if __name__=="__main__":
    np.random.seed(1)
    h  = 0.73 # 0.73
    om = 0.25 # 0.25
    ol = 1.0 - om
    w0 = -1.0 # -1.0
    w1 = 0.0 # 0.0
    r0 = 1e-10 # in Mpc^{-3}yr^{-1}
    W  = 0.0
    R  = 0.0
    Q  = 0.0
    # e(z) = r0*(1.0+W)*exp(Q*z)/(exp(R*z)+W)

    # EDITABLE
    redshift_max = 1.0
    catalog_name = "test_catalog_z_1_h_073_SNR_20"
    
    C = EMRIDistribution(redshift_max  = redshift_max, h = h, omega_m = om, omega_lambda = ol, w0 = w0, w1 = w1, r0 = r0, W = W, R = R, Q = Q)
    C.get_catalog(T = 10, SNR_threshold = 20)
    C.save_catalog_ids(catalog_name)
    z  = np.linspace(C.z_min,C.z_max,1000)
    plt.hist(C.catalog[:,0],bins=30,density=True,alpha=0.5)
    plt.plot(z,[C.dist(zi) for zi in z])
    os.system('mkdir -p Figs')
    plt.savefig('Figs/{}.png'.format(catalog_name), bbox_inches='tight')

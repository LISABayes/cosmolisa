import numpy as np
import scipy.stats
import lal
import os
import sys
import cosmolisa.cosmology as cs
import cosmolisa.likelihood as lk
import cosmolisa.galaxy as gal
import matplotlib.pyplot as plt
from optparse import OptionParser

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
def rejection_sampling(zmin, zmax, distribution, pmax, N):
    """
    Samples the cosmologically correct redshift
    distribution
    
    Parameters
    ===========
    min ::`obj`: `float`
    max ::`obj`: `float`
    distribution ::`obj`: `lambda`
    pmax ::`obj`: `float`
    
    Returns
    ===========
    z ::`obj`: `float`
    """
    i = 0
    redshifts = []
    while i < 10*N:
        test = np.log(pmax * np.random.uniform(0,1))
        z = np.random.uniform(zmin,zmax)
        prob = np.log(distribution(z))
        if (test < prob):
            redshifts.append(z)
            i += 1
    return np.random.choice(redshifts, size = N, replace = False)

def normal(x,m,s):
    u = (x-m)/s
    return np.exp(-u**2)/np.sqrt(2.0*np.pi*s**2)

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
        self.A0       = 0.000405736691211125 #in rads^2
        self.V0       = 0.1358e5
        self.SNR0     = 87.
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
        try: self.r0              = getattr(self,'r0')
        except: self.r0           = 1.0
        try: self.W               = getattr(self,'W')
        except: self.W            = 0.0
        try: self.Q               = getattr(self,'Q')
        except: self.Q            = 0.0
        try: self.R               = getattr(self,'R')
        except: self.R            = 0.0
        
        # galaxy population parameters
        
        try: self.phistar0        = getattr(self,'phistar0')
        except: self.phistar0     = 1e-2 # in galaxies per Mpc^{-3}
        try: self.logMstar0       = getattr(self,'logMstar0')
        except: self.logMstar0    = -20.7
        try: self.alpha0          = getattr(self,'alpha0')
        except: self.alpha0       = -1.23
        try: self.Mmin            = getattr(self,'Mmin')
        except: self.Mmin         = -25
        try: self.Mmax            = getattr(self,'Mmax')
        except: self.Mmax         = -15
        try: self.m_threshold     = getattr(self,'m_threshold')
        except: self.m_threshold  = 17.7
        
        self.fiducial_O = cs.CosmologicalParameters(self.h, self.omega_m, self.omega_lambda, self.w0, self.w1)
        
        # luminosity function evolution model
        try: self.slope_model_choice        = getattr(self,'slope_model_choice')
        except: self.slope_model_choice     = 0
        try: self.cutoff_model_choice       = getattr(self,'cutoff_model_choice')
        except: self.cutoff_model_choice    = 0
        try: self.density_model_choice      = getattr(self,'density_model_choice')
        except: self.density_model_choice   = 0

        if self.cutoff_model_choice == 1:
            try:
                self.logMstar_exponent = getattr(self,'logMstar_exponent')
            except:
                print("must define an exponent for the cutoff luminosity evolution")
                exit(-1)
        else:
            self.logMstar_exponent = 0

        if self.slope_model_choice == 1:
            try:
                self.logMstar_exponent = getattr(self,'alpha_exponent')
            except:
                print("must define an exponent for the slope evolution")
                exit(-1)
        else:
            self.alpha_exponent = 0

        if self.density_model_choice == 1:
            try:
                self.phistar_exponent = getattr(self,'phistar_exponent')
            except:
                print("must define an exponent for the density evolution")
                exit(-1)
        else:
            self.phistar_exponent = 0

        self.galaxy_distribution  = gal.GalaxyDistribution(self.fiducial_O,
                                                           self.phistar0,
                                                           self.phistar_exponent,
                                                           self.logMstar0,
                                                           self.logMstar_exponent,
                                                           self.alpha0,
                                                           self.alpha_exponent,
                                                           self.Mmin,
                                                           self.Mmax,
                                                           self.z_min,
                                                           self.z_max,
                                                           self.ra_min,
                                                           self.ra_max,
                                                           self.dec_min,
                                                           self.dec_max,
                                                           self.m_threshold,
                                                           self.slope_model_choice,
                                                           self.cutoff_model_choice,
                                                           self.density_model_choice)

        # now we are ready to sample the EMRI according to the cosmology and rate that we specified
        # find the maximum of the probability for efficiency
        zt        = np.linspace(0,self.z_max,1000)
        self.norm = lk.integrated_rate(self.r0, self.W, self.R, self.Q, self.fiducial_O, self.z_min, self.z_max)
        print(self.norm)
        self.rate = lambda z: cs.StarFormationDensity(z, self.r0, self.W, self.R, self.Q)*self.fiducial_O.UniformComovingVolumeDensity(z)
        self.dist = lambda z: cs.StarFormationDensity(z, self.r0, self.W, self.R, self.Q)*self.fiducial_O.UniformComovingVolumeDensity(z)/self.norm
        self.pmax = np.max([self.dist(zi) for zi in zt])
            
        self.ra_pdf     = scipy.stats.uniform(loc = self.ra_min, scale = self.ra_max-self.ra_min)
        # dec distributed as cos(dec) in [-np.pi/2, np.pi/2] implies sin(dec) uniformly distributed in [-1,1]
        self.sindec_min = np.sin(self.dec_min)
        self.sindec_max = np.sin(self.dec_max)
        self.sindec_pdf = scipy.stats.uniform(loc = self.sindec_min, scale = self.sindec_max-self.sindec_min)
        
        self.galaxy_pmax = None
        self.galaxy_norm = None
        
    def get_sample(self, N, *args, **kwargs):
        ra    = self.ra_pdf.rvs(size = N)
        dec   = np.arcsin(self.sindec_pdf.rvs(size = N))
        z     = np.array(rejection_sampling(self.z_min, self.z_max, self.dist, self.pmax, N))
        d     = np.array([self.fiducial_O.LuminosityDistance(zi) for zi in z])
        return np.column_stack((z,d,ra,dec))
    
    def get_bare_catalog(self, T = 10, *args, **kwargs):
        N = np.random.poisson(self.norm*T)
        print("expected number of sources = ",N)
        self.samps = self.get_sample(N)
        return self.samps
    
    def get_catalog(self, T = 10, SNR_threshold = 20, *args, **kwargs):
        if hasattr(self,'samps'):
            print('we already have generated the catalog of GWs, dressing it up with SNRs')
        else:
            self.samps = self.get_bare_catalog(T = T, *args, **kwargs)
        snrs = self.compute_SNR(self.samps[:,1])
        e_d  = self.credible_distance_error(snrs)
        Vc   = self.credible_volume(snrs)
        self.catalog = np.column_stack((self.samps,snrs,e_d/self.samps[:,1],Vc))
        self.catalog = self.find_redshift_limits()
        (idx,) = np.where(snrs > SNR_threshold)
#        idx = []
        self.p = lk.gw_selection_probability_sfr(1e-5,
                                   self.z_max,
                                   self.r0,
                                   self.W,
                                   self.R,
                                   self.Q,
                                   SNR_threshold,
                                   self.fiducial_O,
                                   self.norm)
                                   
                                   
        print("d threshold = ", lk.threshold_distance(SNR_threshold))
        print("Effective number of sources = ",len(idx))
        self.catalog = self.catalog[idx,:]
        return self.catalog
    
    def compute_SNR(self, distance):
        return np.array([lk.snr_vs_distance(d) for d in distance])
    
    def compute_area(self, SNR):
        return self.A0 * (self.SNR0/SNR)**2
    
    def credible_volume(self, SNR):
        # see https://arxiv.org/pdf/1801.08009.pdf
        return self.V0 * (self.SNR0/SNR)**(6)

    def credible_distance_error(self, SNR):
        # see https://arxiv.org/pdf/1801.08009.pdf
        return np.array([lk.distance_error_vs_snr(S) for S in SNR])

    def find_redshift_limits(self,
                             h_w  = (0.6,0.86),
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
    
    def get_z_limits(self,i):
        return self.catalog[i,8], self.catalog[i,9]

    def get_ra_limits(self, i, A):
        ra = self.catalog[i,2]
        return ra-np.sqrt(A/np.pi), ra+np.sqrt(A/np.pi)
    
    def get_dec_limits(self, i, A):
        dec = self.catalog[i,3]
        return dec-np.sqrt(A/np.pi), dec+np.sqrt(A/np.pi)
    
    def generate_galaxies(self, i):

        Vc = self.catalog[i,6]
        D  = self.catalog[i,1]
        dD = D*self.catalog[i,5]
        A  = self.compute_area(self.catalog[i,4])
        N_gal_tot = int(A/(4.0*np.pi)*self.galaxy_distribution.get_norm(zmin = self.catalog[i,8], zmax = self.catalog[i,9],selection = 0))
        N_gal = int(A/(4.0*np.pi)*self.galaxy_distribution.get_norm(zmin = self.catalog[i,8], zmax = self.catalog[i,9],selection = 1))
        print(i,": SNR =",self.catalog[i,4],"area(sr) =",A,"area(deg^2) =",A*(180./np.pi)**2,"Ntot =",N_gal_tot,"N_gal =",N_gal)
        if N_gal < 1:
            N_gal = 1
        
        zmin, zmax     = self.get_z_limits(i)
        ramin, ramax   = self.get_ra_limits(i, A)
        decmin, decmax = self.get_dec_limits(i, A)
        samps = self.galaxy_distribution.sample(zmin, zmax, ramin, ramax, decmin, decmax, N_gal, selection = 1)
        # append the true host
        true_host = self.galaxy_distribution.sample(zmin, zmax, ramin, ramax, decmin, decmax, 1, selection = 1)
        true_host[0,1] = self.catalog[i,0]
        true_host[0,2] = self.catalog[i,2]
        true_host[0,3] = self.catalog[i,3]
        idx = np.random.randint(N_gal)
        print(idx,samps.shape)
        samps[idx, :] = true_host
#        z_cosmo.append(self.catalog[i,0])
        z_cosmo = samps[:,1]
        Mags    = samps[:,0]
        z_obs   = z_cosmo + np.random.normal(0.0, 0.0015, size = z_cosmo.shape[0])
        ra      = samps[:,2]
        dec     = samps[:,3]
#        logM    = np.random.uniform(10, 13, size = z_cosmo.shape[0])
        dz      = np.ones(N_gal)*0.0015
        W       = normal(self.catalog[i,2],ra,np.sqrt(A/np.pi))*normal(self.catalog[i,3],dec,np.sqrt(A/np.pi))#np.random.uniform(0.0, 1.0, size = z_cosmo.shape[0])
        W      /= W.sum()
        return z_cosmo, z_obs, Mags, ra, dec, W
        
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
        
        
        """
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
                                                                self.catalog[i,3],
                                                                self.catalog[i,2],
                                                                self.catalog[i,1],
                                                                self.catalog[i,4],
                                                                self.catalog[i,4])),
                                                                fmt = '%d %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f',
                                                                delimiter =' ')
            try:
                z_cosmo, z_obs, Mags, ra, dec, W = self.generate_galaxies(i)
            except:
                pass
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
                                                                           Mags,
                                                                           W,
                                                                           dec,
                                                                           np.zeros(N),
                                                                           np.zeros(N),
                                                                           ra,
                                                                           np.zeros(N),
                                                                           np.zeros(N),
                                                                           np.zeros(N),
                                                                           np.zeros(N),
                                                                           np.zeros(N))))
            else:
                print("Too many hosts (or zero), EVENT_1{:03d} is unusable".format(i+1))
        return
        
        
        
usage=""" %prog (options)"""

if __name__=='__main__':

    parser = OptionParser(usage)
    parser.add_option('--r0', default=1e-12, type='float', metavar='r0', help='local merger rate in Mpc^{-3}yr^{-1}')
    parser.add_option('--W', default=0.0, type='float', metavar='W', help='merger rate parameter W')
    parser.add_option('--R', default=0.0, type='float', metavar='R', help='merger rate parameter R')
    parser.add_option('--Q', default=0.0, type='float', metavar='Q', help='merger rate parameter Q')
    parser.add_option('--h', default=0.73, type='float', metavar='h', help='h')
    parser.add_option('--om', default=0.25, type='float', metavar='om', help='om')
    parser.add_option('--ol', default=0.75, type='float', metavar='ol', help='ol')
    parser.add_option('--w0', default=-1.0, type='float', metavar='w0', help='w0')
    parser.add_option('--w1', default=0.0, type='float', metavar='w1', help='w1')
    parser.add_option('--alpha', default=-1.23, type='float', metavar='alpha', help='alpha')
    parser.add_option('--logMstar', default=-20.7, type='float', metavar='logMstar', help='logMstar')
    parser.add_option('--phistar', default=1e-5, type='float', metavar='phistar', help='phistar')
    parser.add_option('--Mmin', default=-25.0, type='float', metavar='Mmin', help='Mmin')
    parser.add_option('--Mmax', default=-15.0, type='float', metavar='Mmax', help='Mmax')
    parser.add_option('--m_threshold', default=20.0, type='float', metavar='m_threshold', help='m_threshold')
    parser.add_option('--zmax', default=1.0, type='float', metavar='zmax', help='maximum redshift')
    parser.add_option('--seed', default=1, type='float', metavar='seed', help='seed initialisation')
    parser.add_option('--snrmin', default=20, type='float', metavar='snrmin', help='snr threshold')
    parser.add_option('--T', default=10, type='float', metavar='T', help='observation time')
    parser.add_option('--output', default = './', metavar='DIR', help='Directory for output.')
    (opts,args)=parser.parse_args()
    
    np.random.seed(opts.seed)
    h  = opts.h # 0.73
    om = opts.om # 0.25
    ol = opts.ol
    w0 = opts.w0
    w1 = opts.w1 # 0.0
    r0 = opts.r0 # in Mpc^{-3}yr^{-1}
    W  = opts.W
    R  = opts.R
    Q  = opts.Q
    T  = opts.T
    logMstar = opts.logMstar
    alpha = opts.alpha
    phistar = opts.phistar
    snr_th = opts.snrmin
    Mmin   = opts.Mmin
    Mmax   = opts.Mmax
    m_threshold = opts.m_threshold
    # e(z) = r0*(1.0+W)*exp(Q*z)/(exp(R*z)+W)

    # EDITABLE
    redshift_max = opts.zmax
    catalog_name = opts.output
    os.system('mkdir -p {0}'.format(catalog_name))
    C = EMRIDistribution(redshift_max  = redshift_max,
                         h = h,
                         omega_m = om,
                         omega_lambda = ol,
                         w0 = w0,
                         w1 = w1,
                         r0 = r0,
                         W = W,
                         R = R,
                         Q = Q,
                         Mmin = Mmin,
                         Mmax = Mmax,
                         alpha0 = alpha,
                         logMstar0 = logMstar,
                         phistar0 = phistar,
                         m_threshold = m_threshold)
                         
    C.get_catalog(T = T, SNR_threshold = snr_th)
    C.save_catalog_ids(catalog_name)
    print('fraction of detected events = ',C.p)
    z  = np.linspace(C.z_min,C.z_max,1000)
    
    os.system('mkdir -p {0}'.format(os.path.join(catalog_name,'figures')))
    
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    ax.hist(C.catalog[:,0],bins=np.linspace(0.0,C.z_max,100),alpha=0.5,cumulative=True)
    ax.hist(C.samps[:,0],bins=np.linspace(0.0,C.z_max,100),alpha=0.5,facecolor='r',cumulative=True)
    ax.plot(z,T*C.norm*np.cumsum(np.array([C.dist(zi) for zi in z]))*np.diff(z)[0], lw=1.5, color='k')
    ax.set_xlabel('redshift z')
    ax.set_ylabel('$R(z_{max})\cdot T\cdot p(z|\Lambda,\Omega,I)$')
    plt.savefig('{0}/{1}.pdf'.format(os.path.join(catalog_name,'figures'),'number_of_events'), bbox_inches='tight')
    
    plt.clf()
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    ax.hist(C.catalog[:,0],bins=np.linspace(0.0,C.z_max,100),alpha=0.5,density=True)
    ax.hist(C.samps[:,0],bins=np.linspace(0.0,C.z_max,100),alpha=0.5,facecolor='r',density=True)
    ax.plot(z,np.array([C.dist(zi) for zi in z]), lw=1.5, color='k')
    ax.set_xlabel('redshift z')
    ax.set_ylabel('$p(z|\Lambda,\Omega,I)$')
    plt.savefig('{0}/{1}.pdf'.format(os.path.join(catalog_name,'figures'),'distribution'), bbox_inches='tight')

    plt.clf()
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    ax.plot(z, [cs.StarFormationDensity(zi, C.r0, C.W, C.R, C.Q)/1e-12 for zi in z])
    ax.set_xlabel('redshift z')
    ax.set_ylabel('merger rate [10$^{-12}$ Mpc$^{-3}$ yr$^{-1}$]')
    plt.savefig('{0}/{1}.pdf'.format(os.path.join(catalog_name,'figures'),'merger_rate'), bbox_inches='tight')

    plt.clf()
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    ax.hist(C.catalog[:,4],bins=32,alpha=0.5)
    ax.set_xlabel('SNR')
    ax.set_ylabel('number of events')
    plt.savefig('{0}/{1}.pdf'.format(os.path.join(catalog_name,'figures'),'snr'), bbox_inches='tight')

    plt.clf()
    fig = plt.figure(1)
    ax  = fig.add_subplot(111, projection='aitoff')
    Sc = ax.scatter(np.pi-C.catalog[:,2],C.catalog[:,3],c=C.catalog[:,0])
    plt.grid()
    plt.colorbar(Sc)
    plt.savefig('{0}/{1}.pdf'.format(os.path.join(catalog_name,'figures'),'sky_position'), bbox_inches='tight')

    
    galaxies_z = []
    galaxies_m = []
    
    import matplotlib.cm as cm
    colors = iter(cm.rainbow(np.linspace(0, 1, C.catalog.shape[0])))
    
    plt.clf()
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    ax1  = fig1.add_subplot(111)
    ax2  = fig2.add_subplot(111, projection='aitoff')
    iterator = range(0,C.catalog.shape[0])
    
    for i in iterator:
        try:
            z, m, W, dec, ra = np.loadtxt(catalog_name+"/EVENT_1{:03d}/ERRORBOX.dat".format(i+1), usecols = (1,3,4,5,8), unpack=True)
            c = next(colors)
            ax1.scatter(z, m, color = c, alpha=0.5,label ="EVENT_1{:03d}".format(i+1))
            ax1.axvline(C.catalog[i,0], color = c, linestyle='dotted')
            ax2.scatter(np.pi-ra, dec, color = c, alpha=0.5)
            ax2.plot(np.pi-C.catalog[:,2],C.catalog[:,3], color=c, marker='+', linestyle='None', label ="EVENT_1{:03d}".format(i+1))
            galaxies_z.extend(z)
            galaxies_m.extend(m)
        except:
            pass
    
    ax1.set_xlabel('redshift')
    ax1.set_ylabel('magnitude')
    fig1.legend(ncol=4,bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    fig2.legend(ncol=4,bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    fig1.savefig('{0}/{1}.pdf'.format(os.path.join(catalog_name,'figures'),'galaxies'), bbox_inches='tight')
    fig2.savefig('{0}/{1}.pdf'.format(os.path.join(catalog_name,'figures'),'skyposition_galaxies'), bbox_inches='tight')

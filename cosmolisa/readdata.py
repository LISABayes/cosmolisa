import numpy as np
import sys
import os

class Galaxy(object):
    """
    Galaxy class:
    initialise a galaxy defined by its redshift, redshift error,
    weight determined by its angular position
    relative to the LISA posterior and magnitude
    """
    def __init__(self, redshift, dredshift, weight, dl, magnitude):
        
        self.redshift   = redshift
        self.dredshift  = dredshift
        self.weight     = weight
        self.dl         = dl
        self.magnitude  = magnitude

class Event(object):
    """
    Event class:
    initialise a GW event based on its distance and potential
    galaxy hosts
    """
    def __init__(self,
                 ID,
                 dl,
                 sigma,
                 sigma_gw_theta,
                 sigma_gw_phi,
                 redshifts,
                 dredshifts,
                 weights,
                 magnitudes,
                 zmin,
                 zmax,
                 snr,
                 z_true,
                 dl_host,
                 snr_threshold = 8.0,
                 VC = None):

        self.potential_galaxy_hosts = [Galaxy(r,dr,w,dl,m) for r,dr,w,dl,m in zip(redshifts,dredshifts,weights,dl_host,magnitudes)]
        self.n_hosts                = len(self.potential_galaxy_hosts)
        self.ID                     = ID
        self.dl                     = dl
        self.sigma                  = sigma
        self.sigma_gw_theta         = sigma_gw_theta
        self.sigma_gw_phi           = sigma_gw_phi
        self.dmax                   = (self.dl+3.0*self.sigma)
        self.dmin                   = (self.dl-3.0*self.sigma)
        self.zmin                   = zmin
        self.zmax                   = zmax
        self.snr                    = snr
        self.VC                     = VC
        self.z_true                 = z_true
        if self.dmin < 0.0: self.dmin = 0.0

def read_MBH_event(input_folder, event_number, max_distance = None, max_hosts = None):
    
    all_files   = os.listdir(input_folder)
    events_list = [f for f in all_files if 'EVENT' in f or 'event' in f]
    
    if event_number is None:
        
        events = []
        
        for ev in events_list:
            
            event_file          = open(input_folder+"/"+ev+"/ID.dat","r")
            event_id,dl,sigma   = event_file.readline().split(None)
            ID                  = np.int(event_id)
            dl                  = np.float64(dl)
            sigma               = np.float64(sigma)*dl
            event_file.close()
            
            try:
                redshifts,d_redshifts   = np.loadtxt(input_folder+"/"+ev+"/ERRORBOX.dat",unpack=True)
                redshifts               = np.atleast_1d(redshifts)
                d_redshifts             = np.atleast_1d(d_redshifts)
                weights                 = np.ones(len(redshifts))
                magnitudes              = np.ones(len(redshifts))
                zmin                    = np.maximum(redshifts - 5.0*d_redshifts, 0.0)
                zmax                    = redshifts + 5.0*d_redshifts
                events.append(Event(ID,
                                    dl,
                                    sigma,
                                    1,
                                    1,
                                    redshifts,
                                    d_redshifts,
                                    weights,
                                    magnitudes,
                                    zmin,
                                    zmax,
                                    -1,
                                    -1,
                                    [0]))
                sys.stderr.write("Selecting event %s at a distance %s (error %s), hosts %d\n"%(event_id,dl,sigma,len(redshifts)))
            except:
                if (TypeError, NameError): raise
                else: sys.stderr.write("Event %s at a distance %s (error %s) has no hosts, skipping\n"%(event_id,dl,sigma))

        if max_distance is not None:
            distance_limited_events = [e for e in events if e.dl < max_distance]
        else:
            distance_limited_events = [e for e in events]

        if max_hosts is not None:
            analysis_events = [e for e in distance_limited_events if len(e.n_hosts) < max_hosts]
        else:
            analysis_events = [e for e in distance_limited_events]

    else:
        event_file          = open(input_folder+"/"+events_list[event_number]+"/ID.dat","r")
        event_id,dl,sigma   = event_file.readline().split(None)
        ID                  = np.int(event_id)
        dl                  = np.float64(dl)
        sigma               = np.float64(sigma)*dl
        event_file.close()
        try:
            redshifts,d_redshifts   = np.loadtxt(input_folder+"/"+events_list[event_number]+"/ERRORBOX.dat",unpack=True)
            redshifts               = np.atleast_1d(redshifts)
            d_redshifts             = np.atleast_1d(d_redshifts)
            weights                 = np.atleast_1d(len(redshifts))
            magnitudes              = np.atleast_1d(len(redshifts))
            zmin                    = np.maximum(redshifts - 10.0*d_redshifts, 0.0)
            zmax                    = redshifts + 10.0*d_redshifts
            analysis_events         = [Event(ID,
                                            dl,
                                            sigma,
                                            1,
                                            1,
                                            redshifts,
                                            d_redshifts,
                                            weights,
                                            magnitudes,
                                            zmin,
                                            zmax,
                                            -1,
                                            -1,
                                            [0])]
            sys.stderr.write("Selecting event %s at a distance %s (error %s), hosts %d\n"%(event_id,dl,sigma,len(redshifts)))
        except:
            sys.stderr.write("Event %s at a distance %s (error %s) has no hosts, skipping\n"%(event_id,dl,sigma))
            exit()

    sys.stderr.write("Read %d events\n"%len(analysis_events))
    return analysis_events

def read_EMRI_event(input_folder, event_number, max_hosts=None, one_host_selection=0, z_selection=None, snr_selection=None, snr_threshold=0.0, event_ID_list=None, zhorizon=None):
    """
    The file ID.dat has a single row containing:
    1-event ID
    2-Luminosity distance dL (Mpc)
    3-relative error on luminosity distance delta{dL}/dL (usually few %)
    4-rough estimate of comoving volume of the errorbox
    5-observed redshift of the true host (true cosmological, not apparent)
    6-minimum redshift assuming the *true cosmology*
    7-maximum redshift assuming the *true cosmology*
    8-fiducial redshift (i.e. the redshift corresponding to the measured distance in the true cosmology)
    9-minimum redshift adding the cosmological prior
    10-maximum redshift adding the cosmological prior
    11-theta offset of the host compared to LISA best sky location (in sigmas, i.e. theta-theta_best/sigma{theta})
    12-same for phi
    13-same for dL
    14-theta host (rad)
    15-phi host (rad)
    16-dL host (Mpc)
    17-SNR
    18-SNR at the true distance
    
    The file ERRORBOX.dat has all the info you need to run the inference code. Each row is a possible host within the errorbox. Columns are:
    1-best luminosity distance measured by LISA (the same as col 1 in ID.dat)
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
    13-best dL measured by LISA
    14-difference between the above two in units of LISA Dl error
    """
    all_files   = os.listdir(input_folder)
    print("Reading {}".format(input_folder))
    events_list = [f for f in all_files if 'EVENT' in f or 'event' in f]
    pv = 0.0015 # redshift error associated to peculiar velocity value (https://arxiv.org/abs/1703.01300)

    if (event_number is None):

        events = []
        for ev in events_list:
            # Some test catalogs have an additional unused column, so a try/except is needed
            try:
                event_file      = open(input_folder+"/"+ev+"/ID.dat","r")
                # 1     ,2 ,3    ,4 ,5              ,6        ,7        ,8     ,9   ,10  , , , , , , ,17 ,18
                event_id,dl,sigma,Vc,z_observed_true,zmin_true,zmax_true,z_true,zmin,zmax,_,_,_,_,_,_,snr,snr_true = event_file.readline().split(None)
            except (ValueError):
                event_file      = open(input_folder+"/"+ev+"/ID.dat","r")
                # 1     ,2 ,3    ,4 ,5              ,6        ,7        ,8     ,9   ,10  , , , , , , ,17 ,18      ,19
                event_id,dl,sigma,Vc,z_observed_true,zmin_true,zmax_true,z_true,zmin,zmax,_,_,_,_,_,_,snr,snr_true,_ = event_file.readline().split(None)
            ID              = np.int(event_id)
            dl              = np.float64(dl)
            sigma           = np.float64(sigma)*dl
            zmin            = np.float64(zmin)
            zmax            = np.float64(zmax)
            snr             = np.float64(snr)
            VC              = np.float64(Vc)
            z_true          = np.float64(z_true)
            event_file.close()
            try:
                try:
                    # 1    ,2      ,3  ,4   ,5      ,6    ,7         ,8     ,9  ,10      ,11  ,12     ,13       ,14
                    best_dl,zcosmo,zobs,logM,weights,theta,best_theta,dtheta,phi,best_phi,dphi,dl_host,best_dl_2,deltadl = np.loadtxt(input_folder+"/"+ev+"/ERRORBOX.dat",unpack=True)
                except:
                    # 1    ,2      ,3  ,4   ,5      ,6    ,7         ,8     ,9  ,10      ,11  ,12     ,13       ,14, 15
                    best_dl,zcosmo,zobs,logM,weights,theta,best_theta,dtheta,phi,best_phi,dphi,dl_host,best_dl_2,deltadl,_ = np.loadtxt(input_folder+"/"+ev+"/ERRORBOX.dat",unpack=True)
                redshifts       = np.atleast_1d(zobs)
                d_redshifts     = np.ones(len(redshifts))*pv
                weights         = np.atleast_1d(weights)
                magnitudes      = np.atleast_1d(magnitudes)
                sigma_gw_theta  = np.mean((theta-best_theta)/dtheta)
                sigma_gw_phi    = np.mean((phi-best_phi)/dphi)
                if not (isinstance(dl_host, type(redshifts))):
                    dl_host = np.atleast_1d(dl_host)
                events.append(Event(ID,
                                    dl,
                                    sigma,
                                    sigma_gw_theta,
                                    sigma_gw_phi,
                                    redshifts,
                                    d_redshifts,
                                    weights,
                                    magnitudes,
                                    zmin,
                                    zmax,
                                    snr,
                                    z_true,
                                    dl_host,
                                    VC = VC))
                print("Reading event %s at a distance %s (error %s), hosts %d"%(event_id,dl,sigma,len(redshifts)))
            except:
                print("Event %s at a distance %s (error %s) has no hosts, skipping\n"%(event_id,dl,sigma))

        if (snr_selection is not None):
            new_list = sorted(events, key=lambda x: getattr(x, 'snr'))
            if (snr_selection > 0):
                events = new_list[:snr_selection]
            elif (snr_selection < 0):
                events = new_list[snr_selection:]
            print("\nSelected {} events from SNR={} to SNR={}:".format(len(events), events[0].snr, events[abs(snr_selection)-1].snr))            
            for e in events:
                print("ID: {}  |  SNR: {}".format(str(e.ID).ljust(3), str(e.snr).ljust(9)))
            # CHECK BLOCK - Split in redshift
            # events_selected = []
            # for e in events:
            #     print(e.ID)
            #     if (e.z_true > 0.3):
            #         events_selected.append(e) 
            #     else:
            #         print("Event {} removed (z={}).".format(e.ID, e.z_true))
            # events = events_selected
            # print("\nAfter z-selection (z>0.3), will run a joint analysis on {} events.\n".format(len(events)))
            # print("Selected {} events from snr={} to snr={}.".format(len(events), events[0].snr, events[len(events)-1].snr))
            # CHECK BLOCK - If EMRIs at z>0.3 are selected, remove host galaxies at z<0.3 
            # for e in events:
            #     print("\nEvent", e.ID)
            #     galaxy_selected = []
            #     print("Original galaxy hosts:",len(e.potential_galaxy_hosts))
            #     for gal in e.potential_galaxy_hosts: 
            #         if (gal.redshift > 0.3):
            #             galaxy_selected.append(gal) 
            #         else:
            #             print("Galaxy host {} removed (z={}).".format(gal, gal.redshift))
            #     e.potential_galaxy_hosts = galaxy_selected
            #     print("Selected galaxy hosts (z > 0.3):", len(e.potential_galaxy_hosts))
            # print("Selected {} events from snr={} to snr={}.".format(len(events), events[0].snr, events[len(events)-1].snr))

        if (z_selection is not None):
            new_list = sorted(events, key=lambda x: getattr(x, 'z_true'))
            if (z_selection > 0):
                events = new_list[:z_selection]
            elif (z_selection < 0):
                events = new_list[z_selection:]
            print("\nSelected {} events from z={} to z={}:".format(len(events), events[0].z_true, events[abs(z_selection)-1].z_true))
            for e in events:
                print("ID: {}  |  z_true: {}".format(str(e.ID).ljust(3), str(e.z_true).ljust(7)))

        if (zhorizon is not None):
            events = [e for e in events if e.z_true < zhorizon]
            events = sorted(events, key=lambda x: getattr(x, 'z_true'))
            print("\nSelected {} events from z={} to z={} (zhorizon={}):".format(len(events), events[0].z_true, events[len(events)-1].z_true, zhorizon))
            for e in events:
                print("ID: {}  |  z_true: {}".format(str(e.ID).ljust(3), str(e.z_true).ljust(7)))           

        if (max_hosts is not None):
            events = [e for e in events if e.n_hosts < max_hosts]
            events = sorted(events, key=lambda x: getattr(x, 'n_hosts'))
            print("\nSelected {} events having hosts from n={} to n={} (max hosts imposed={}):".format(len(events), events[0].n_hosts, events[len(events)-1].n_hosts, max_hosts))
            for e in events:
                print("ID: {}  |  n_hosts: {}".format(str(e.ID).ljust(3), str(e.n_hosts).ljust(7)))

        if (snr_threshold is not 0.0):
            if snr_threshold > 0:
                events = [e for e in events if e.snr > snr_threshold]
            else:
                events = [e for e in events if e.snr < abs(snr_threshold)]
            events = sorted(events, key=lambda x: getattr(x, 'snr'))
            print("\nSelected {} events from SNR={} to SNR={} (SNR_threshold={}):".format(len(events), events[0].snr, events[len(events)-1].snr, snr_threshold))
            for e in events:
                print("ID: {}  |  SNR: {}".format(str(e.ID).ljust(3), str(e.snr).ljust(7)))     


        if (event_ID_list is not None):
            event_list = []
            ID_list = event_ID_list.split(',')
            events = [e for e in events if str(e.ID) in ID_list]

        if(one_host_selection):
            for e in events:
                z_differences = []
                for gal in e.potential_galaxy_hosts:
                    z_diff = abs(e.z_true - gal.redshift)
                    z_differences.append(z_diff)
                    if (z_diff == min(z_differences)):
                        selected_gal = gal 
                e.potential_galaxy_hosts = [selected_gal]
            print("\nUsing only the nearest host to the EMRI:")
            events = sorted(events, key=lambda x: getattr(x, 'ID'))
            for e in events:
                print("ID: {}  |  SNR: {}  |  dl: {} Mpc  |  z_true: {} |  z_host: {} |  hosts: {}".format(
                str(e.ID).ljust(3), str(e.snr).ljust(9), str(e.dl).ljust(7), str(e.z_true).ljust(7), 
                str(e.potential_galaxy_hosts[0].redshift).ljust(7), str(len(e.potential_galaxy_hosts)).ljust(4)))

        analysis_events = events

    else:
        events_list.sort()
        analysis_events = []
        event_file      = open(input_folder+"/"+events_list[event_number]+"/ID.dat","r")
        event_id,dl,sigma,Vc,z_observed_true,zmin_true,zmax_true,z_true,zmin,zmax,_,_,_,_,_,_,snr,snr_true = event_file.readline().split(None)
        ID              = np.int(event_id)
        dl              = np.float64(dl)
        sigma           = np.float64(sigma)*dl
        zmin            = np.float64(zmin)
        zmax            = np.float64(zmax)
        snr             = np.float64(snr)
        VC              = np.float64(Vc)
        z_true          = np.float64(z_true)
        event_file.close()
#        try:
        best_dl,zcosmo,zobs,magnitudes,weights,theta,best_theta,dtheta,phi,best_phi,dphi,dl_host,best_dl_2,deltadl = np.loadtxt(input_folder+"/"+events_list[event_number]+"/ERRORBOX.dat",unpack=True)
        redshifts = np.atleast_1d(zobs)
        d_redshifts     = np.ones(len(redshifts))*pv
        weights         = np.atleast_1d(weights)
        magnitudes      = np.atleast_1d(magnitudes)
        analysis_events.append(Event(ID,dl,sigma,0.0,0.0,redshifts,d_redshifts,weights,magnitudes,zmin,zmax,snr,z_true,dl_host,VC = VC))
        sys.stderr.write("Selecting event %s at a distance %s (error %s), hosts %d\n"%(event_id,dl,sigma,len(redshifts)))
#        except:
#            sys.stderr.write("Event %s at a distance %s (error %s) has no hosts, skipping\n"%(event_id,dl,sigma))

    return analysis_events

def read_DEBUG_event(datafile, *args, **kwargs):
    """
    Parameters:
    ==============
    datafile: file containing the data for the hosts
    The columns must be:
    1-ID sorgente
    2-Dl LISA best measurement
    3-Delta{Dl}
    4-redshift dell'host
    5-unused
    6-weight (set to 1)
    7-z minimo considerando variazione cosmologia, delta{Dl} e peculiar vel
    8-z max considerando variazione cosmologia, delta{Dl} e peculiar vel
    9-z corrispondente al Dl misurato da LISA
    """
    event_id, dl, sigma, redshift, _, weights, zmin, zmax, redshift_inv_d   = np.loadtxt(datafile, unpack = True)
        
    events = []
        
    for i in range(len(event_id)):
        
        events.append(Event(event_id[i],dl[i],sigma[i],[redshift[i]],[0.0015], [weights[i]],zmin[i],zmax[i]))

    events = events[:50]
    sys.stderr.write("Selected %d events\n"%len(events))
    return events

def read_event(event_class,*args,**kwargs):
    if   (event_class == "MBH"):   return read_MBH_event(*args, **kwargs)
    elif (event_class == "EMRI"):  return read_EMRI_event(*args, **kwargs)
    elif (event_class == "sBH"):   return read_EMRI_event(*args, **kwargs)
    elif (event_class == "DEBUG"): return read_DEBUG_event(*args, **kwargs)
    else:
        print("I do not know the class %s, exiting\n"%event_class)
        exit(-1)

if __name__=="__main__":
    input_folder = '/Users/wdp/repositories/LISA/LISA_BHB/errorbox_data/EMRI_data/EMRI_M1_GAUSS'
    event_number = None
    e = read_event("EMRI",input_folder, event_number)
    print(e)

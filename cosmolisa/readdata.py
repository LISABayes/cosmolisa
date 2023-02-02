import numpy as np
import sys
import os

class Galaxy:
    """Galaxy class:
    Initialise a galaxy defined by its redshift, redshift error,
    weight determined by its angular position relative to
    the detector posterior, and magnitude (if available).
    """
    def __init__(self, redshift, dredshift, weight, magnitude):
        
        self.redshift = redshift
        self.dredshift = dredshift
        self.weight = weight
        self.magnitude = magnitude

class Event:
    """Event class:
    Initialise a GW event based on its properties and 
    potential galaxy hosts.
    """
    def __init__(self,
                 ID,
                 dl,
                 sigmadl,
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
                 z_cosmo_true_host,
                 dl_host,
                 VC = None):

        self.potential_galaxy_hosts = [Galaxy(r, dr, w, m)
            for r, dr, w, m in zip(redshifts, dredshifts, weights,
            magnitudes)]
        self.n_hosts = len(self.potential_galaxy_hosts)
        self.ID = ID
        self.dl = dl
        self.sigmadl = sigmadl
        self.sigma_gw_theta = sigma_gw_theta
        self.sigma_gw_phi = sigma_gw_phi
        self.dmax = (self.dl + 3.0*self.sigmadl)
        self.dmin = (self.dl - 3.0*self.sigmadl)
        self.zmin = zmin
        self.zmax = zmax
        self.snr = snr
        self.VC = VC
        self.z_true = z_true
        self.z_cosmo_true_host = z_cosmo_true_host
        if (self.dmin < 0.0): self.dmin = 0.0

def read_MBHB_event(input_folder, event_number=None):
    """Read MBHB data to be passed to CosmologicalModel class.
    #########################################################
    If data is stored in two files:
    The file ID.dat (no header) has a single row containing:
    1-event ID
    2-luminosity distance dL (Mpc) (scattered)
    3-relative error on luminosity distance delta{dL}/dL 
        (not including propagated z error)
    The file ERRORBOX.dat (no header) contains:
    1-event redshift 
    2-absolute redshift error
    #########################################################
    If data is stored in a single file:
    The file ID.dat (with header) contains:
    1-z_true: the true binary redshift
    2-z_shifted: shifted redshift because of the error in the 
        EM measurement
    3-error_z: redshift error	
    4-d_LGpc: luminosity distance dL in Gpc
    5-d_L_shiftedGpc: shifted dL because of noise,
        lensing, and peculiar velocity errors
    6-sigma_dl_posterior1sigmaGpc: error from the posterior
        distribution
    7-sigma_dl_fisherGpc: error coming from the Fisher
    8-sigma_dl_pvGpc: error coming from the peculiar velocity
    9-sigma_dl_lensingGpc: error coming from lensing
    10-sigma_dl_combinedGpc: error obtained combining
        sigma_dl_posterior, lensing, and peculiar velocity
    11-string_index: source index
    12-LSST_detected: (boolean variable) 1 if the source is observed
        by LSST, 0 otherwise
    13-SKA_detected: the same as above (SKA)
    14-Athena_detected: the same as above (Athena)
    """
    all_files = os.listdir(input_folder)
    print(f"\nReading {input_folder}")
    events_list = [f for f in all_files if ("EVENT" in f)]
    # Two-file format.
    try:
        if (event_number is None):
            analysis_events = []
            for k, ev in enumerate(events_list):
                sys.stderr.write("Reading {0} out of {1} events\r".format(
                    k+1, len(events_list)))
                event_file = open(input_folder+"/"+ev+"/ID.dat", 'r')
                event_id, dl, rel_sigmadl = event_file.readline().split(None)
                ID = np.int(event_id)
                dl = np.float64(dl)
                sigmadl = np.float64(rel_sigmadl)*dl
                event_file.close()      

                try:
                    redshift, d_redshift = np.loadtxt(input_folder
                        +"/"+ev+"/ERRORBOX.dat", unpack=True)
                    redshift = np.atleast_1d(redshift)
                    d_redshift = np.atleast_1d(d_redshift)
                    weights = np.ones(len(redshift))
                    magnitudes = np.ones(len(redshift))
                    zmin = np.float64(np.maximum(
                        redshift - 5.0*d_redshift, 0.0))
                    zmax = np.float64(redshift + 5.0*d_redshift)
                    analysis_events.append(Event(ID,
                                                dl,
                                                sigmadl,
                                                1.0,
                                                1.0,
                                                redshift,
                                                d_redshift,
                                                weights,
                                                magnitudes,
                                                zmin,
                                                zmax,
                                                -1,
                                                -1,
                                                -1,
                                                [0]))
                except:
                    if (TypeError, NameError): 
                        raise
                    else: 
                        sys.stderr.write("Event %s at a distance"%(event_id)
                            +" %s (error %s) has no hosts,"%(dl, sigmadl)
                            +" skipping\n")
        else:
            event_file = open(input_folder+"/"+events_list[event_number] 
                            +"/ID.dat", 'r')
            event_id, dl, sigmadl = event_file.readline().split(None)
            ID = np.int(event_id)
            dl = np.float64(dl)
            sigmadl = np.float64(sigmadl)*dl
            event_file.close()
            try:
                redshift, d_redshift = np.loadtxt(input_folder+"/" 
                    +events_list[event_number]+"/ERRORBOX.dat", unpack=True)
                redshift = np.atleast_1d(redshift)
                d_redshift = np.atleast_1d(d_redshift)
                weights = np.atleast_1d(len(redshift))
                magnitudes = np.atleast_1d(len(redshift))
                zmin = np.float64(np.maximum(
                    redshift - 10.0*d_redshift, 0.0))
                zmax = np.float64(redshift + 10.0*d_redshift)
                analysis_events = [Event(ID,
                                        dl,
                                        sigmadl,
                                        1.0,
                                        1.0,
                                        redshift,
                                        d_redshift,
                                        weights,
                                        magnitudes,
                                        zmin,
                                        zmax,
                                        -1,
                                        -1,
                                        -1,
                                        [0])]
                sys.stderr.write("Selecting event %s"%(event_id)
                    +"at a distance %s (error %s), "%(dl, sigmadl)
                    +"hosts %d\n"%(len(redshifts)))
            except:
                sys.stderr.write("Event %s at a distance"%(event_id)
                    +"%s (error %s) has no hosts, skipping\n"%(dl, sigmadl))
                exit()
    # One-file format.
    except:
        analysis_events = []
        for k, ev in enumerate(events_list):
            sys.stderr.write("Reading {0} out of {1} events\r".format(
                k+1, len(events_list)))
            params = np.genfromtxt(input_folder+"/"+ev+"/ID.dat",
                                        names=True)
            ID = np.int(params["ID"])
            dl = np.float64(params["d_L_shiftedGpc"])*1e3  # in Mpc
            sigmadl = np.float64(params["sigma_dl_fisherGpc"])*1e3  # in Mpc
            redshift = np.atleast_1d(params["z_true"])
            d_redshift = np.atleast_1d(params["error_z"])
            snr = np.float64(params["SNR"])
            weights = np.ones(len(redshift))
            magnitudes = np.ones(len(redshift))
            zmin = np.float64(np.maximum(
                redshift - 5.0*d_redshift, 0.0))
            zmax = np.float64(redshift + 5.0*d_redshift)
            analysis_events.append(Event(ID,
                                        dl,
                                        sigmadl,
                                        1.0,
                                        1.0,
                                        redshift,
                                        d_redshift,
                                        weights,
                                        magnitudes,
                                        zmin,
                                        zmax,
                                        snr,
                                        -1,
                                        -1,
                                        [0]))
        
    sys.stderr.write("\n%d MBHB events loaded\n"%len(analysis_events))

    return analysis_events

def read_dark_siren_event(input_folder, event_number,
                          max_hosts=None, one_host_selection=0,
                          z_event_sel=None, snr_selection=None,
                          snr_threshold=0.0, sigma_pv=0.0023,
                          event_ID_list=None, zhorizon=None,
                          z_gal_cosmo=0, dl_cutoff=None,
                          reduced_cat=None, **kwargs):
    """Read dark_siren data to be passed to CosmologicalModel class.
    The file ID.dat has a single row containing:
    1-event ID
    2-luminosity distance dL (Mpc)
    3-relative error on luminosity distance delta{dL}/dL
    4-rough estimate of comoving volume of the errorbox
    5-observed redshift of the true host
        (true cosmological, not apparent)
    6-minimum redshift assuming the *true cosmology*
    7-maximum redshift assuming the *true cosmology*
    8-fiducial redshift (i.e. the redshift corresponding to the 
        measured distance in the true cosmology)
    9-minimum redshift adding the cosmological prior
    10-maximum redshift adding the cosmological prior
    11-theta offset of the host compared to detector best sky location 
        (in sigmas, i.e. (theta-theta_best)/sigma{theta})
    12-same for phi
    13-same for dL
    14-theta host (rad)
    15-phi host (rad)
    16-dL host (Mpc)
    17-SNR
    18-SNR at the true distance
    
    The file ERRORBOX.dat has all the info you need to run the
    inference code. Each row is a possible host within the errorbox.
    Columns are:
    1-best luminosity distance measured by the detector 
        (the same as col 1 in ID.dat)
    2-redshift of the host candidate (without peculiar velocity)
    3-redshift of the host candidate (with peculiar velocity)
    4-log_10 of the host candidate mass in solar masses
    5-probability of the host according to the multivariate gaussian
        including the prior on cosmology (all rows add to 1)
    6-theta of the host candidate
    7-best theta measured by the detector
    8-difference between the above two in units of detector theta error
    9-phi of the host candidate
    10-best phi measured by the detector
    11-difference between the above two in units of detector phi error
    12-luminosity distance of the host candidate 
        (in the galaxy catalog cosmology)
    13-best dL measured by the detector
    14-difference between the above two in units of detector dL error
    """
    all_files = os.listdir(input_folder)
    print(f"\nReading {input_folder}")
    events_list = [f for f in all_files if "EVENT" in f]
    pv = sigma_pv

    if (event_number is None):
        events = []
        for k, ev in enumerate(events_list):
            # Different catalogs have different numbers of columns,
            # so a try/except is used.
            sys.stderr.write("Reading {0} out of {1} events\r".format(
                k+1, len(events_list)))
            try:
                event_file = open(input_folder + "/" + ev + "/ID.dat", 'r')
                # 1      , 2 , 3        , 4 , 5              ,
                (event_id, dl, rel_sigmadl, Vc, z_observed_true,
                    # 6      , 7        , 8     ,
                    zmin_true, zmax_true, z_true,
                    # 9 , 10  ,  , , , , , , 17 , 18      , 19
                    zmin, zmax, _,_,_,_,_,_, snr, snr_true, _) = (event_file
                    .readline().split(None))
            except(ValueError):
                try:
                    event_file = open(input_folder+"/"+ev+"/ID.dat", 'r')
                    # 1      , 2 , 3        , 4 , 5              ,
                    (event_id, dl, rel_sigmadl, Vc, z_observed_true,
                        # 6      , 7        , 8     ,
                        zmin_true, zmax_true, z_true,
                        # 9 , 10  ,  , , , , , ,
                        zmin, zmax, _,_,_,_,_,_,
                        # 17, 18
                        snr, snr_true) = event_file.readline().split(None)
                except(ValueError):
                    event_file = open(input_folder+"/"+ev+"/ID.dat", 'r')
                    # 1      , 2 , 3        , 4 , 5              ,
                    (event_id, dl, rel_sigmadl, Vc, z_observed_true,
                    # 6      , 7        , 8     ,                 
                    zmin_true, zmax_true, z_true,
                    # 9 , 10  ,  , , , , , , , , ,
                    zmin, zmax, _,_,_,_,_,_,_,_,_,
                    # 20, 21
                    snr, snr_true) = event_file.readline().split(None)

            ID = np.int(event_id)
            dl = np.float64(dl)
            sigmadl = np.float64(rel_sigmadl)*dl
            zmin = np.float64(zmin)
            zmax = np.float64(zmax)
            snr = np.float64(snr)
            VC = np.float64(Vc)
            z_cosmo_true_host = np.float64(z_observed_true)
            z_true = np.float64(z_true)
            event_file.close()
            try:
                try:
                    # 1     , 2      , 3  , 4   , 5      , 6    , 7         ,
                    (best_dl, zcosmo, zobs, logM, weights, theta, best_theta,
                        # 8   , 9  , 10      , 11  ,                    
                        dtheta, phi, best_phi, dphi,
                        # 12   , 13       , 14
                        dl_host, best_dl_2, deltadl) = (np.loadtxt(
                        input_folder+"/"+ev+"/ERRORBOX.dat", unpack=True))
                except:
                    # 1     , 2     , 3   , 4   , 5      , 6    , 7         ,
                    (best_dl, zcosmo, zobs, logM, weights, theta, best_theta,
                        # 8   , 9  , 10      , 11  ,
                        dtheta, phi, best_phi, dphi,
                        # 12   , 13       , 14,    , 15
                        dl_host, best_dl_2, deltadl, _) = (np.loadtxt(
                        input_folder+"/"+ev+"/ERRORBOX.dat", unpack=True))
                if not z_gal_cosmo:
                    redshifts = np.atleast_1d(zobs)
                else:
                    redshifts = np.atleast_1d(zcosmo)
                d_redshifts = np.ones(len(redshifts))*pv
                weights = np.atleast_1d(weights)
                magnitudes = np.atleast_1d(logM)   # Fictitious values 
                sigma_gw_theta = np.mean((theta-best_theta)/dtheta)
                sigma_gw_phi = np.mean((phi-best_phi)/dphi)
                if not (isinstance(dl_host, type(redshifts))):
                    dl_host = np.atleast_1d(dl_host)
                events.append(Event(ID,
                                    dl,
                                    sigmadl,
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
                                    z_cosmo_true_host,
                                    dl_host,
                                    VC = VC))
            except:
                print(f"Event {event_id} at a distance {dl} (error {sigmadl})"
                      " has no hosts, skipping\n")

        if (snr_selection is not None):
            new_list = sorted(events, key=lambda x: getattr(x, 'snr'))
            if (snr_selection > 0):
                events = new_list[:snr_selection]
            elif (snr_selection < 0):
                events = new_list[snr_selection:]
            print(f"\nSelected {len(events)} events from SNR={events[0].snr}" 
                  f" to SNR={events[abs(snr_selection)-1].snr}:")            
            for e in events:
                print("ID: {}  |  SNR: {}".format(str(e.ID).ljust(3),
                                                  str(e.snr).ljust(9)))

        if (z_event_sel is not None):
            new_list = sorted(events, key=lambda x: getattr(x, 'z_true'))
            if (z_event_sel > 0):
                events = new_list[:z_event_sel]
            elif (z_event_sel < 0):
                events = new_list[z_event_sel:]
            print(f"\nSelected {len(events)} events from z={events[0].z_true}"
                  f" to z={events[abs(z_event_sel)-1].z_true}:")
            for e in events:
                print("ID: {}  |  z_true: {}".format(str(e.ID).ljust(3),
                                                     str(e.z_true).ljust(7)))

        if (zhorizon is not None):
            if (',' in zhorizon):
                z_horizons = zhorizon.split(',')
                z_hor_min = float(z_horizons[0])
                z_hor_max = float(z_horizons[1])
            else:
                z_hor_min = 0.0
                z_hor_max = float(zhorizon)
            events = [e for e in events 
                      if (z_hor_min <= e.z_true <= z_hor_max)]
            events = sorted(events, key=lambda x: getattr(x, 'z_true'))
            if (len(events) != 0):
                print(f"\nSelected {len(events)} events from"
                      f" z={events[0].z_true} to"
                      f" z={events[len(events)-1].z_true}"
                      f" (z_hor_min, z_hor_max=[{z_hor_min},{z_hor_max}]):")
                for e in events:
                    print("ID: {}  |  z_true: {}".format(str(e.ID).ljust(3),
                                                    str(e.z_true).ljust(7))) 
            else:
                print("Zero events found in the redshift window"
                      f" [{z_hor_min},{z_hor_max}].")

        if (dl_cutoff is not None):
            print("\nSelecting events according to"
                  f" dL(omega_true,e.zmax) < {dl_cutoff} (Mpc):")
            events = [e for e in events 
                if (kwargs['omega_true'].LuminosityDistance(e.zmax)
                < dl_cutoff)]
            print("\nSelected {} events from dl={} to dl={} (Mpc)."
                .format(len(events), events[0].dl, events[len(events)-1].dl))  

        if not (max_hosts == 0):
            events = [e for e in events if e.n_hosts <= max_hosts]
            events = sorted(events, key=lambda x: getattr(x, 'n_hosts'))
            print(f"\nSelected {len(events)} events having hosts from"
                  f" n={events[0].n_hosts} to"
                  f" n={events[len(events)-1].n_hosts}"
                  f" (max hosts imposed={max_hosts}):")
            for e in events:
                print("ID: {}  |  n_hosts: {}".format(str(e.ID).ljust(3),
                                                    str(e.n_hosts).ljust(7)))

        if not (snr_threshold == 0.0):
            if (reduced_cat is None):
                if (snr_threshold > 0):
                    print("\nSelecting events according to"
                          f" snr_threshold > {snr_threshold}:")
                    events = [e for e in events if e.snr > snr_threshold]
                else:
                    print(f"\nSelecting events up to"
                    " snr_threshold < {snr_threshold}:")
                    events = [e for e in events if e.snr < abs(snr_threshold)]
                events = sorted(events, key=lambda x: getattr(x, 'snr'))
                print("\nSelected {} events".format(len(events))
                      +" from SNR={}".format(events[0].snr)
                      +" to SNR={}".format(events[len(events)-1].snr)
                      +" (SNR_threshold={}):".format(snr_threshold))
                for e in events:
                    print("ID: {}  |  SNR: {}".format(str(e.ID).ljust(3),
                                                    str(e.snr).ljust(7)))
            else:
                # Draw a number of events in the 4-year scenario.
                N = np.int(np.random.poisson(len(events)*4./10.))
                print(f"\nReduced number of events: {N}")
                selected_events = []
                k = 0
                while k < N and not(len(events) == 0):
                    idx = np.random.randint(len(events))
                    selected_event = events.pop(idx)
                    print("Drawn event {0}: ID={1} - SNR={2:.2f}".format(k+1,
                        str(selected_event.ID).ljust(3), selected_event.snr))
                    if (snr_threshold > 0.0):
                        if (selected_event.snr > snr_threshold):
                            print("Selected: ID="
                                +"{0}".format(str(selected_event.ID).ljust(3))
                                +" - SNR={1:.2f}".format(selected_event.snr)
                                +" > {2:.2f}".format(snr_threshold))
                            selected_events.append(selected_event)
                        else: pass
                        k += 1
                    else:
                        if (selected_event.snr < abs(snr_threshold)):
                            print("Selected: ID="
                                +"{0}".format(str(selected_event.ID).ljust(3))
                                +" - SNR={1:.2f}".format(selected_event.snr)
                                +" < {2:.2f}".format(snr_threshold))
                            selected_events.append(selected_event)
                        else: pass
                        k += 1
                events = selected_events
                events = sorted(selected_events, 
                                key=lambda x: getattr(x, 'snr'))
                print("\nSelected {} events".format(len(events))
                    +" from SNR={}".format(events[0].snr)
                    +" to SNR={}:".format(events[len(events)-1].snr))
                for e in events:
                    print("ID: {}  |  dl: {}".format(str(e.ID).ljust(3),
                                                     str(e.dl).ljust(9)))

        if (event_ID_list is not None):
            event_list = []
            ID_list = event_ID_list.split(',')
            events = [e for e in events if str(e.ID) in ID_list]

        if (one_host_selection):
            for e in events:
                z_differences = []
                for gal in e.potential_galaxy_hosts:
                    z_diff = abs(e.z_true - gal.redshift)
                    z_differences.append(z_diff)
                    if (z_diff == min(z_differences)):
                        selected_gal = gal 
                e.potential_galaxy_hosts = [selected_gal]
            print("\nUsing only the nearest host to the GW source:")
            events = sorted(events, key=lambda x: getattr(x, 'ID'))
            for e in events:
                print(f"ID: {str(e.ID).ljust(3)}  |  "
                      f"SNR: {str(e.snr).ljust(9)}  |  "
                      f"dl: {str(e.dl).ljust(7)} Mpc  |  "
                      f"z_true: {str(e.z_true).ljust(7)} |  "
                      "z_nearest_host: "
                      f"{str(e.potential_galaxy_hosts[0].redshift).ljust(7)}"
                      " |  hosts:" 
                      f"{str(len(e.potential_galaxy_hosts)).ljust(4)}")

        analysis_events = events
        del events_list
    else:
        events_list.sort()
        analysis_events = []
        event_file = open(input_folder+"/"+events_list[event_number]
                          +"/ID.dat","r")
        (event_id, dl, rel_sigmadl, Vc, z_observed_true, zmin_true, zmax_true,
            z_true, zmin, zmax, _,_,_,_,_,_,
            snr, snr_true) = event_file.readline().split(None)
        ID = np.int(event_id)
        dl = np.float64(dl)
        sigmadl = np.float64(rel_sigmadl)*dl
        zmin = np.float64(zmin)
        zmax = np.float64(zmax)
        snr = np.float64(snr)
        VC = np.float64(Vc)
        z_true = np.float64(z_true)
        event_file.close()
        try:
            (best_dl, zcosmo, zobs, magnitudes, weights, theta, best_theta,
                dtheta, phi, best_phi, dphi, dl_host, best_dl_2, 
                deltadl) = np.loadtxt(input_folder+"/"
                +events_list[event_number]+"/ERRORBOX.dat",unpack=True)
            redshifts = np.atleast_1d(zobs)
            d_redshifts = np.ones(len(redshifts))*pv
            weights = np.atleast_1d(weights)
            magnitudes = np.atleast_1d(magnitudes)
            analysis_events.append(Event(ID, dl, sigmadl, 0.0, 0.0, redshifts,
                                         d_redshifts, weights, magnitudes,
                                         zmin, zmax, snr, z_true, dl_host,
                                         VC=VC))
        except:
            sys.stderr.write(f"Event {event_id} at a distance {dl} (error"
                             f"{sigmadl}) has no EM counterpart, skipping\n")

    return analysis_events

def pick_random_events(events, number):
    print(f"\nSelecting {number} random events for joint analysis.")
    if (number >= len(events)):
        print(f"Required {number} random events, but the catalog has only"
              f" {len(events)}. Running on {len(events)} events.")
        number = len(events)
    events = np.random.choice(events, size=number, replace=False)
    return events
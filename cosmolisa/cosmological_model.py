#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from optparse import OptionParser
from scipy.special import logsumexp
from scipy.stats import norm
import lal
import cpnest.model
import sys
import os
import readdata
import matplotlib
import corner
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import cosmolisa.cosmology as cs
import cosmolisa.likelihood as lk
import cosmolisa.prior as pr


"""
Default formulation with em_selection = 0
    Di   = a GW
    O    = omega (cosmological parameters)

    p(Di|O H I) = int dL dz_gw p(z_gw|O H I)*p(dL|z_gw O H I)*p(Di|dL z_gw O H I)

Alternative formulation with em_selection = 1
    G    = the GW is in a galaxy that I see
    barG = the GW is in a galaxy that I do not see
    Di   = a GW
    O    = omega (cosmological parameters)
    I    = I see only GW with SNR > 20

    p(Di|dL z_gw O H I) = p(Di(G+barG))|dL z_gw O H I) = p(Di G|dL z_gw O H I) + p(Di barG|dL z_gw O H I)
    p(Di|O H I) = int dL dz_gw p(z_gw|O H I)*p(dL|z_gw O H I)*p(Di|dL z_gw O H I)
"""

class CosmologicalModel(cpnest.model.Model):

    names  = [] #'h','om','ol','w0','w1']
    bounds = [] #[0.6,0.86],[0.04,0.5],[0.0,1.0],[-3.0,-0.3],[-1.0,1.0]]
    
    def __init__(self, model, data, *args, **kwargs):

        super(CosmologicalModel,self).__init__()
        # Set up the data
        self.data             = data
        self.N                = len(self.data)
        self.model            = model
        self.em_selection     = kwargs['em_selection']
        self.z_threshold      = kwargs['z_threshold']
        self.snr_threshold    = kwargs['snr_threshold']
        self.event_class      = kwargs['event_class']
        self.redshift_prior   = kwargs['redshift_prior']
        self.time_redshifting = kwargs['time_redshifting']
        self.vc_normalization = kwargs['vc_normalization']
        self.lk_sel_fun       = kwargs['lk_sel_fun']
        self.detection_corr   = kwargs['detection_corr']
        self.approx_int       = kwargs['approx_int']
        self.dl_cutoff        = kwargs['dl_cutoff']
        self.O                = None
        
        if (self.model == "LambdaCDM"):
            
            self.names  = ['h','om']
            self.bounds = [[0.6,0.86],[0.04,0.5]]
        
        elif (self.model == "CLambdaCDM"):
            
            self.names  = ['h','om','ol']
            self.bounds = [[0.6,0.86],[0.04,0.5],[0.0,1.0]]

        elif (self.model == "LambdaCDMDE"):
            
            self.names  = ['h','om','ol','w0','w1']
            self.bounds = [[0.6,0.86],[0.04,0.5],[0.0,1.0],[-3.0,-0.3],[-1.0,1.0]]
            
        elif (self.model == "DE"):
            
            self.names  = ['w0','w1']
            self.bounds = [[-3.0,-0.3],[-1.0,1.0]]
        
        else:
            
            print("Cosmological model %s not supported. Exiting...\n"%self.model)
            exit()
        
        for e in self.data:
            self.names.append('z%d'%e.ID)
            self.bounds.append([e.zmin,e.zmax])
            
#        self.names.append('logr0')
#        self.bounds.append([np.log(1e-12),np.log(1e-9)])
        self._initialise_galaxy_hosts()
        
        print("==================================================")
        print("cpnest model initialised with:")
        print("Cosmological model: {0}".format(self.model))
        print("Number of events:   {0}".format(len(self.data)))
        print("EM correction:      {0}".format(self.em_selection))
        print("Redshift prior:     {0}".format(self.redshift_prior))
        print("Time redshifting:   {0}".format(self.time_redshifting))
        print("Vc normalization:   {0}".format(self.vc_normalization))
        print("lk_sel_fun:         {0}".format(self.lk_sel_fun))
        print("detection_corr:     {0}".format(self.detection_corr))
        print("approx_int:         {0}".format(self.approx_int))
        print("==================================================")

    def _initialise_galaxy_hosts(self):
        self.hosts = {e.ID:np.array([(g.redshift,g.dredshift,g.weight) for g in e.potential_galaxy_hosts]) for e in self.data}

    def log_prior(self,x):
    
        logP = super(CosmologicalModel,self).log_prior(x)
        
        if np.isfinite(logP):
        
            if   (self.model == "LambdaCDM"):
                self.O = cs.CosmologicalParameters(x['h'],x['om'],1.0-x['om'],truths['w0'],truths['w1'])
            elif (self.model == "CLambdaCDM"):
                self.O = cs.CosmologicalParameters(x['h'],x['om'],x['ol'],truths['w0'],truths['w1'])
            elif (self.model == "LambdaCDMDE"):
                self.O = cs.CosmologicalParameters(x['h'],x['om'],x['ol'],x['w0'],x['w1'])
            elif (self.model == "DE"):
                self.O = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],x['w0'],x['w1'])

            if not (self.lk_sel_fun):         
                """
                Apply a uniform in comoving volume density redshift prior
                """
                if (self.redshift_prior == 1):
                    # Compute p(z_gw|...)
                    logP += np.sum([pr.logprior_redshift_single_event(self.O,
                                                                    x['z%d'%e.ID],
                                                                    zmax = self.bounds[2+j][1],
                                                                    dl_cutoff = self.dl_cutoff)
                                                                    for j,e in enumerate(self.data)])
                                                                    
                elif (self.redshift_prior == 0) and (self.time_redshifting == 1):
                    logP += np.sum([-np.log((1+x['z%d'%e.ID])) for e in self.data])

                    if (self.vc_normalization == 1):
                        logP -= np.sum([np.log(self.O.IntegrateComovingVolumeDensity(self.bounds[2+j][1]))
                                        for j,e in enumerate(self.data)])

        return logP

    def log_likelihood(self,x):
        
        if (self.lk_sel_fun == 1):
            logL = np.sum([lk.logLikelihood_single_event_sel_fun(self.hosts[e.ID],
                                                                 e.dl,
                                                                 e.sigma,
                                                                 self.O,
                                                                 x['z%d'%e.ID],
                                                                 approx_int = self.approx_int,
                                                                 zmin = self.bounds[2+j][0],
                                                                 zmax = self.bounds[2+j][1])
                                                                 for j,e in enumerate(self.data)])

        elif (self.detection_corr == 1):
            logL = np.sum([lk.logLikelihood_single_event_snr_threshold(self.hosts[e.ID],
                                                        e.dl,
                                                        e.sigma,
                                                        e.snr,
                                                        self.O,
                                                        x['z%d'%e.ID],
                                                        em_selection = self.em_selection,
                                                        zmin = self.bounds[2+j][0],
                                                        zmax = self.bounds[2+j][1],
                                                        SNR_threshold = 20.0)
                                                        for j,e in enumerate(self.data)])
        else:
            # Compute sum_GW p(z_gw|...)*p(dL|...)*p(Di|...)
            logL = np.sum([lk.logLikelihood_single_event(self.hosts[e.ID],
                                                        e.dl,
                                                        e.sigma,
                                                        self.O,
                                                        x['z%d'%e.ID],
                                                        em_selection = self.em_selection,
                                                        zmin = self.bounds[2+j][0],
                                                        zmax = self.bounds[2+j][1])
                                                        for j,e in enumerate(self.data)])

        self.O.DestroyCosmologicalParameters()

        return logL

truths = {'h':0.73,'om':0.25,'ol':0.75,'w0':-1.0,'w1':0.0} # 0.73, 0.25, 0.75, -1.0, 0.0
usage=""" %prog (options)"""

if __name__=='__main__':

    parser = OptionParser(usage)
    parser.add_option('-d', '--data',        default=None,        type='string', metavar='data',             help='Galaxy data location.')
    parser.add_option('-o', '--out_dir',     default=None,        type='string', metavar='DIR',              help='Directory for output.')
    parser.add_option('-c', '--event_class', default=None,        type='string', metavar='event_class',      help='Class of the event(s) [MBH, EMRI, sBH].')
    parser.add_option('-e', '--event',       default=None,        type='int',    metavar='event',            help='Event number.')
    parser.add_option('-m', '--model',       default='LambdaCDM', type='string', metavar='model',            help='Cosmological model to assume for the analysis (default LambdaCDM). Supports LambdaCDM, CLambdaCDM, LambdaCDMDE, and DE.')
    parser.add_option('-j', '--joint',       default=0,           type='int',    metavar='joint',            help='Run a joint analysis for N events, randomly selected (EMRI only).')
    parser.add_option('-z', '--zhorizon',    default=1000.0,      type='float',  metavar='zhorizon',         help='Horizon redshift corresponding to the SNR threshold.')
    parser.add_option('--dl_cutoff',         default=-1.0,        type='float',  metavar='dl_cutoff',        help='Max EMRI dL(omega_true,zmax) allowed (in Mpc). This cutoff supersedes the zhorizon one.')
    parser.add_option('--z_selection',       default=None,        type='int',    metavar='z_selection',      help='Select events according to redshift.')
    parser.add_option('--one_host_sel',      default=0,           type='int',    metavar='one_host_sel',     help='Select only the nearest host in redshift for each EMRI.')
    parser.add_option('--max_hosts',         default=None,        type='int',    metavar='max_hosts',        help='Select events according to the allowed maximum number of hosts.')
    parser.add_option('--snr_selection',     default=None,        type='int',    metavar='snr_selection',    help='Select events according to SNR.')
    parser.add_option('--snr_threshold',     default=0.0,         type='float',  metavar='snr_threshold',    help='SNR detection threshold.')
    parser.add_option('--em_selection',      default=0,           type='int',    metavar='em_selection',     help='Use EM selection function.')
    parser.add_option('--redshift_prior',    default=0,           type='int',    metavar='redshift_prior',   help='Adopt a redshift prior with comoving volume factor.')
    parser.add_option('--time_redshifting',  default=0,           type='int',    metavar='time_redshifting', help='Add a factor 1/1+z_gw as redshift prior.')
    parser.add_option('--vc_normalization',  default=0,           type='int',    metavar='vc_normalization', help='Add a covolume factor normalization to the redshift prior.')
    parser.add_option('--lk_sel_fun',        default=0,           type='int',    metavar='lk_sel_fun',       help='Single-event likelihood a la Jon Gair.')
    parser.add_option('--detection_corr',    default=0,           type='int',    metavar='detection_corr',   help='Single-event likelihood including detection correction')
    parser.add_option('--approx_int',        default=0,           type='int',    metavar='approx_int',       help='Approximate the in-catalog weight with the selection function.')
    parser.add_option('--reduced_catalog',   default=0,           type='int',    metavar='reduced_catalog',  help='Select randomly only a fraction of the catalog.')
    parser.add_option('--threads',           default=None,        type='int',    metavar='threads',          help='Number of threads (default = 1/core).')
    parser.add_option('--seed',              default=0,           type='int',    metavar='seed',             help='Random seed initialisation.')
    parser.add_option('--nlive',             default=5000,        type='int',    metavar='nlive',            help='Number of live points.')
    parser.add_option('--poolsize',          default=256,         type='int',    metavar='poolsize',         help='Poolsize for the samplers.')
    parser.add_option('--maxmcmc',           default=1000,        type='int',    metavar='maxmcmc',          help='Maximum number of mcmc steps.')
    parser.add_option('--postprocess',       default=0,           type='int',    metavar='postprocess',      help='Run only the postprocessing. It works only with reduced_catalog=0.')
    parser.add_option('--screen_output',     default=0,           type='int',    metavar='screen_output',    help='Print the output on screen or save it into a file.')
    (opts,args)=parser.parse_args()

    em_selection       = opts.em_selection
    dl_cutoff          = opts.dl_cutoff
    z_selection        = opts.z_selection
    snr_selection      = opts.snr_selection
    one_host_selection = opts.one_host_sel
    zhorizon           = opts.zhorizon
    max_hosts          = opts.max_hosts
    snr_threshold      = opts.snr_threshold
    redshift_prior     = opts.redshift_prior
    time_redshifting   = opts.time_redshifting
    vc_normalization   = opts.vc_normalization
    lk_sel_fun         = opts.lk_sel_fun
    detection_corr     = opts.detection_corr
    approx_int         = opts.approx_int
    model              = opts.model
    joint              = opts.joint
    event_class        = opts.event_class
    reduced_catalog    = opts.reduced_catalog
    postprocess        = opts.postprocess
    screen_output      = opts.screen_output
    out_dir            = opts.out_dir

    omega_true = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],truths['w0'],truths['w1'])

    if not (screen_output):
        if not (postprocess):
            directory = out_dir
            os.system('mkdir -p {0}'.format(directory))

            sys.stdout = open(os.path.join(directory,'stdout.txt'), 'w')
            sys.stderr = open(os.path.join(directory,'stderr.txt'), 'w')

    print("The output will be saved in {}".format(out_dir))
    if (event_class == "MBH"):
        # if running on SMBH override the selection functions
        em_selection = 0

    if (event_class == "EMRI"):
        if (snr_selection is not None):
            events = readdata.read_event(event_class, opts.data, None, snr_selection=snr_selection, one_host_selection=one_host_selection)
        elif (z_selection is not None):
            events = readdata.read_event(event_class, opts.data, None, z_selection=z_selection, one_host_selection=one_host_selection)
        elif (dl_cutoff > 0) and (zhorizon == 1000):
            all_events = readdata.read_event(event_class, opts.data, None, one_host_selection=one_host_selection)
            events_selected = []
            print("\nSelecting events according to dl_cutoff={}:".format(dl_cutoff))
            for e in all_events:
                if (omega_true.LuminosityDistance(e.zmax) < dl_cutoff):
                    events_selected.append(e)
                    print("Event {} selected: dl(z_max)={}.".format(str(e.ID).ljust(3), omega_true.LuminosityDistance(e.zmax)))
            events = sorted(events_selected, key=lambda x: getattr(x, 'dl'))
            print("\nSelected {} events from dl={} to dl={}:".format(len(events), events[0].dl, events[len(events)-1].dl))            
            for e in events:
                print("ID: {}  |  dl: {}".format(str(e.ID).ljust(3), str(e.dl).ljust(9)))     
        elif (zhorizon < 1000):
            events = readdata.read_event(event_class, opts.data, None, zhorizon=zhorizon, one_host_selection=one_host_selection)
        elif (max_hosts is not None):
            events = readdata.read_event(event_class, opts.data, None, max_hosts=max_hosts, one_host_selection=one_host_selection)
        elif (snr_threshold > 0.0):
            if not reduced_catalog:
                events = readdata.read_event(event_class, opts.data, None, snr_threshold=snr_threshold, one_host_selection=one_host_selection)
            else:
                events = readdata.read_event(event_class, opts.data, None, one_host_selection=one_host_selection)
                # Draw a number of events in the 4-year scenario
                N = np.int(np.random.poisson(len(events)*4./10.))
                print("\nReduced number of events: {}".format(N))
                selected_events = []
                k = 0
                while k < N and not(len(events) == 0):
                    idx = np.random.randint(len(events))
                    selected_event = events.pop(idx)
                    print("Drawn event {0}: ID={1} - SNR={2:.2f}".format(k+1, str(selected_event.ID).ljust(3), selected_event.snr))
                    if selected_event.snr > snr_threshold:
                        print("Selected: ID={0} - SNR={1:.2f} > {2:.2f}".format(str(selected_event.ID).ljust(3), selected_event.snr, snr_threshold))
                        selected_events.append(selected_event)
                    else: pass
                    k += 1
                events = selected_events
                events = sorted(selected_events, key=lambda x: getattr(x, 'snr'))
                print("\nSelected {} events from SNR={} to SNR={}:".format(len(events), events[0].snr, events[len(events)-1].snr))
                for e in events:
                    print("ID: {}  |  dl: {}".format(str(e.ID).ljust(3), str(e.dl).ljust(9)))
        else:
            events = readdata.read_event(event_class, opts.data, None, one_host_selection=one_host_selection)

        if (joint != 0):
            N = joint
            if (N > len(events)):
                N = len(events)
                print("The catalog has a number of selected events smaller than the chosen number ({}). Running on {}".format(N, len(events)))
            events = np.random.choice(events, size = N, replace = False)
            print("==================================================")
            print("Selecting a random catalog of {0} events for joint analysis:".format(N))
            print("==================================================")
            if not(len(events) == 0):
                for e in events:
                    print("event {0}: distance {1} \pm {2} Mpc, z \in [{3},{4}] galaxies {5}".format(e.ID,e.dl,e.sigma,e.zmin,e.zmax,len(e.potential_galaxy_hosts)))
                print("==================================================")
            else:
                print("None of the drawn events has z<{0}. No data to analyse. Exiting.\n".format(zhorizon))
                exit()
    else:
        events = readdata.read_event(event_class, opts.data, opts.event)

    if (len(events) == 0):
        print("The passed catalog is empty. Exiting.\n")
        exit()

    print("\nDetailed list of the %d selected events:\n"%len(events))
    print("==================================================")
    events = sorted(events, key=lambda x: getattr(x, 'ID'))
    for e in events:
        print("ID: {}  |  SNR: {}  |  z_true: {} |  dl: {} Mpc  |  sigmadl: {} Mpc  |  hosts: {}".format(
        str(e.ID).ljust(3), str(e.snr).ljust(9), str(e.z_true).ljust(7), 
        str(e.dl).ljust(7), str(e.sigma)[:6].ljust(7), str(len(e.potential_galaxy_hosts)).ljust(4)))

    if out_dir is None:
        output = opts.data+"/EVENT_1%03d/"%(opts.event+1)
    else:
        output = out_dir

    print("==================================================")
    print("\nCPNest will be initialised with:")
    print("poolsize: {0}".format(opts.poolsize))
    print("nlive:    {0}".format(opts.nlive))
    print("maxmcmc:  {0}".format(opts.maxmcmc))
    print("nthreads: {0}".format(opts.threads))

    C = CosmologicalModel(model,
                          events,
                          em_selection     = em_selection,
                          snr_threshold    = snr_threshold,
                          z_threshold      = zhorizon,
                          event_class      = event_class,
                          redshift_prior   = redshift_prior,
                          time_redshifting = time_redshifting,
                          vc_normalization = vc_normalization,
                          lk_sel_fun       = lk_sel_fun,
                          detection_corr   = detection_corr,
                          approx_int       = approx_int,
                          dl_cutoff        = dl_cutoff)

    #FIXME: postprocess doesn't work when events are randomly selected, since 'events' in C are different from the ones read from chain.txt
    if (postprocess == 0):
        work=cpnest.CPNest(C,
                           verbose      = 2,
                           poolsize     = opts.poolsize,
                           nthreads     = opts.threads,
                           nlive        = opts.nlive,
                           maxmcmc      = opts.maxmcmc,
                           output       = output,
                           nhamiltonian = 0)
    # To plot prior&posterior: verbose =3, or prior-sampling = True

        work.run()
        print('log Evidence {0}'.format(work.NS.logZ))
        x = work.posterior_samples.ravel()

        # Save git info
        with open("{}/git_info.txt".format(out_dir), "w+") as fileout:
            subprocess.call(["git", "diff"], stdout=fileout)

    else:
        print("Reading the chain...")
        x = np.genfromtxt(os.path.join(output,"chain_"+str(opts.nlive)+"_1234.txt"), names=True)
        from cpnest import nest2pos
        print("Drawing posterior samples...")
        x = nest2pos.draw_posterior_many([x], [opts.nlive], verbose=False)

    ###############################################
    ################# MAKE PLOTS ##################
    ###############################################
    
    if (event_class == "EMRI"):
        for e in C.data:
            fig = plt.figure()
            ax  = fig.add_subplot(111)
            z   = np.linspace(e.zmin,e.zmax, 100)

            if (em_selection):
                ax2 = ax.twinx()
                
                if (model == "DE"): normalisation = matplotlib.colors.Normalize(vmin=np.min(x['w0']), vmax=np.max(x['w0']))
                else:               normalisation = matplotlib.colors.Normalize(vmin=np.min(x['h']), vmax=np.max(x['h']))
                # choose a colormap
                c_m = matplotlib.cm.cool
                # create a ScalarMappable and initialize a data structure
                s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=normalisation)
                s_m.set_array([])
                for i in range(x.shape[0])[::10]:
                    if (model == "LambdaCDM"): O = cs.CosmologicalParameters(x['h'][i],x['om'][i],1.0-x['om'][i],truths['w0'],truths['w1'])
                    elif (model == "CLambdaCDM"): O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],truths['w0'],truths['w1'])
                    elif (model == "LambdaCDMDE"): O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],x['w0'][i],x['w1'][i])
                    elif (model == "DE"): O = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],x['w0'][i],x['w1'][i])
                    distances = np.array([O.LuminosityDistance(zi) for zi in z])
                    if (model == "DE"):  ax2.plot(z, [lk.em_selection_function(d) for d in distances], lw = 0.15, color=s_m.to_rgba(x['w0'][i]), alpha = 0.5)
                    else: ax2.plot(z, [lk.em_selection_function(d) for d in distances], lw = 0.15, color=s_m.to_rgba(x['h'][i]), alpha = 0.5)
                    O.DestroyCosmologicalParameters()
                CB = plt.colorbar(s_m, orientation='vertical', pad=0.15)
                if (model == "DE"): CB.set_label('w_0')
                else: CB.set_label('h')
                ax2.set_ylim(0.0, 1.0)
                ax2.set_ylabel('selection function')
            ax.axvline(e.z_true, linestyle='dotted', lw=0.5, color='k')
            ax.hist(x['z%d'%e.ID], bins=z, density=True, alpha = 0.5, facecolor="green")
            ax.hist(x['z%d'%e.ID], bins=z, density=True, alpha = 0.5, histtype='step', edgecolor="k")

            for g in e.potential_galaxy_hosts:
                zg = np.linspace(g.redshift - 5*g.dredshift, g.redshift+5*g.dredshift, 100)
                pg = norm.pdf(zg, g.redshift, g.dredshift*(1+g.redshift))*g.weight
                ax.plot(zg, pg, lw=0.5, color='k')
            ax.set_xlabel('$z_{%d}$'%e.ID)
            ax.set_ylabel('probability density')
            plt.savefig(os.path.join(output,'redshift_%d'%e.ID+'.png'), bbox_inches='tight')
            plt.close()
    
    if (event_class == "MBH"):
        dl = [e.dl/1e3 for e in C.data]
        ztrue = [e.potential_galaxy_hosts[0].redshift for e in C.data]
        dztrue = np.squeeze([[ztrue[i]-e.zmin,e.zmax-ztrue[i]] for i,e in enumerate(C.data)]).T
        deltadl = [np.sqrt((e.sigma/1e3)**2+(lk.sigma_weak_lensing(e.potential_galaxy_hosts[0].redshift,e.dl)/1e3)**2) for e in C.data]
        z = [np.median(x['z%d'%e.ID]) for e in C.data]
        deltaz = [2*np.std(x['z%d'%e.ID]) for e in C.data]
        
        redshift = np.logspace(-3,1.0,100)

        # loop over the posterior samples to get all models to then average
        # for the plot
        
        models = []
        
        for k in range(x.shape[0]):
            if (model == "LambdaCDM"):
                omega = cs.CosmologicalParameters(x['h'][k],
                                               x['om'][k],
                                               1.0-x['om'][k],
                                               truths['w0'],
                                               truths['w1'])
            elif (model == "CLambdaCDM"):
                omega = cs.CosmologicalParameters(x['h'][k],
                                               x['om'][k],
                                               x['ol'][k],
                                               truths['w0'],
                                               truths['w1'])
            elif (model == "LambdaCDMDE"):
                omega = cs.CosmologicalParameters(x['h'][k],
                                               x['om'][k],
                                               x['ol'][k],
                                               x['w0'][k],
                                               x['w1'][k])
            elif (model == "DE"):
                omega = cs.CosmologicalParameters(truths['h'],
                                               truths['om'],
                                               truths['ol'],
                                               x['w0'][k],
                                               x['w1'][k])
            else:
                print(opts.model,"is unknown. Exiting.")
                exit()
            models.append([omega.LuminosityDistance(zi)/1e3 for zi in redshift])
            omega.DestroyCosmologicalParameters()
        
        models = np.array(models)
        model2p5,model16,model50,model84,model97p5 = np.percentile(models,[2.7,16.0,50.0,84.0,97.5],axis = 0)
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(z,dl,xerr=deltaz,yerr=deltadl,markersize=1,linewidth=2,color='k',fmt='o')
        ax.plot(redshift,[omega_true.LuminosityDistance(zi)/1e3 for zi in redshift],linestyle='dashed',color='red', zorder = 22)
        ax.plot(redshift,model50,color='k')
        ax.errorbar(ztrue, dl, yerr=deltadl, xerr=dztrue, markersize=2,linewidth=1,color='r',fmt='o')
        ax.fill_between(redshift,model2p5,model97p5,facecolor='turquoise')
        ax.fill_between(redshift,model16,model84,facecolor='cyan')
        ax.set_xlabel(r"z")
        ax.set_ylabel(r"$D_L$/Gpc")
#        ax.set_xlim(np.min(redshift),0.8)
#        ax.set_ylim(0.0,4.0)
        fig.savefig(os.path.join(output,'regression.pdf'),bbox_inches='tight')
        plt.close()
    

    if (model == "LambdaCDM"):
        samps = np.column_stack((x['h'],x['om']))
        fig = corner.corner(samps,
               labels= [r'$h$',
                        r'$\Omega_m$'],
               quantiles=[0.05, 0.5, 0.95],
               show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
               use_math_text=True, truths=[truths['h'],truths['om']],
               filename=os.path.join(output,'joint_posterior.pdf'))
#        axes = fig.get_axes()
#        axes[0].set_xlim(0.69, 0.77)
#        axes[2].set_xlim(0.69, 0.77)
#        axes[3].set_xlim(0.04, 0.5)
#        axes[2].set_ylim(0.04, 0.5)
    
    if (model == "CLambdaCDM"):
        samps = np.column_stack((x['h'],x['om'],x['ol'],1.0-x['om']-x['ol']))
        fig = corner.corner(samps,
               labels= [r'$h$',
                        r'$\Omega_m$',
                        r'$\Omega_\Lambda$',
                        r'$\Omega_k$'],
               quantiles=[0.05, 0.5, 0.95],
               show_titles=True, title_kwargs={"fontsize": 12},
               use_math_text=True, truths=[truths['h'],truths['om'],truths['ol'],0.0],
               filename=os.path.join(output,'joint_posterior.pdf'))
               
    if (model == "LambdaCDMDE"):
        samps = np.column_stack((x['h'],x['om'],x['ol'],x['w0'],x['w1']))
        fig = corner.corner(samps,
                        labels= [r'$h$',
                                 r'$\Omega_m$',
                                 r'$\Omega_\Lambda$',
                                 r'$w_0$',
                                 r'$w_a$'],
                        quantiles=[0.05, 0.5, 0.95],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        use_math_text=True, truths=[truths['h'],truths['om'],truths['ol'],truths['w0'],truths['w1']],
                        filename=os.path.join(output,'joint_posterior.pdf'))

    if (model == "DE"):
        samps = np.column_stack((x['w0'],x['w1']))
        fig = corner.corner(samps,
                        labels= [r'$w_0$',
                                 r'$w_a$'],
                        quantiles=[0.05, 0.5, 0.95],
                        show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                        use_math_text=True, truths=[truths['w0'],truths['w1']],
                        filename=os.path.join(output,'joint_posterior.pdf'))
#        axes = fig.get_axes()
#        axes[0].set_xlim(-1.22, -0.53)
#        axes[2].set_xlim(-1.22, -0.53)
#        axes[3].set_xlim(-1.0, 1.0)
#        axes[2].set_ylim(-1.0, 1.0)

    fig.savefig(os.path.join(output,'joint_posterior.pdf'), bbox_inches='tight')


############################################################
############################################################
# UNUSED CODE

#    redshifts = [e.z_true for e in events]
#    galaxy_redshifts = [g.redshift for e in events for g in e.potential_galaxy_hosts]
#
#    import matplotlib
#    import matplotlib.pyplot as plt
#    fig = plt.figure(figsize=(10,8))
#    z = np.linspace(0.0,0.63,100)
#    normalisation = matplotlib.colors.Normalize(vmin=0.5, vmax=1.0)
#    normalisation2 = matplotlib.colors.Normalize(vmin=0.04, vmax=1.0)
#    # choose a colormap
#    c_m = matplotlib.cm.cool
#
#    # create a ScalarMappable and initialize a data structure
#    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=normalisation)
#    s_m.set_array([])
#
#    # choose a colormap
#    c_m2 = matplotlib.cm.rainbow
#
#    # create a ScalarMappable and initialize a data structure
#    s_m2 = matplotlib.cm.ScalarMappable(cmap=c_m2, norm=normalisation2)
#    s_m2.set_array([])
#
#    plt.hist(redshifts, bins=z, density=True, alpha = 0.5, facecolor="yellow", cumulative=True)
#    plt.hist(redshifts, bins=z, density=True, alpha = 0.5, histtype='step', edgecolor="k", cumulative=True)
##    plt.hist(galaxy_redshifts, bins=z, density=True, alpha = 0.5, facecolor="green", cumulative=True)
##    plt.hist(galaxy_redshifts, bins=z, density=True, alpha = 0.5, histtype='step', edgecolor="k", linestyle='dashed', cumulative=True)
#    for _ in range(1000):
#        h = np.random.uniform(0.5,1.0)
#        om = np.random.uniform(0.04,1.0)
#        ol = 1.0-om
##        h = 0.73
##        om = 0.25
##        ol = 1.0-om
#        O = cs.CosmologicalParameters(h,om,ol,-1.0,0.0)
##        distances = np.array([O.LuminosityDistance(zi) for zi in z])
##        plt.plot(z, [lk.em_selection_function(d) for d in distances], lw = 0.25, color=s_m.to_rgba(h), alpha = 0.75, linestyle='dashed')
#        pz = np.array([O.UniformComovingVolumeDensity(zi) for zi in z])/O.IntegrateComovingVolumeDensity(z.max())
##        pz = np.array([O.ComovingVolumeElement(zi) for zi in z])/O.IntegrateComovingVolume(z.max())
#        plt.plot(z,np.cumsum(pz)/pz.sum(), lw = 0.15, color=s_m2.to_rgba(om), alpha = 0.5)
#        O.DestroyCosmologicalParameters()
#
##        p = lal.CreateCosmologicalParametersAndRate()
##        p.omega.h = h
##        p.omega.om = om
##        p.omega.ol = ol
##        p.omega.w0 = -1.0
##
##        p.rate.r0 = 1e-12
##        p.rate.W  = np.random.uniform(0.0,10.0)
##        p.rate.Q  = np.random.normal(0.63,0.01)
##        p.rate.R  = np.random.normal(1.0,0.1)
##        pz = np.array([lal.RateWeightedUniformComovingVolumeDensity(zi, p) for zi in z])/lal.IntegrateRateWeightedComovingVolumeDensity(p,z.max())
##        plt.plot(z, np.cumsum(pz)/pz.sum(), color=s_m2.to_rgba(om), linewidth = 0.5, linestyle='solid', alpha = 0.5)
##        lal.DestroyCosmologicalParametersAndRate(p)
##        plt.plot(z, pz/(pz*np.diff(z)[0]).sum(), color=s_m2.to_rgba(om), linewidth = 0.5, linestyle='solid', alpha = 0.5)
#
#
#
#    O = cs.CosmologicalParameters(0.73,0.25,0.75,-1.0,0.0)
#    pz = np.array([O.ComovingVolumeElement(zi) for zi in z])/O.IntegrateComovingVolume(z.max())
#    pz = np.array([O.UniformComovingVolumeDensity(zi) for zi in z])/O.IntegrateComovingVolumeDensity(z.max())
#    distances = np.array([O.LuminosityDistance(zi) for zi in z])
#    plt.plot(z, [lk.em_selection_function(d) for d in distances], lw = 0.5, color='k', linestyle='dashed')
#    O.DestroyCosmologicalParameters()
#    plt.plot(z,np.cumsum(pz)/pz.sum(), lw = 0.5, color='k')
##    plt.plot(z,pz/(pz*np.diff(z)[0]).sum(), lw = 0.5, color='k')
#    CB = plt.colorbar(s_m, orientation='vertical', pad=0.15)
#    CB.set_label(r'$h$')
#    CB = plt.colorbar(s_m2, orientation='horizontal', pad=0.15)
#    CB.set_label(r'$\Omega_m$')
#    plt.xlabel('redshift')
##    plt.xlim(0.,0.3)
#    plt.show()
#    exit()

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
import scipy.special as ss
import matplotlib.pyplot as plt
import cosmolisa.cosmology as cs
import cosmolisa.likelihood as lk
import cosmolisa.galaxy as gal
import cosmolisa.prior as pr

class CosmologicalModel(cpnest.model.Model):

    names  = [] #'h','om','ol','w0','w1']
    bounds = [] #[0.6,0.86],[0.04,0.5],[0.0,1.0],[-3.0,-0.3],[-1.0,1.0]]
    
    def __init__(self, model, data, corrections, *args, **kwargs):

        super(CosmologicalModel,self).__init__()
        # Set up the data
        self.data                = data
        self.N                   = len(self.data)
        self.model               = model.split('+')
        self.corrections         = corrections.split('+')
        self.em_selection        = kwargs['em_selection']
        self.z_threshold         = kwargs['z_threshold']
        self.snr_threshold       = kwargs['snr_threshold']
        self.event_class         = kwargs['event_class']
        self.redshift_prior      = kwargs['redshift_prior']
        self.time_redshifting    = kwargs['time_redshifting']
        self.vc_normalization    = kwargs['vc_normalization']
        self.lk_sel_fun          = kwargs['lk_sel_fun']
        self.detection_corr      = kwargs['detection_corr']
        self.approx_int          = kwargs['approx_int']
        self.dl_cutoff           = kwargs['dl_cutoff']
        self.sfr                 = kwargs['sfr']
        self.T                   = kwargs['T']
        self.luminosity_function = kwargs['luminosity_function']
        self.magnitude_threshold = kwargs['m_threshold']
        self.O                   = None
        
        self.Mmin = -25.0
        self.Mmax = -15.0
        self.rate = 0
        self.luminosity = 0
        self.gw = 0
        self.cosmology = 0
        
        if ("LambdaCDM_h" in self.model):
            
            self.cosmology = 1
            self.npar      = 1
            self.names     = ['h']
            self.bounds    = [[0.6,0.86]]            

        if ("LambdaCDM_om" in self.model):
            
            self.cosmology = 1
            self.npar      = 1
            self.names     = ['om']
            self.bounds    = [[0.04,0.5]]

        if ("LambdaCDM" in self.model):
            
            self.cosmology = 1
            self.npar      = 2
            self.names     = ['h','om']
            self.bounds    = [[0.6,0.86],[0.04,0.5]]

        if ("CLambdaCDM" in self.model):
            
            self.cosmology = 1
            self.npar      = 3
            self.names     = ['h','om','ol']
            self.bounds    = [[0.6,0.86],[0.04,0.5],[0.0,1.0]]

        if ("LambdaCDMDE" in self.model):
            
            self.cosmology = 1
            self.npar      = 5
            self.names     = ['h','om','ol','w0','w1']
            self.bounds    = [[0.6,0.86],[0.04,0.5],[0.0,1.0],[-3.0,-0.3],[-1.0,1.0]]

        if ("DE" in self.model):
            
            self.cosmology = 1
            self.npar      = 2
            self.names     = ['w0','w1']
            self.bounds    = [[-3.0,-0.3],[-1.0,1.0]]

        if ("GW" in self.model):
            self.gw = 1
        else:
            self.gw = 0
        
        if ("Rate" in self.model):
#           e(z) = r0*(1.0+W)*exp(Q*z)/(exp(R*z)+W)
            self.rate = 1
            self.gw_correction = 1
#            self.names.append('log10r0')
#            self.bounds.append([-20,-7])
#            self.names.append('W')
#            self.bounds.append([0.0,100.0])
#            self.names.append('Q')
#            self.bounds.append([0.0,10.0])
#            self.names.append('R')
#            self.bounds.append([0.0,10.0])
            self.names.append('log10r0')
            self.bounds.append([np.log10(4e-11),np.log10(6e-11)])
            self.names.append('W')
            self.bounds.append([40.0,42.0])
            self.names.append('Q')
            self.bounds.append([2.3,2.5])
            self.names.append('R')
            self.bounds.append([5.1,5.3])
            
        if ("Luminosity" in self.model):
        
            self.luminosity = 1
            self.em_correction = 1
            self.names.append('phistar0')
            self.bounds.append([1e-5,1e-1])
            self.names.append('phistar_exponent')
            self.bounds.append([-0.1,0.1])
            self.names.append('Mstar0')
            self.bounds.append([-22,-18])
            self.names.append('Mstar_exponent')
            self.bounds.append([-0.1,0.1])
            self.names.append('alpha0')
            self.bounds.append([-2.0,-1.0])
            self.names.append('alpha_exponent')
            self.bounds.append([-0.1,0.1])

        # if we are using GWs, add the relevant redshift parameters
        if self.gw == 1:
            for e in self.data:
                self.names.append('z%d'%e.ID)
                self.bounds.append([e.zmin,e.zmax])
        else:
            self.gw_redshifts = np.array([e.z_true for e in self.data])
        
        self._initialise_galaxy_hosts()
        
        if not("Rate" in self.model):
            if ("GW" in corrections):
                self.gw_correction = 1
            else:
                self.gw_correction = 0
        
        if not("Luminosity" in self.model):
            if ("EM" in corrections):
                self.em_correction = 1
            else:
                self.em_correction = 0
               
        print("==================================================")
        print("cpnest model initialised with:")
        print("Analysis model: {0}".format(self.model))
        print("Number of events: {0}".format(len(self.data)))
        print("EM correction: {0}".format(self.em_correction))
        print("GW correction: {0}".format(self.gw_correction))
        print("Free parameters: {0}".format(self.names))
        print("==================================================")
        print("Prior bounds:")
        for name,bound in zip(self.names, self.bounds):
            print("{}: {}".format(str(name).ljust(4), bound))
        print("==================================================")

    def _initialise_galaxy_hosts(self):
        self.hosts             = {e.ID:np.array([(g.redshift,g.dredshift,g.weight,g.magnitude) for g in e.potential_galaxy_hosts]) for e in self.data}
        self.galaxy_redshifts    = np.hstack([self.hosts[e.ID][:,0] for e in self.data]).copy(order='C')
        self.galaxy_magnitudes   = np.hstack([self.hosts[e.ID][:,3] for e in self.data]).copy(order='C')
        self.areas               = {e.ID:0.000405736691211125 * (87./e.snr)**2 for e in self.data}
        
    def log_prior(self,x):
    
        logP = super(CosmologicalModel,self).log_prior(x)
        
        if np.isfinite(logP):
            
            # check for the cosmological model
            if ("LambdaCDM_h" in self.model):
                self.O = cs.CosmologicalParameters(x['h'],truths['om'],truths['ol'],truths['w0'],truths['w1'])  
            elif ("LambdaCDM_om" in self.model):
                self.O = cs.CosmologicalParameters(truths['h'],x['om'],1.0-x['om'],truths['w0'],truths['w1'])                                
            elif  ("LambdaCDM" in self.model):
                self.O = cs.CosmologicalParameters(x['h'],x['om'],1.0-x['om'],truths['w0'],truths['w1'])
            elif ("CLambdaCDM" in self.model):
                self.O = cs.CosmologicalParameters(x['h'],x['om'],x['ol'],truths['w0'],truths['w1'])
            elif ("LambdaCDMDE" in self.model):
                self.O = cs.CosmologicalParameters(x['h'],x['om'],x['ol'],x['w0'],x['w1'])
            elif ("DE" in self.model):
                self.O = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],x['w0'],x['w1'])
            else:
                self.O = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],truths['w0'],truths['w1'])
            # check for the rate model or GW corrections
            if ("Rate" in self.model):
                self.r0 = 10**x['log10r0']
                self.W  = x['W']
                self.Q  = x['Q']
                self.R  = x['R']
                if self.R <= self.Q:
                    # we want the merger rate to asymptotically either go to zero or to a finite number
                    self.O.DestroyCosmologicalParameters()
                    return -np.inf
            
            elif self.gw_correction == 1:
                self.r0 = truths['r0']
                self.W  = truths['W']
                self.Q  = truths['Q']
                self.R  = truths['R']
            # check for the luminosity model or EM corrections
            if ("Luminosity" in self.model):
                self.phistar0            = x['phistar0']
                self.phistar_exponent    = x['phistar_exponent']
                self.Mstar0              = x['Mstar0']
                self.Mstar_exponent      = x['Mstar_exponent']
                self.alpha0              = x['alpha0']
                self.alpha_exponent      = x['alpha_exponent']
            elif self.em_correction == 1:
                self.phistar0            = truths['phistar0']
                self.phistar_exponent    = truths['phistar_exponent']
                self.Mstar0              = truths['Mstar0']
                self.Mstar_exponent      = truths['Mstar_exponent']
                self.alpha0              = truths['alpha0']
                self.alpha_exponent      = truths['alpha_exponent']
        
        return logP

    def log_likelihood(self,x):
        
        logL_GW         = 0.0
        logL_rate       = 0.0
        logL_luminosity = 0.0
        
        # if we are looking at the luminosity function
        if self.luminosity == 1 and self.gw == 0:
            for e in self.data:
                Schecter = gal.GalaxyDistribution(self.O,
                                                  self.phistar0,
                                                  self.phistar_exponent,
                                                  self.Mstar0,
                                                  self.Mstar_exponent,
                                                  self.alpha0,
                                                  self.alpha_exponent,
                                                  self.Mmin,
                                                  self.Mmax,
                                                  e.zmin,
                                                  e.zmax,
                                                  0.0,
                                                  2.0*np.pi,
                                                  -0.5*np.pi,
                                                  0.5*np.pi,
                                                  self.magnitude_threshold,
                                                  self.areas[e.ID],
                                                  0,0,0)

                logL_luminosity += Schecter.loglikelihood(self.hosts[e.ID][:,3].copy(order='C'), self.hosts[e.ID][:,0].copy(order='C'))

            # if we do not care about GWs, return
            return logL_luminosity
        
        # if we are estimating the rate or we are correcting for GW selection effects, we need this part
        if self.rate == 1 or self.gw_correction == 1:
            Rtot    = lk.integrated_rate(self.r0, self.W, self.R, self.Q, self.O, 1e-5, self.z_threshold)
            Ntot    = Rtot*self.T
            
            # compute the probability of observing the events we observed
            selection_probability = lk.gw_selection_probability_sfr(1e-5, self.z_threshold,
                                                                    self.r0, self.W, self.R, self.Q,
                                                                    self.snr_threshold, self.O, Ntot)
            # compute the rate for the observed events
            Rdet      = Rtot*selection_probability
            Ndet      = Ntot*selection_probability
            logL_rate = -Ndet+self.N*np.log(Ntot)
#            print(selection_probability, Rdet, Rtot, Ndet, Ntot, self.N)
            # if we do not care about GWs, compute the rate density at the known gw redshifts and return
            if self.gw == 0:
                return logL_rate+np.sum([lk.logLikelihood_single_event_rate_only(self.O, e.z_true, self.r0, self.W, self.R, self.Q, Ntot) for e in self.data])
            
        # if we are correcting for EM selection effects, we need this part
        if self.em_correction == 1:
        
            for j,e in enumerate(self.data):
                    
                    Sch = gal.GalaxyDistribution(self.O,
                                                 self.phistar0,
                                                 self.phistar_exponent,
                                                 self.Mstar0,
                                                 self.Mstar_exponent,
                                                 self.alpha0,
                                                 self.alpha_exponent,
                                                 self.Mmin,
                                                 self.Mmax,
                                                 e.zmin,
                                                 e.zmax,
                                                 0.0,
                                                 2.0*np.pi,
                                                 -0.5*np.pi,
                                                 0.5*np.pi,
                                                 self.magnitude_threshold,
                                                 self.areas[e.ID],
                                                 0,0,0)
                    
                    logL_GW += lk.logLikelihood_single_event_sel_fun(self.hosts[e.ID],
                                                                     e.dl,
                                                                     e.sigma,
                                                                     self.O,
                                                                     Sch,
                                                                     x['z%d'%e.ID],
                                                                     zmin = e.zmin,
                                                                     zmax = e.zmax)
                    if self.luminosity == 1:
                        logL_luminosity += Sch.loglikelihood(self.hosts[e.ID][:,3].copy(order='C'), self.hosts[e.ID][:,0].copy(order='C'))

        # we assume the catalog is complete and no correction is necessary
        else:
            logL_GW += np.sum([lk.logLikelihood_single_event(self.hosts[e.ID],
                                                             e.dl,
                                                             e.sigma,
                                                             self.O,
                                                             x['z%d'%e.ID],
                                                             zmin = self.bounds[self.npar+j][0],
                                                             zmax = self.bounds[self.npar+j][1])
                                                             for j,e in enumerate(self.data)])

        self.O.DestroyCosmologicalParameters()

        return logL_GW+logL_rate+logL_luminosity

truths = {'h':0.73,
          'om':0.25,
          'ol':0.75,
          'w0':-1.0,
          'w1':0.0,
          'r0':5e-10,
          'Q':2.4,
          'W':41.,
          'R':5.2,
          'phistar0':1e-2,
          'Mstar0':-20.7,
          'alpha0':-1.23,
          'phistar_exponent':0.0,
          'Mstar_exponent':0.0,
          'alpha_exponent':0.0}
usage=""" %prog (options)"""

if __name__=='__main__':

    parser = OptionParser(usage)
    parser.add_option('-d', '--data',        default=None,        type='string', metavar='data',             help='Galaxy data location.')
    parser.add_option('-o', '--out_dir',     default=None,        type='string', metavar='DIR',              help='Directory for output.')
    parser.add_option('-c', '--event_class', default=None,        type='string', metavar='event_class',      help='Class of the event(s) [MBH, EMRI, sBH].')
    parser.add_option('-e', '--event',       default=None,        type='int',    metavar='event',            help='Event number.')
    parser.add_option('-m', '--model',       default='LambdaCDM', type='string', metavar='model',            help='Cosmological model to assume for the analysis (default LambdaCDM). Supports LambdaCDM, CLambdaCDM, LambdaCDMDE, and DE.')
    parser.add_option('--corrections',       default='None',      type='string', metavar='corrections',      help='family of corrections (GW+EM)')
    parser.add_option('-j', '--joint',       default=0,           type='int',    metavar='joint',            help='Run a joint analysis for N events, randomly selected (EMRI only).')
    parser.add_option('-z', '--zhorizon',    default='1000.0',    type='string', metavar='zhorizon',         help='String to impose low-high cutoffs in redshift. It can be a single number (upper limit) or a string with z_min and z_max separated by a comma.')
    parser.add_option('--dl_cutoff',         default=-1.0,        type='float',  metavar='dl_cutoff',        help='Max EMRI dL(omega_true,zmax) allowed (in Mpc). This cutoff supersedes the zhorizon one.')
    parser.add_option('--z_selection',       default=None,        type='int',    metavar='z_selection',      help='Select ad integer number of events according to redshift.')
    parser.add_option('--one_host_sel',      default=0,           type='int',    metavar='one_host_sel',     help='Select only the nearest host in redshift for each EMRI.')
    parser.add_option('--event_ID_list',     default=None,        type='string', metavar='event_ID_list',    help='String of specific ID events to be read.')
    parser.add_option('--max_hosts',         default=None,        type='int',    metavar='max_hosts',        help='Select events according to the allowed maximum number of hosts.')
    parser.add_option('--snr_selection',     default=None,        type='int',    metavar='snr_selection',    help='Select events according to SNR.')
    parser.add_option('--snr_threshold',     default=0.0,         type='float',  metavar='snr_threshold',    help='SNR detection threshold.')
    parser.add_option('--em_selection',      default=0,           type='int',    metavar='em_selection',     help='Use EM selection function.')
    parser.add_option('--redshift_prior',    default=0,           type='int',    metavar='redshift_prior',   help='Adopt a redshift prior with comoving volume factor.')
    parser.add_option('--T',                 default=10,          type='float',  metavar='T',                help='Observation time')
    parser.add_option('--time_redshifting',  default=0,           type='int',    metavar='time_redshifting', help='Add a factor 1/1+z_gw as redshift prior.')
    parser.add_option('--vc_normalization',  default=0,           type='int',    metavar='vc_normalization', help='Add a covolume factor normalization to the redshift prior.')
    parser.add_option('--lk_sel_fun',        default=0,           type='int',    metavar='lk_sel_fun',       help='Single-event likelihood a la Jon Gair.')
    parser.add_option('--detection_corr',    default=0,           type='int',    metavar='detection_corr',   help='Single-event likelihood including detection correction')
    parser.add_option('--sfr',               default=0,           type='int',    metavar='sfr',              help='fit the sfr parameters too')
    parser.add_option('--approx_int',        default=0,           type='int',    metavar='approx_int',       help='Approximate the in-catalog weight with the selection function.')
    parser.add_option('--reduced_catalog',   default=0,           type='int',    metavar='reduced_catalog',  help='Select randomly only a fraction of the catalog.')
    parser.add_option('--luminosity',        default=0,           type='int',    metavar='luminosity',       help='infer also the luminosity function')
    parser.add_option('--m_threshold',       default=20,          type='float',  metavar='m_threshold',      help='apparent magnitude threshold')
    parser.add_option('--threads',           default=None,        type='int',    metavar='threads',          help='Number of threads (default = 1/core).')
    parser.add_option('--seed',              default=0,           type='int',    metavar='seed',             help='Random seed initialisation.')
    parser.add_option('--nlive',             default=5000,        type='int',    metavar='nlive',            help='Number of live points.')
    parser.add_option('--poolsize',          default=256,         type='int',    metavar='poolsize',         help='Poolsize for the samplers.')
    parser.add_option('--maxmcmc',           default=1000,        type='int',    metavar='maxmcmc',          help='Maximum number of mcmc steps.')
    parser.add_option('--postprocess',       default=0,           type='int',    metavar='postprocess',      help='Run only the postprocessing. It works only with reduced_catalog=0.')
    parser.add_option('--screen_output',     default=0,           type='int',    metavar='screen_output',    help='Print the output on screen or save it into a file.')
    (opts,args)=parser.parse_args()

    em_selection        = opts.em_selection
    dl_cutoff           = opts.dl_cutoff
    z_selection         = opts.z_selection
    snr_selection       = opts.snr_selection
    zhorizon            = opts.zhorizon
    snr_threshold       = opts.snr_threshold
    redshift_prior      = opts.redshift_prior
    time_redshifting    = opts.time_redshifting
    vc_normalization    = opts.vc_normalization
    max_hosts           = opts.max_hosts
    event_ID_list       = opts.event_ID_list
    one_host_selection  = opts.one_host_sel
    lk_sel_fun          = opts.lk_sel_fun
    detection_corr      = opts.detection_corr
    approx_int          = opts.approx_int
    model               = opts.model
    corrections         = opts.corrections
    joint               = opts.joint
    event_class         = opts.event_class
    reduced_catalog     = opts.reduced_catalog
    postprocess         = opts.postprocess
    screen_output       = opts.screen_output
    out_dir             = opts.out_dir
    sfr                 = opts.sfr
    T                   = opts.T
    luminosity_function = opts.luminosity
    m_threshold         = opts.m_threshold

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
        events = readdata.read_event(event_class, opts.data, opts.event)

    if ((event_class == "EMRI") or (event_class == "sBH")):
        if (snr_selection is not None):
            events = readdata.read_event(event_class, opts.data, None, snr_selection=snr_selection, one_host_selection=one_host_selection)
        elif (z_selection is not None):
            events = readdata.read_event(event_class, opts.data, None, z_selection=z_selection, one_host_selection=one_host_selection)
        elif (dl_cutoff > 0) and (',' not in zhorizon) and (zhorizon is '1000.0'):
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
        elif (zhorizon is not '1000.0'):
            events = readdata.read_event(event_class, opts.data, None, zhorizon=zhorizon, one_host_selection=one_host_selection)
        elif (max_hosts is not None):
            events = readdata.read_event(event_class, opts.data, None, max_hosts=max_hosts, one_host_selection=one_host_selection)
        elif (event_ID_list is not None):
            events = readdata.read_event(event_class, opts.data, None, event_ID_list=event_ID_list, one_host_selection=one_host_selection)
        elif (snr_threshold is not 0.0):
            if not reduced_catalog:
                events = readdata.read_event(event_class, opts.data, None, snr_threshold=snr_threshold, one_host_selection=one_host_selection)
            else:
                events = readdata.read_event(event_class, opts.data, None, one_host_selection=one_host_selection)
                # Draw a number of events in the 4-year scenario
                if (event_class == "sBH"):
                    N = np.int(np.random.poisson(len(events)*4./40.))
                elif (event_class == "EMRI"):
                    N = np.int(np.random.poisson(len(events)*4./10.))
                print("\nReduced number of events: {}".format(N))
                selected_events = []
                k = 0
                while k < N and not(len(events) == 0):
                    idx = np.random.randint(len(events))
                    selected_event = events.pop(idx)
                    print("Drawn event {0}: ID={1} - SNR={2:.2f}".format(k+1, str(selected_event.ID).ljust(3), selected_event.snr))
                    if snr_threshold > 0.0:
                        if selected_event.snr > snr_threshold:
                            print("Selected: ID={0} - SNR={1:.2f} > {2:.2f}".format(str(selected_event.ID).ljust(3), selected_event.snr, snr_threshold))
                            selected_events.append(selected_event)
                        else: pass
                        k += 1
                    else:
                        if selected_event.snr < abs(snr_threshold):
                            print("Selected: ID={0} - SNR={1:.2f} < {2:.2f}".format(str(selected_event.ID).ljust(3), selected_event.snr, snr_threshold))
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
    if event_class == 'MBH':
        events = sorted(events, key=lambda x: getattr(x, 'ID'))
        for e in events:
            print("ID: {}  |  z_host: {} |  dl: {} Mpc  |  sigmadl: {} Mpc  | hosts: {}".format(
            str(e.ID).ljust(3), str(e.potential_galaxy_hosts[0].redshift).ljust(8), 
            str(e.dl).ljust(9), str(e.sigma)[:6].ljust(7), str(len(e.potential_galaxy_hosts)).ljust(4)))
    else:
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
                          corrections,
                          em_selection        = em_selection,
                          snr_threshold       = snr_threshold,
                          z_threshold         = zhorizon,
                          event_class         = event_class,
                          redshift_prior      = redshift_prior,
                          time_redshifting    = time_redshifting,
                          vc_normalization    = vc_normalization,
                          lk_sel_fun          = lk_sel_fun,
                          detection_corr      = detection_corr,
                          approx_int          = approx_int,
                          dl_cutoff           = dl_cutoff,
                          sfr                 = sfr,
                          T                   = T,
                          luminosity_function = luminosity_function,
                          m_threshold         = m_threshold)

    #IMPROVEME: postprocess doesn't work when events are randomly selected, since 'events' in C are different from the ones read from chain.txt
    if (postprocess == 0):
        work=cpnest.CPNest(C,
                           verbose      = 2, # To plot prior&posterior: verbose = 3, or prior-sampling = True
                           poolsize     = opts.poolsize,
                           nthreads     = opts.threads,
                           nlive        = opts.nlive,
                           maxmcmc      = opts.maxmcmc,
                           output       = output,
                           nhamiltonian = 0,
                        #    nslice       = 0
                           )

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
    
    if ((event_class == "EMRI") or (event_class == "sBH")):
        if C.gw == 1:
            for e in C.data:
                fig = plt.figure()
                ax  = fig.add_subplot(111)
                z   = np.linspace(e.zmin,e.zmax, 100)

                if (em_selection):
                    ax2 = ax.twinx()
                    
                    if ("DE" in C.model): normalisation = matplotlib.colors.Normalize(vmin=np.min(x['w0']), vmax=np.max(x['w0']))
                    else:               normalisation = matplotlib.colors.Normalize(vmin=np.min(x['h']), vmax=np.max(x['h']))
                    # choose a colormap
                    c_m = matplotlib.cm.cool
                    # create a ScalarMappable and initialize a data structure
                    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=normalisation)
                    s_m.set_array([])
                    for i in range(x.shape[0])[::10]:
                        if ("LambdaCDM_h" in C.model): O = cs.CosmologicalParameters(x['h'][i],truths['om'],truths['ol'],truths['w0'],truths['w1'])
                        elif ("LambdaCDM" in C.model): O = cs.CosmologicalParameters(x['h'][i],x['om'][i],1.0-x['om'][i],truths['w0'],truths['w1'])
                        elif ("CLambdaCDM" in C.model): O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],truths['w0'],truths['w1'])
                        elif ("LambdaCDMDE" in C.model): O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],x['w0'][i],x['w1'][i])
                        elif ("DE" in C.model): O = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],x['w0'][i],x['w1'][i])
                        distances = np.array([O.LuminosityDistance(zi) for zi in z])
                        if ("DE" in C.model):  ax2.plot(z, [lk.em_selection_function(d) for d in distances], lw = 0.15, color=s_m.to_rgba(x['w0'][i]), alpha = 0.5)
                        else: ax2.plot(z, [lk.em_selection_function(d) for d in distances], lw = 0.15, color=s_m.to_rgba(x['h'][i]), alpha = 0.5)
                        O.DestroyCosmologicalParameters()
                    CB = plt.colorbar(s_m, orientation='vertical', pad=0.15)
                    if ("DE" in C.model): CB.set_label('w_0')
                    else: CB.set_label('h')
                    ax2.set_ylim(0.0, 1.0)
                    ax2.set_ylabel('selection function')

                # Plot the likelihood  
                distance_likelihood = []
                print("redshift plot of event", e.ID)
                for i in range(x.shape[0])[::10]:
                    if ("LambdaCDM_h" in C.model): O = cs.CosmologicalParameters(x['h'][i],truths['om'],truths['ol'],truths['w0'],truths['w1'])
                    elif ("LambdaCDM" in C.model): O = cs.CosmologicalParameters(x['h'][i],x['om'][i],1.0-x['om'][i],truths['w0'],truths['w1'])
                    elif ("CLambdaCDM" in C.model): O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],truths['w0'],truths['w1'])
                    elif ("LambdaCDMDE" in C.model): O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],x['w0'][i],x['w1'][i])
                    elif ("DE" in C.model): O = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],x['w0'][i],x['w1'][i])
                    # distance_likelihood.append(np.array([lk.logLikelihood_single_event(C.hosts[e.ID], e.dl, e.sigma, O, zi) for zi in z]))
                    distance_likelihood.append(np.array([-0.5*((O.LuminosityDistance(zi)-e.dl)/e.sigma)**2 for zi in z]))
                    O.DestroyCosmologicalParameters()
                distance_likelihood = np.exp(np.array(distance_likelihood))
                l,m,h = np.percentile(distance_likelihood,[5,50,95],axis = 0)
                ax2 = ax.twinx()
                ax2.plot(z, m, linestyle = 'dashed', color='k', lw=0.75)
                ax2.fill_between(z,l,h,facecolor='magenta', alpha=0.5)
                omega_truth = cs.CosmologicalParameters(truths['h'],
                                               truths['om'],
                                               truths['ol'],
                                               truths['w0'],
                                               truths['w1'])
                ax2.plot(z, np.exp(np.array([-0.5*((omega_truth.LuminosityDistance(zi)-e.dl)/e.sigma)**2 for zi in z])), linestyle = 'dashed', color='gold', lw=1.5)
                ax.axvline(lk.find_redshift(omega_truth,e.dl), linestyle='dotted', lw=0.8, color='red')
                omega_truth.DestroyCosmologicalParameters()
                ax.axvline(e.z_true, linestyle='dotted', lw=0.8, color='k')
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
            if ("LambdaCDM_h" in C.model):
                omega = cs.CosmologicalParameters(x['h'][k],
                                               truths['om'],
                                               truths['ol'],
                                               truths['w0'],
                                               truths['w1'])
            elif ("LambdaCDM_om" in C.model):
                omega = cs.CosmologicalParameters(truths['h'],
                                               x['om'][k],
                                               1.0-x['om'][k],
                                               truths['w0'],
                                               truths['w1'])                                               
            elif ("LambdaCDM" in C.model):
                omega = cs.CosmologicalParameters(x['h'][k],
                                               x['om'][k],
                                               1.0-x['om'][k],
                                               truths['w0'],
                                               truths['w1'])
            elif ("CLambdaCDM" in C.model):
                omega = cs.CosmologicalParameters(x['h'][k],
                                               x['om'][k],
                                               x['ol'][k],
                                               truths['w0'],
                                               truths['w1'])
            elif ("LambdaCDMDE" in C.model):
                omega = cs.CosmologicalParameters(x['h'][k],
                                               x['om'][k],
                                               x['ol'][k],
                                               x['w0'][k],
                                               x['w1'][k])
            elif ("DE" in C.model):
                omega = cs.CosmologicalParameters(truths['h'],
                                               truths['om'],
                                               truths['ol'],
                                               x['w0'][k],
                                               x['w1'][k])
            else:
                omega = cs.CosmologicalParameters(truths['h'],
                                               truths['om'],
                                               truths['ol'],
                                               truths['w0'],
                                               truths['w1'])
            models.append([omega.LuminosityDistance(zi)/1e3 for zi in redshift])
            omega.DestroyCosmologicalParameters()
        
        models = np.array(models)
        model2p5,model16,model50,model84,model97p5 = np.percentile(models,[2.5,16.0,50.0,84.0,97.5],axis = 0)
        
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
    
    if C.cosmology == 1:

        if ("LambdaCDM_h" in C.model):
            fig = plt.figure()
            plt.hist(x['h'], density=True, alpha = 1.0, histtype='step', edgecolor="black")
            plt.axvline(truths['h'], linestyle='dashed', color='r')
            quantiles = np.quantile(x['h'], [0.05, 0.5, 0.95])
            plt.title(r'$h = {med:.3f}({low:.3f},+{up:.3f})$'.format(med=quantiles[1], low=quantiles[0]-quantiles[1], up=quantiles[2]-quantiles[1]), size = 16)
            plt.xlabel(r'$h$')
            plt.savefig(os.path.join(output,'h_histogram.pdf'), bbox_inches='tight')

        if ("LambdaCDM_om" in C.model):
            fig = plt.figure()
            plt.hist(x['om'], density=True, alpha = 1.0, histtype='step', edgecolor="black")
            plt.axvline(truths['om'], linestyle='dashed', color='r')
            quantiles = np.quantile(x['om'], [0.05, 0.5, 0.95])
            plt.title(r'$\Omega_m = {med:.3f}({low:.3f},+{up:.3f})$'.format(med=quantiles[1], low=quantiles[0]-quantiles[1], up=quantiles[2]-quantiles[1]), size = 16)
            plt.xlabel(r'$\Omega_m$')
            plt.savefig(os.path.join(output,'om_histogram.pdf'), bbox_inches='tight')        

        if ("LambdaCDM" in C.model):
            samps = np.column_stack((x['h'],x['om']))
            fig = corner.corner(samps,
                   labels= [r'$h$',
                            r'$\Omega_m$'],
                   quantiles=[0.05, 0.5, 0.95],
                   show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                   use_math_text=True, truths=[truths['h'],truths['om']])
    #        axes = fig.get_axes()
    #        axes[0].set_xlim(0.69, 0.77)
    #        axes[2].set_xlim(0.69, 0.77)
    #        axes[3].set_xlim(0.04, 0.5)
    #        axes[2].set_ylim(0.04, 0.5)    

        if ("CLambdaCDM" in C.model):
            samps = np.column_stack((x['h'],x['om'],x['ol'],1.0-x['om']-x['ol']))
            fig = corner.corner(samps,
                   labels= [r'$h$',
                            r'$\Omega_m$',
                            r'$\Omega_\Lambda$',
                            r'$\Omega_k$'],
                   quantiles=[0.05, 0.5, 0.95],
                   show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                   use_math_text=True, truths=[truths['h'],truths['om'],truths['ol'],0.0])
                   
        if ("LambdaCDMDE" in C.model):
            samps = np.column_stack((x['h'],x['om'],x['ol'],x['w0'],x['w1']))
            fig = corner.corner(samps,
                            labels= [r'$h$',
                                     r'$\Omega_m$',
                                     r'$\Omega_\Lambda$',
                                     r'$w_0$',
                                     r'$w_a$'],
                            quantiles=[0.05, 0.5, 0.95],
                            show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                            use_math_text=True, truths=[truths['h'],truths['om'],truths['ol'],truths['w0'],truths['w1']])

        if ("DE" in C.model):
            samps = np.column_stack((x['w0'],x['w1']))
            fig = corner.corner(samps,
                            labels= [r'$w_0$',
                                     r'$w_a$'],
                            quantiles=[0.05, 0.5, 0.95],
                            show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                            use_math_text=True, truths=[truths['w0'],truths['w1']])
    #        axes = fig.get_axes()
    #        axes[0].set_xlim(-1.22, -0.53)
    #        axes[2].set_xlim(-1.22, -0.53)
    #        axes[3].set_xlim(-1.0, 1.0)
    #        axes[2].set_ylim(-1.0, 1.0)
        if(('LambdaCDM_h' not in C.model) and ('LambdaCDM_om' not in C.model)):
            fig.savefig(os.path.join(output,'corner_plot.pdf'), bbox_inches='tight')

    if ("Rate" in C.model):
        z   = np.linspace(0.0,C.z_threshold,100)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        sfr = []
        Rtot = np.zeros(x.shape[0],dtype=np.float64)
        selection_probability = np.zeros(x.shape[0],dtype=np.float64)
        for i in range(x.shape[0]):
            r0  = 10**x['log10r0'][i]
            W   = x['W'][i]
            Q   = x['Q'][i]
            R   = x['R'][i]
            if ("LambdaCDM" in C.model):
                O = cs.CosmologicalParameters(x['h'][i],x['om'][i],1.0-x['om'][i],truths['w0'],truths['w1'])
            elif ("CLambdaCDM" in C.model):
                O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],truths['w0'],truths['w1'])
            elif ("LambdaCDMDE" in C.model):
                O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],x['w0'][i],x['w1'][i])
            elif ("DE" in C.model):
                O = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],x['w0'][i],x['w1'][i])
            else:
                O = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],truths['w0'],truths['w1'])
            # compute the expected rate parameter integrated to the maximum redshift
            # this will also serve as normalisation constant for the individual dR/dz_i
            Rtot[i] = lk.integrated_rate(r0, W, R, Q, O, 0.0, C.z_threshold)
            selection_probability[i] = lk.gw_selection_probability_sfr(1e-5, C.z_threshold, r0, W, R, Q, C.snr_threshold, O, Rtot[i])
            v = np.array([cs.StarFormationDensity(zi, r0, W, R, Q)*O.UniformComovingVolumeDensity(zi)/Rtot[i] for zi in z])
#            ax.plot(z,v,color='k', linewidth=.3)
            sfr.append(v)

        Rtot_true = lk.integrated_rate(truths['r0'], truths['W'], truths['R'], truths['Q'], omega_true, 0.0, C.z_threshold)
        sfr   = np.array(sfr)
        sfr_true = np.array([cs.StarFormationDensity(zi, truths['r0'], truths['W'], truths['R'], truths['Q'])*omega_true.UniformComovingVolumeDensity(zi)/Rtot_true for zi in z])
        
        l,m,h = np.percentile(sfr,[5,50,95],axis=0)
        ax.plot(z,m,color='k', linewidth=.7)
        ax.fill_between(z,l,h,facecolor='lightgray')
        ax.plot(z,sfr_true,linestyle='dashed',color='red')
        ax.set_xlabel('redshift')
        ax.set_ylabel('$p(z|\Lambda,\Omega,I)$')
        fig.savefig(os.path.join(output,'redshift_distribution.pdf'), bbox_inches='tight')
        
        tmp = np.cumsum(sfr, axis = 1)*np.diff(z)[0]
        nevents         = Rtot[:,None]*C.T*tmp
        nevents_true    = Rtot_true*C.T*np.cumsum(sfr_true)*np.diff(z)[0]
        l,m,h = np.percentile(nevents,[5,50,95],axis=0)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(z,m,color='k', linewidth=.7)
        ax.fill_between(z,l,h,facecolor='lightgray')
        ax.plot(z,nevents_true, color='r', linestyle='dashed')
        plt.yscale('log')
        ax.set_xlabel('redshift z')
        ax.set_ylabel('$R(z_{max})\cdot T\cdot p(z|\Lambda,\Omega,I)$')
        plt.savefig(os.path.join(output,'number_of_events.pdf'), bbox_inches='tight')
        
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.hist(Rtot, bins = 100, histtype='step')
        ax.axvline(Rtot_true, linestyle='dashed', color='r')
        ax.set_xlabel('global rate')
        ax.set_ylabel('number')
        fig.savefig(os.path.join(output,'global_rate.pdf'), bbox_inches='tight')
        print('merger rate =', np.percentile(Rtot,[5,50,95]),'true = ',Rtot_true)
        
        true_selection_prob = lk.gw_selection_probability_sfr(1e-5, C.z_threshold, truths['r0'], truths['W'], truths['R'], truths['Q'], C.snr_threshold, omega_true, Rtot_true)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.hist(selection_probability, bins = 100, histtype='step')
        ax.axvline(true_selection_prob, linestyle='dashed', color='r')
        ax.set_xlabel('selection probability')
        ax.set_ylabel('number')
        fig.savefig(os.path.join(output,'selection_probability.pdf'), bbox_inches='tight')

        print('p_det =',np.percentile(selection_probability,[5,50,95]),'true = ',true_selection_prob)

        samps = np.column_stack((x['log10r0'],x['W'],x['R'],x['Q']))
        fig = corner.corner(samps,
                        labels= [r'$\log_{10} r_0$',
                                 r'$W$',
                                 r'$R$',
                                 r'$Q$'],
                        quantiles=[0.05, 0.5, 0.95],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        use_math_text=True, truths=[np.log10(truths['r0']),truths['W'],truths['R'],truths['Q']],
                        filename=os.path.join(output,'joint_rate_posterior.pdf'))
        fig.savefig(os.path.join(output,'joint_rate_posterior.pdf'), bbox_inches='tight')
    
    if ("Luminosity" in C.model):
        distributions = []
        Z   = np.linspace(0.0,C.z_threshold,100)
        M   = np.linspace(C.Mmin,C.Mmax,100)
        luminosity_function_0 = []
        luminosity_function_1 = []
        luminosity_function_2 = []
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        for i in range(x.shape[0]):
            sys.stderr.write("processing {0} out of {1}\r".format(i+1,x.shape[0]))
            phistar0            = x['phistar0'][i]
            phistar_exponent    = x['phistar_exponent'][i]
            Mstar0              = x['Mstar0'][i]
            Mstar_exponent      = x['Mstar_exponent'][i]
            alpha0              = x['alpha0'][i]
            alpha_exponent      = x['alpha_exponent'][i]
            if ("LambdaCDM" in C.model):
                O = cs.CosmologicalParameters(x['h'][i],x['om'][i],1.0-x['om'][i],truths['w0'],truths['w1'])
            elif ("CLambdaCDM" in C.model):
                O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],truths['w0'],truths['w1'])
            elif ("LambdaCDMDE" in C.model):
                O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],x['w0'][i],x['w1'][i])
            elif ("DE" in C.model):
                O = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],x['w0'][i],x['w1'][i])
            else:
                O = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],truths['w0'],truths['w1'])
            S = gal.GalaxyDistribution(O,
                                       phistar0,
                                       phistar_exponent,
                                       Mstar0,
                                       Mstar_exponent,
                                       alpha0,
                                       alpha_exponent,
                                       C.Mmin,
                                       C.Mmax,
                                       0.0,
                                       C.z_threshold,
                                       0.0,
                                       2.0*np.pi,
                                       -0.5*np.pi,
                                       0.5*np.pi,
                                       C.magnitude_threshold,
                                       4.0*np.pi,
                                       1,1,1)

            PMZ = np.array([S.pdf(Mi, Zj, 1) for Mi in M for Zj in Z]).reshape(100,100)
            distributions.append(PMZ)
            luminosity_function_0.append(np.array([S.luminosity_function(Mi, 1e-5, 0) for Mi in M]))
            luminosity_function_1.append(np.array([S.luminosity_function(Mi, S.zmax/2., 0) for Mi in M]))
            luminosity_function_2.append(np.array([S.luminosity_function(Mi, S.zmax, 0) for Mi in M]))

        sys.stderr.write("\n")
        distributions = np.array(distributions)
        pmzl,pmzm,pmzh = np.percentile(distributions,[5,50,95],axis=0)
        pl,pm,ph = np.percentile(luminosity_function_0,[5,50,95],axis=0)
        ax.fill_between(M,pl,ph,facecolor='magenta',alpha=0.5)
        ax.plot(M,pm,linestyle='dashed',color='r',label="z = 0.0")
        
        pl,pm,ph = np.percentile(luminosity_function_1,[5,50,95],axis=0)
        ax.fill_between(M,pl,ph,facecolor='green',alpha=0.5)
        ax.plot(M,pm,linestyle='dashed',color='g',label="z = {0:.1f}".format(S.zmax/2.))
        
        pl,pm,ph = np.percentile(luminosity_function_2,[5,50,95],axis=0)
        ax.fill_between(M,pl,ph,facecolor='turquoise',alpha=0.5)
        ax.plot(M,pm,linestyle='dashed',color='b',label="z = {0:.1f}".format(S.zmax))
        
        St = gal.GalaxyDistribution(cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],truths['w0'],truths['w1']),
                                    truths['phistar0'],
                                    truths['phistar_exponent'],
                                    truths['Mstar0'],
                                    truths['Mstar_exponent'],
                                    truths['alpha0'],
                                    truths['alpha_exponent'],
                                    C.Mmin,
                                    C.Mmax,
                                    0.0,
                                    C.z_threshold,
                                    0.0,
                                    2.0*np.pi,
                                    -0.5*np.pi,
                                    0.5*np.pi,
                                    C.magnitude_threshold,
                                    4.0*np.pi,
                                    1,1,1)
        
        ax.plot(M,np.array([St.luminosity_function(Mi, 1e-5, 0) for Mi in M]),linestyle='solid',color='k',lw=1.5,zorder=0)
        plt.legend(fancybox=True)
        ax.set_xlabel('magnitude')
        ax.set_ylabel('$\phi(M|\Omega,I)$')
        fig.savefig(os.path.join(output,'luminosity_function.pdf'), bbox_inches='tight')
        
        magnitude_probability = np.sum(pmzl*np.diff(Z)[0],axis=1),np.sum(pmzm*np.diff(Z)[0],axis=1),np.sum(pmzh*np.diff(Z)[0],axis=1)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.fill_between(M,magnitude_probability[0],magnitude_probability[2],facecolor='lightgray')
        ax.plot(M,magnitude_probability[1],linestyle='dashed',color='k')
        ax.hist(C.galaxy_magnitudes, 100, density = True, facecolor='turquoise')
        ax.set_xlabel('magnitude')
        ax.set_ylabel('$\phi(M|\Omega,I)$')
        fig.savefig(os.path.join(output,'luminosity_probability.pdf'), bbox_inches='tight')

        redshift_probability = np.sum(pmzl*np.diff(M)[0],axis=0),np.sum(pmzm*np.diff(M)[0],axis=0),np.sum(pmzh*np.diff(M)[0],axis=0)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.fill_between(Z,redshift_probability[0],redshift_probability[2],facecolor='lightgray')
        ax.plot(Z,redshift_probability[1],linestyle='dashed',color='k')
        ax.hist(C.galaxy_redshifts, 100, density = True, facecolor='turquoise')
        ax.set_xlabel('redshift')
        ax.set_ylabel('$\phi(z|\Omega,I)$')
        fig.savefig(os.path.join(output,'galaxy_redshift_probability.pdf'), bbox_inches='tight')
        
        samps = np.column_stack((x['phistar0'],x['phistar_exponent'],x['Mstar0'],x['Mstar_exponent'],x['alpha0'],x['alpha_exponent']))
        fig = corner.corner(samps,
                        labels= [r'$\phi^{*}/Mpc^{3}$',
                                 r'$a$',
                                 r'$M^{*}$',
                                 r'$b$',
                                 r'$\alpha$',
                                 r'$c$'],
                        quantiles=[0.05, 0.5, 0.95],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        use_math_text=True, truths=[truths['phistar0'],truths['phistar_exponent'],
                                                    truths['Mstar0'],truths['Mstar_exponent'],
                                                    truths['alpha0'],truths['alpha_exponent']],
                        filename=os.path.join(output,'joint_luminosity_posterior.pdf'))
        fig.savefig(os.path.join(output,'joint_luminosity_posterior.pdf'), bbox_inches='tight')
        
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

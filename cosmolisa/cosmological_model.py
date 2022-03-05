#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
import os
import ray
import time
import configparser
import subprocess
import numpy as np
import matplotlib
import corner
import matplotlib.pyplot as plt
from optparse import OptionParser
from configparser import ConfigParser
from scipy.stats import norm

# Import internal and external modules
from cosmolisa import readdata
from cosmolisa import cosmology as cs
from cosmolisa import likelihood as lk
from cosmolisa import galaxy as gal
import cpnest.model

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
        self.dl_cutoff           = kwargs['dl_cutoff']
        self.sfr                 = kwargs['sfr']
        self.T                   = kwargs['T']
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
            self.names.append('log10r0')
            self.bounds.append([-15,-8])
            self.names.append('W')
            self.bounds.append([0.0,300.0])
            self.names.append('Q')
            self.bounds.append([0.0,15.0])
            self.names.append('R')
            self.bounds.append([0.0,15.0])
            
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
               
        print("\n====================================================================================================\n")
        print("CosmologicalModel model initialised with:")
        print("Event class: {0}".format(self.event_class))
        print("Analysis model: {0}".format(self.model))
        print("Number of events: {0}".format(len(self.data)))
        print("EM correction: {0}".format(self.em_correction))
        print("GW correction: {0}".format(self.gw_correction))
        print("Free parameters: {0}".format(self.names))
        print("\n====================================================================================================\n")
        print("Prior bounds:")
        for name,bound in zip(self.names, self.bounds):
            print("{}: {}".format(str(name).ljust(4), bound))
        print("\n====================================================================================================\n")

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

#IMPROVEME: most of the options work only for EMRI and sBH. Extend to MBHB.
usage="""\n\n %prog --config-file config.ini\n
    ######################################################################################################################################################
    IMPORTANT: This code requires the installation of the CPNest branch 'massively_parallel': https://github.com/johnveitch/cpnest/tree/massively_parallel
    ######################################################################################################################################################

    #=======================#
    # Input parameters      #
    #=======================#

    'data'                        Default: ''.                      Data location.
    'outdir'                      Default: './default_dir'.         Directory for output.
    'event_class'                 Default: ''.                      Class of the event(s) ['MBHB', 'EMRI', 'sBH'].
    'model'                       Default: ''.                      Specify the cosmological model to assume for the analysis ['LambdaCDM', 'LambdaCDM_h', LambdaCDM_om, 'CLambdaCDM', 'LambdaCDMDE', 'DE'] and the type of analysis ['GW','Rate', 'Luminosity'] separated by a '+'.
    'corrections'                 Default: ''.                      Family of corrections ('GW', 'EM') separated by a '+'
    'joint'                       Default: 0.                       Run a joint analysis for N events, randomly selected.
    'zhorizon'                    Default: '1000.0'.                Impose low-high cutoffs in redshift. It can be a single number (upper limit) or a string with z_min and z_max separated by a comma.
    'dl_cutoff'                   Default: -1.0.                    Max EMRI dL(omega_true,zmax) allowed (in Mpc). This cutoff supersedes the zhorizon one.
    'z_event_sel'                 Default: 0.                       Select N events ordered by redshift. If positive (negative), choose the X nearest (farthest) events.
    'one_host_sel'                Default: 0.                       For each event, associate only the nearest-in-redshift host.
    'event_ID_list'               Default: ''.                      String of specific ID events to be read.
    'max_hosts'                   Default: 0.                       Select events according to the allowed maximum number of hosts.
    'z_gal_cosmo'                 Default: 0.                       If set to 1, read and use the cosmological redshift of the galaxies instead of the observed one.
    'snr_selection'               Default: 0.                       Select N events according to SNR (if N>0 the N loudest, if N<0 the N faintest).
    'snr_threshold'               Default: 0.0.                     Impose an SNR detection threshold X>0 (X<0) and select the events above (belove) X.
    'em_selection'                Default: 0.                       Use an EM selection function.
    'T'                           Default: 10.0.                    Observation time (yr).
    'sfr'                         Default: 0.                       Fit the star formation parameters too.
    'reduced_catalog'             Default: 0.                       Select randomly only a fraction of the catalog (4 yrs of observation, hardcoded).
    'm_threshold'                 Default: 20.                      Apparent magnitude threshold.
    'postprocess'                 Default: 0.                       Run only the postprocessing. It works only with reduced_catalog=0.
    'screen_output'               Default: 0.                       Print the output on screen or save it into a file.
    'verbose'                     Default: 2.                       Sampler verbose.
    'maxmcmc'                     Default: 5000.                    Maximum MCMC steps for MHS sampling chains.
    'nensemble'                   Default: 1.                       Number of sampler threads using an ensemble sampler. Equal to the number of LP evolved at each NS step. It must be a positive multiple of nnest.
    'nslice'                      Default: 0.                       Number of sampler threads using a slice sampler.
    'nhamiltonian'                Default: 0.                       Number of sampler threads using a hamiltonian sampler.
    'nnest'                       Default: 1.                       Number of parallel independent nested samplers.
    'nlive'                       Default: 1000.                    Number of live points.
    'seed'                        Default: 0.                       Random seed initialisation.
    'obj_store_mem'               Default: 2e9.                     Amount of memory reserved for ray object store. Default: 2GB.

"""

def main():

    run_time = time.perf_counter()
    parser = OptionParser(usage)
    parser.add_option('--config-file', type='string', metavar = 'config_file', default = None)

    (opts,args) = parser.parse_args()
    config_file = opts.config_file

    if not(config_file):
        parser.print_help()
        parser.error('Please specify a config file.')
    if not(os.path.exists(config_file)):
        parser.error('Config file {} not found.'.format(config_file))
    Config = configparser.ConfigParser()
    Config.read(config_file)

    config_par = {
                'data'                      :  '',
                'outdir'                    :  './default_dir',
                'event_class'               :  '',
                'model'                     :  '',
                'corrections'               :  '',
                'joint'                     :  0,
                'zhorizon'                  :  '1000.0',
                'dl_cutoff'                 :  -1.0,
                'z_event_sel'               :  0,
                'one_host_sel'              :  0,
                'event_ID_list'             :  '',
                'max_hosts'                 :  0,
                'z_gal_cosmo'               :  0,
                'snr_selection'             :  0,
                'snr_threshold'             :  0.0,
                'em_selection'              :  0,
                'T'                         :  10.,
                'sfr'                       :  0,
                'reduced_catalog'           :  0,
                'm_threshold'               :  20,
                'postprocess'               :  0,
                'screen_output'             :  0,    
                'verbose'                   :  2,
                'maxmcmc'                   :  5000,
                'nensemble'                 :  1,
                'nslice'                    :  0,
                'nhamiltonian'              :  0,
                'nnest'                     :  1,
                'nlive'                     :  1000,
                'seed'                      :  0,
                'obj_store_mem'             :  2e9,
                }

    for key in config_par:
        keytype = type(config_par[key])
        try: 
            config_par[key]=keytype(Config.get("input parameters",key))
        except (KeyError, configparser.NoOptionError, TypeError):
            pass

    try:
        outdir = str(config_par['outdir'])
    except(KeyError, ValueError):
        outdir = 'default_dir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    os.system('mkdir -p {}/CPNest'.format(outdir))
    os.system('mkdir -p {}/Plots'.format(outdir))
    #FIXME: avoid cp command when reading the config file from the outdir directory to avoid the 'same file' cp error
    os.system('cp {} {}/.'.format(opts.config_file, outdir))
    output_sampler = os.path.join(outdir,'CPNest')

    if not(config_par['screen_output']):
        if not(config_par['postprocess']):
            sys.stdout = open(os.path.join(outdir,'stdout.txt'), 'w')
            sys.stderr = open(os.path.join(outdir,'stderr.txt'), 'w')

    print("\n"+"cpnest installation version:", cpnest.__version__)
    print("ray version:", ray.__version__)

    max_len_keyword = len('reduced_catalog')
    print(('\nReading config file: {}\n'.format(config_file)))
    for key in config_par:
        print(("{name} : {value}".format(name=key.ljust(max_len_keyword), value=config_par[key])))


    omega_true = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],truths['w0'],truths['w1'])

    formatting_string = "===================================================================================================="

    if (config_par['event_class'] == "MBHB"):
        # If running on MBHB, override the selection functions
        em_selection = 0
        events = readdata.read_event(config_par['event_class'], config_par['data'])

    if ((config_par['event_class'] == "EMRI") or (config_par['event_class'] == "sBH")):
        if (config_par['snr_selection'] != 0):
            events = readdata.read_event(config_par['event_class'], config_par['data'], None, snr_selection=config_par['snr_selection'], one_host_selection=config_par['one_host_sel'], z_gal_cosmo=config_par['z_gal_cosmo'])
        elif (config_par['z_event_sel'] != 0):
            events = readdata.read_event(config_par['event_class'], config_par['data'], None, z_event_sel=config_par['z_event_sel'], one_host_selection=config_par['one_host_sel'], z_gal_cosmo=config_par['z_gal_cosmo'])
        elif (config_par['dl_cutoff'] > 0) and (',' not in config_par['zhorizon']) and (config_par['zhorizon'] == '1000.0'):
            all_events = readdata.read_event(config_par['event_class'], config_par['data'], None, one_host_selection=config_par['one_host_sel'], z_gal_cosmo=config_par['z_gal_cosmo'])
            events_selected = []
            print("\nSelecting events according to dl_cutoff={}:".format(config_par['dl_cutoff']))
            for e in all_events:
                if (omega_true.LuminosityDistance(e.zmax) < config_par['dl_cutoff']):
                    events_selected.append(e)
                    print("Event {} selected: dl(z_max)={}.".format(str(e.ID).ljust(3), omega_true.LuminosityDistance(e.zmax)))
            events = sorted(events_selected, key=lambda x: getattr(x, 'dl'))
            print("\nSelected {} events from dl={} to dl={}:".format(len(events), events[0].dl, events[len(events)-1].dl))            
            for e in events:
                print("ID: {}  |  dl: {}".format(str(e.ID).ljust(3), str(e.dl).ljust(9)))     
        elif ((config_par['zhorizon'] != '1000.0') and (config_par['snr_threshold'] == 0.0)):
            events = readdata.read_event(config_par['event_class'], config_par['data'], None, zhorizon=config_par['zhorizon'], one_host_selection=config_par['one_host_sel'], z_gal_cosmo=config_par['z_gal_cosmo'])
        elif (config_par['max_hosts'] != 0):
            events = readdata.read_event(config_par['event_class'], config_par['data'], None, max_hosts=config_par['max_hosts'], one_host_selection=config_par['one_host_sel'], z_gal_cosmo=config_par['z_gal_cosmo'])
        elif (config_par['event_ID_list'] != ''):
            events = readdata.read_event(config_par['event_class'], config_par['data'], None, event_ID_list=config_par['event_ID_list'], one_host_selection=config_par['one_host_sel'], z_gal_cosmo=config_par['z_gal_cosmo'])
        elif (config_par['snr_threshold'] != 0.0):
            print("\nSelecting events according to snr_threshold={}:".format(config_par['snr_threshold']))
            if not config_par['reduced_catalog']:
                events = readdata.read_event(config_par['event_class'], config_par['data'], None, snr_threshold=config_par['snr_threshold'], one_host_selection=config_par['one_host_sel'], z_gal_cosmo=config_par['z_gal_cosmo'])
            else:
                events = readdata.read_event(config_par['event_class'], config_par['data'], None, one_host_selection=config_par['one_host_sel'], z_gal_cosmo=config_par['z_gal_cosmo'])
                # Draw a number of events in the 4-year scenario
                if (config_par['event_class'] == "sBH"):
                    N = np.int(np.random.poisson(len(events)*4./10.))
                elif (config_par['event_class'] == "EMRI"):
                    N = np.int(np.random.poisson(len(events)*4./10.))
                print("\nReduced number of events: {}".format(N))
                selected_events = []
                k = 0
                while k < N and not(len(events) == 0):
                    idx = np.random.randint(len(events))
                    selected_event = events.pop(idx)
                    print("Drawn event {0}: ID={1} - SNR={2:.2f}".format(k+1, str(selected_event.ID).ljust(3), selected_event.snr))
                    if config_par['snr_threshold'] > 0.0:
                        if selected_event.snr > config_par['snr_threshold']:
                            print("Selected: ID={0} - SNR={1:.2f} > {2:.2f}".format(str(selected_event.ID).ljust(3), selected_event.snr, config_par['snr_threshold']))
                            selected_events.append(selected_event)
                        else: pass
                        k += 1
                    else:
                        if selected_event.snr < abs(config_par['snr_threshold']):
                            print("Selected: ID={0} - SNR={1:.2f} < {2:.2f}".format(str(selected_event.ID).ljust(3), selected_event.snr, config_par['snr_threshold']))
                            selected_events.append(selected_event)
                        else: pass
                        k += 1                        
                events = selected_events
                events = sorted(selected_events, key=lambda x: getattr(x, 'snr'))
                print("\nSelected {} events from SNR={} to SNR={}:".format(len(events), events[0].snr, events[len(events)-1].snr))
                for e in events:
                    print("ID: {}  |  dl: {}".format(str(e.ID).ljust(3), str(e.dl).ljust(9)))
        else:
            events = readdata.read_event(config_par['event_class'], config_par['data'], None, one_host_selection=config_par['one_host_sel'], z_gal_cosmo=config_par['z_gal_cosmo'])

        if (config_par['joint'] != 0):
            N = joint
            if (N > len(events)):
                N = len(events)
                print("The catalog has a number of selected events smaller than the chosen number ({}). Running on {}".format(N, len(events)))
            events = np.random.choice(events, size = N, replace = False)
            print(formatting_string)
            print("Selecting a random catalog of {0} events for joint analysis:".format(N))
            print(formatting_string)
            if not(len(events) == 0):
                for e in events:
                    print("event {0}: distance {1} \pm {2} Mpc, z \in [{3},{4}] galaxies {5}".format(e.ID,e.dl,e.sigma,e.zmin,e.zmax,len(e.potential_galaxy_hosts)))
                print(formatting_string)
            else:
                print("None of the drawn events has z<{0}. No data to analyse. Exiting.\n".format(config_par['zhorizon']))
                exit()
    else:
        events = readdata.read_event(config_par['event_class'], config_par['data'], z_gal_cosmo=config_par['z_gal_cosmo'])

    if (len(events) == 0):
        print("The passed catalog is empty. Exiting.\n")
        exit()

    print("\nDetailed list of the %d selected event(s):"%len(events))
    print("\n"+formatting_string)
    if config_par['event_class'] == 'MBHB':
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


    print(formatting_string+"\n")
    print("CPNest will be initialised with:")
    print("verbose:             {0}".format(config_par['verbose']))
    print("nensemble:           {0}".format(config_par['nensemble']))
    print("nslice:              {0}".format(config_par['nslice']))
    print("nhamiltonian:        {0}".format(config_par['nhamiltonian']))
    print("nnest:               {0}".format(config_par['nnest']))
    print("nlive:               {0}".format(config_par['nlive']))
    print("maxmcmc:             {0}".format(config_par['maxmcmc']))
    print("object_store_memory: {0}".format(config_par['obj_store_mem']))

    C = CosmologicalModel(model               = config_par['model'],
                          data                = events,
                          corrections         = config_par['corrections'],
                          em_selection        = config_par['em_selection'],
                          snr_threshold       = config_par['snr_threshold'],
                          z_threshold         = float(config_par['zhorizon']),
                          event_class         = config_par['event_class'],
                          dl_cutoff           = config_par['dl_cutoff'],
                          sfr                 = config_par['sfr'],
                          T                   = config_par['T'],
                          m_threshold         = config_par['m_threshold']
                          )

    #IMPROVEME: postprocess doesn't work when events are randomly selected, since 'events' in C are different from the ones read from chain.txt
    if (config_par['postprocess'] == 0):
        # Each NS can be located in different processors, but all the subprocesses of each NS live on the same processor

        work=cpnest.CPNest(C,
                           verbose             = config_par['verbose'],
                           maxmcmc             = config_par['maxmcmc'],
                           nensemble           = config_par['nensemble'],
                           nslice              = config_par['nslice'],
                           nhamiltonian        = config_par['nhamiltonian'],
                           nnest               = config_par['nnest'],   
                           nlive               = config_par['nlive'],  
                           object_store_memory = config_par['obj_store_mem'],
                           output              = output_sampler
                           )

        work.run()
        print('log Evidence {0}'.format(work.logZ))
        print("\n"+formatting_string+"\n")

        x = work.posterior_samples.ravel()

        ray.shutdown()
        # Save git info
        with open("{}/git_info.txt".format(outdir), "w+") as fileout:
            subprocess.call(["git", "diff"], stdout=fileout)
    else:
        print("Reading the .h5 file...")
        import h5py
        filename = os.path.join(outdir,'CPNest','cpnest.h5')
        h5_file = h5py.File(filename,'r')
        x = h5_file['combined'].get('posterior_samples')

    ############################################################################################################
    #######################################          MAKE PLOTS         ########################################
    ############################################################################################################

    if ((config_par['event_class'] == "EMRI") or (config_par['event_class'] == "sBH")):
        if C.gw == 1:
            for e in C.data:
                fig = plt.figure()
                ax  = fig.add_subplot(111)
                z   = np.linspace(e.zmin,e.zmax, 100)

                if (config_par['em_selection']):
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
                print("Making redshift plot of event", e.ID)
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
                plt.savefig(os.path.join(outdir,'Plots','redshift_%d'%e.ID+'.png'), bbox_inches='tight')
                plt.close()
    
    if (config_par['event_class'] == "MBHB"):
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
        fig.savefig(os.path.join(outdir,'Plots','regression.pdf'),bbox_inches='tight')
        plt.close()
    
    if C.cosmology == 1:

        if ("LambdaCDM_h" in C.model):
            # plots.par_histogram(x['h'], 'h', truths['h'], outdir)
            fig = plt.figure()
            plt.hist(x['h'], density=True, alpha = 1.0, histtype='step', edgecolor="black")
            plt.axvline(truths['h'], linestyle='dashed', color='r')
            quantiles = np.quantile(x['h'], [0.05, 0.5, 0.95])
            plt.title(r'$h = {med:.3f}({low:.3f},+{up:.3f})$'.format(med=quantiles[1], low=quantiles[0]-quantiles[1], up=quantiles[2]-quantiles[1]), size = 16)
            plt.xlabel(r'$h$')
            plt.savefig(os.path.join(outdir,'Plots','h_histogram.pdf'), bbox_inches='tight')

        if ("LambdaCDM_om" in C.model):
            # plots.par_histogram(x['om'], '\Omega_m', truths['om'], outdir)
            fig = plt.figure()
            plt.hist(x['om'], density=True, alpha = 1.0, histtype='step', edgecolor="black")
            plt.axvline(truths['om'], linestyle='dashed', color='r')
            quantiles = np.quantile(x['om'], [0.05, 0.5, 0.95])
            plt.title(r'$\Omega_m = {med:.3f}({low:.3f},+{up:.3f})$'.format(med=quantiles[1], low=quantiles[0]-quantiles[1], up=quantiles[2]-quantiles[1]), size = 16)
            plt.xlabel(r'$\Omega_m$')
            plt.savefig(os.path.join(outdir,'Plots','om_histogram.pdf'), bbox_inches='tight')        

        if ("LambdaCDM" in C.model):
            samps = np.column_stack((x['h'],x['om']))
            fig = corner.corner(samps,
                   labels= [r'$h$',
                            r'$\Omega_m$'],
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                   use_math_text=True, truths=[truths['h'],truths['om']])

        if ("CLambdaCDM" in C.model):
            samps = np.column_stack((x['h'],x['om'],x['ol'],1.0-x['om']-x['ol']))
            fig = corner.corner(samps,
                   labels= [r'$h$',
                            r'$\Omega_m$',
                            r'$\Omega_\Lambda$',
                            r'$\Omega_k$'],
                   quantiles=[0.16, 0.5, 0.84],
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
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                            use_math_text=True, truths=[truths['h'],truths['om'],truths['ol'],truths['w0'],truths['w1']])

        if ("DE" in C.model):
            samps = np.column_stack((x['w0'],x['w1']))
            fig = corner.corner(samps,
                            labels= [r'$w_0$',
                                     r'$w_a$'],
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                            use_math_text=True, truths=[truths['w0'],truths['w1']])

        if(('LambdaCDM_h' not in C.model) and ('LambdaCDM_om' not in C.model)):
            fig.savefig(os.path.join(outdir,'Plots','corner_plot.pdf'), bbox_inches='tight')

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
        fig.savefig(os.path.join(outdir,'Plots','redshift_distribution.pdf'), bbox_inches='tight')
        
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
        plt.savefig(os.path.join(outdir,'Plots','number_of_events.pdf'), bbox_inches='tight')
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.hist(Rtot, bins = 100, histtype='step')
        ax.axvline(Rtot_true, linestyle='dashed', color='r')
        ax.set_xlabel('global rate')
        ax.set_ylabel('number')
        fig.savefig(os.path.join(outdir,'Plots','global_rate.pdf'), bbox_inches='tight')
        print('merger rate =', np.percentile(Rtot,[5,50,95]),'true = ',Rtot_true)
        
        true_selection_prob = lk.gw_selection_probability_sfr(1e-5, C.z_threshold, truths['r0'], truths['W'], truths['R'], truths['Q'], C.snr_threshold, omega_true, Rtot_true)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.hist(selection_probability, bins = 100, histtype='step')
        ax.axvline(true_selection_prob, linestyle='dashed', color='r')
        ax.set_xlabel('selection probability')
        ax.set_ylabel('number')
        fig.savefig(os.path.join(outdir,'Plots','selection_probability.pdf'), bbox_inches='tight')

        print('p_det =',np.percentile(selection_probability,[5,50,95]),'true = ',true_selection_prob)

        samps = np.column_stack((x['h'],x['om'],x['log10r0'],x['W'],x['R'],x['Q']))
        fig = corner.corner(samps,
                        labels= [r'$h$',
                                 r'$\Omega_m$',
                                 r'$\log_{10} r_0$',
                                 r'$W$',
                                 r'$R$',
                                 r'$Q$'],
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 12},
                        use_math_text=True, #truths=[np.log10(truths['r0']),truths['W'],truths['R'],truths['Q']],
                        filename=os.path.join(outdir,'Plots','joint_rate_posterior.pdf'))
        fig.savefig(os.path.join(outdir,'Plots','corner_plot_rate.pdf'), bbox_inches='tight')
    
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
            sys.stderr.write("Processing {0} out of {1}\r".format(i+1,x.shape[0]))
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
        fig.savefig(os.path.join(outdir,'Plots','luminosity_function.pdf'), bbox_inches='tight')
        
        magnitude_probability = np.sum(pmzl*np.diff(Z)[0],axis=1),np.sum(pmzm*np.diff(Z)[0],axis=1),np.sum(pmzh*np.diff(Z)[0],axis=1)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.fill_between(M,magnitude_probability[0],magnitude_probability[2],facecolor='lightgray')
        ax.plot(M,magnitude_probability[1],linestyle='dashed',color='k')
        ax.hist(C.galaxy_magnitudes, 100, density = True, facecolor='turquoise')
        ax.set_xlabel('magnitude')
        ax.set_ylabel('$\phi(M|\Omega,I)$')
        fig.savefig(os.path.join(outdir,'Plots','luminosity_probability.pdf'), bbox_inches='tight')

        redshift_probability = np.sum(pmzl*np.diff(M)[0],axis=0),np.sum(pmzm*np.diff(M)[0],axis=0),np.sum(pmzh*np.diff(M)[0],axis=0)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.fill_between(Z,redshift_probability[0],redshift_probability[2],facecolor='lightgray')
        ax.plot(Z,redshift_probability[1],linestyle='dashed',color='k')
        ax.hist(C.galaxy_redshifts, 100, density = True, facecolor='turquoise')
        ax.set_xlabel('redshift')
        ax.set_ylabel('$\phi(z|\Omega,I)$')
        fig.savefig(os.path.join(outdir,'Plots','galaxy_redshift_probability.pdf'), bbox_inches='tight')
        
        samps = np.column_stack((x['phistar0'],x['phistar_exponent'],x['Mstar0'],x['Mstar_exponent'],x['alpha0'],x['alpha_exponent']))
        fig = corner.corner(samps,
                        labels= [r'$\phi^{*}/Mpc^{3}$',
                                 r'$a$',
                                 r'$M^{*}$',
                                 r'$b$',
                                 r'$\alpha$',
                                 r'$c$'],
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        use_math_text=True, truths=[truths['phistar0'],truths['phistar_exponent'],
                                                    truths['Mstar0'],truths['Mstar_exponent'],
                                                    truths['alpha0'],truths['alpha_exponent']],
                        filename=os.path.join(outdir,'Plots','joint_luminosity_posterior.pdf'))
        fig.savefig(os.path.join(outdir,'Plots','joint_luminosity_posterior.pdf'), bbox_inches='tight')

    if (config_par['postprocess'] == 0):
        run_time = (time.perf_counter() - run_time)/60.0
        print('\nRun-time (min): {:.2f}\n'.format(run_time))


if __name__=='__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from optparse import OptionParser
from scipy.special import logsumexp
from functools import reduce
from scipy.stats import norm
import unittest
import lal
import cpnest.model
import sys
import os
import readdata
import matplotlib
import corner
import subprocess
import itertools as it
import cosmolisa.cosmology as cs
import numpy as np
import cosmolisa.likelihood as lk
import cosmolisa.prior as pr
import matplotlib.pyplot as plt


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
    bounds = [] #[0.5,1.0],[0.04,1.0],[0.0,1.0],[-2.0,0.0],[-3.0,3.0]]
    
    def __init__(self, model, data, *args, **kwargs):

        super(CosmologicalModel,self).__init__()
        # Set up the data
        self.data           = data
        self.N              = len(self.data)
        self.model          = model
        self.em_selection   = kwargs['em_selection']
        self.z_threshold    = kwargs['z_threshold']
        self.snr_threshold  = kwargs['snr_threshold']
        self.event_class    = kwargs['event_class']
        self.O              = None
        
        if self.model == "LambdaCDM":
            
            self.names  = ['h','om']
            self.bounds = [[0.5,1.0],[0.04,0.5]]
        
        elif self.model == "LambdaCDMDE":
            
            self.names  = ['h','om','ol','w0','w1']
            self.bounds = [[0.5,1.0],[0.04,0.5],[0.0,1.0],[-2.0,0.0],[-3.0,3.0]]
            
        elif self.model == "CLambdaCDM":
            
            self.names  = ['h','om','ol']
            self.bounds = [[0.5,1.0],[0.04,0.5],[0.0,1.0]]
            
        elif self.model == "DE":
            
            self.names  = ['w0','w1']
            self.bounds = [[-3.0,-0.3],[-1.0,1.0]]
        
        else:
            
            print("Cosmological model %s not supported. Exiting...\n"%self.model)
            exit()
        
        for e in self.data:
            self.bounds.append([e.zmin,e.zmax])
            self.names.append('z%d'%e.ID)
            
        self._initialise_galaxy_hosts()
        
        print("==================================================")
        print("cpnest model initialised with:")
        print("Cosmological model: {0}".format(self.model))
        print("Number of events: {0}".format(len(self.data)))
        print("EM correction: {0}".format(self.em_selection))
        print("==================================================")

    def _initialise_galaxy_hosts(self):
        self.hosts = {e.ID:np.array([(g.redshift,g.dredshift,g.weight) for g in e.potential_galaxy_hosts]) for e in self.data}

    def log_prior(self,x):
    
        logP = super(CosmologicalModel,self).log_prior(x)
        
        if np.isfinite(logP):
            """
            apply a uniform in comoving volume density redshift prior
            """
            if self.model == "LambdaCDM":

                self.O = cs.CosmologicalParameters(x['h'],x['om'],1.0-x['om'],-1.0,0.0)

            elif self.model == "CLambdaCDM":

                self.O = cs.CosmologicalParameters(x['h'],x['om'],x['ol'],-1.0,0.0)

            elif self.model == "LambdaCDMDE":

                self.O = cs.CosmologicalParameters(x['h'],x['om'],x['ol'],x['w0'],x['w1'])
            
            elif self.model == "DE":

                self.O = cs.CosmologicalParameters(0.73,0.25,0.75,x['w0'],x['w1'])
            
            log_norm = np.log(self.O.IntegrateComovingVolumeDensity(self.z_threshold))
            logP += np.sum([pr.logprior_redshift_single_event(self.O, x['z%d'%e.ID], log_norm) for e in self.data])

        return logP

    def log_likelihood(self,x):
        
        # Compute sum_GW p(z_gw|...)*p(dL|...)*p(Di|...)
        logL = np.sum([lk.logLikelihood_single_event(self.hosts[e.ID], e.dl, e.sigma,
                       self.O, x['z%d'%e.ID], em_selection = self.em_selection, zmin = self.bounds[2+j][0], zmax = self.bounds[2+j][1]) for j,e in enumerate(self.data)])

        self.O.DestroyCosmologicalParameters()

        return logL

truths = {'h':0.73,'om':0.25,'ol':0.75,'w0':-1.0,'w1':0.0}
usage=""" %prog (options)"""

if __name__=='__main__':

    parser = OptionParser(usage)
    parser.add_option('-d', '--data',        default=None,        type='string', metavar='data',            help='Galaxy data location')
    parser.add_option('-o', '--out-dir',     default=None,        type='string', metavar='DIR',             help='Directory for output')
    parser.add_option('-c', '--event-class', default=None,        type='string', metavar='event_class',     help='Class of the event(s) [MBH, EMRI, sBH]')
    parser.add_option('-e', '--event',       default=None,        type='int',    metavar='event',           help='Event number')
    parser.add_option('-m', '--model',       default='LambdaCDM', type='string', metavar='model',           help='Cosmological model to assume for the analysis (default LambdaCDM). Supports LambdaCDM, CLambdaCDM, LambdaCDMDE, and DE.')
    parser.add_option('-j', '--joint',       default=0,           type='int',    metavar='joint',           help='Run a joint analysis for N events, randomly selected (EMRI only).')
    parser.add_option('-z', '--zhorizon',    default=1000.0,      type='float',  metavar='zhorizon',        help='Horizon redshift corresponding to the SNR threshold')
    parser.add_option('--snr_threshold',     default=0.0,         type='float',  metavar='snr_threshold',   help='SNR detection threshold')
    parser.add_option('--em_selection',      default=0,           type='int',    metavar='em_selection',    help='Use EM selection function')
    parser.add_option('--reduced_catalog',   default=0,           type='int',    metavar='reduced_catalog', help='Select randomly only a fraction of the catalog')
    parser.add_option('-t', '--threads',     default=None,        type='int',    metavar='threads',         help='Number of threads (default = 1/core)')
    parser.add_option('-s', '--seed',        default=0,           type='int',    metavar='seed',            help='Random seed initialisation')
    parser.add_option('--nlive',             default=1000,        type='int',    metavar='nlive',           help='Number of live points')
    parser.add_option('--poolsize',          default=100,         type='int',    metavar='poolsize',        help='Poolsize for the samplers')
    parser.add_option('--maxmcmc',           default=1000,        type='int',    metavar='maxmcmc',         help='Maximum number of mcmc steps')
    parser.add_option('--postprocess',       default=0,           type='int',    metavar='postprocess',     help='Run only the postprocessing. It works only with reduced_catalog=0')
    (opts,args)=parser.parse_args()
    
    em_selection = opts.em_selection

    if opts.event_class == "MBH":
        # if running on SMBH override the selection functions
        em_selection = 0

    if opts.event_class == "EMRI" and opts.joint !=0:
#        np.random.seed(opts.seed)
        events = readdata.read_event(opts.event_class, opts.data, None)
        if len(events) == 0:
            print("The passed catalog is empty.\n")
            exit()
        if opts.reduced_catalog != 0:
            N = np.int(np.random.poisson(len(events)*4./10.))
        else:
            N = opts.joint
        if N > len(events):
            N = len(events)
        print("==================================================")
        print("Will select a random catalog of (max) {0} events for joint analysis:".format(N))
        print("==================================================")
        selected_events = []
        if opts.reduced_catalog == 0:
            while len(selected_events) < N and not(len(events) == 0):
                while True:
                    if len(events) > 0:
                        idx = np.random.randint(len(events))
                        selected_event = events.pop(idx)
                    else:
                        break
                    if selected_event.z_true < opts.zhorizon:
                        print("Drawn event: {0} - True redshift: z={1:.2f}".format(str(selected_event.ID).ljust(3), selected_event.z_true))
                        selected_events.append(selected_event)
                        break
            print("==================================================")
            print("After z-selection (z<{0}), will run a joint analysis on {1} out of {2} random selected events:".format(opts.zhorizon, len(selected_events),N))
        else:
            k = 0
            while k < N and not(len(events) == 0):
                idx = np.random.randint(len(events))
                selected_event = events.pop(idx)
                print("Drawn event: {0} - True redshift: z={1:.2f}".format(str(selected_event.ID).ljust(3), selected_event.z_true))
                if selected_event.z_true < opts.zhorizon:
                    selected_events.append(selected_event)
                else: pass
                k += 1
            print("==================================================")
            print("After catalog reduction and z-selection (z<{0}), will run a joint analysis on {1} out of {2} randomly selected events:".format(opts.zhorizon, len(selected_events),N))
#        else:
#            events = np.random.choice(events, size = N, replace = False)
#            print("z-selection set by default to {0}. Will run a joint analysis on {1} random selected events:".format(opts.zhorizon, N))
        print("==================================================")
        events = np.copy(selected_events)
        if not(len(events) == 0):
            for e in events:
                print("event {0}: distance {1} \pm {2} Mpc, z \in [{3},{4}] galaxies {5}".format(e.ID,e.dl,e.sigma,e.zmin,e.zmax,len(e.potential_galaxy_hosts)))
            print("==================================================")
        else:
            print("None of the drawn events has z<{0}. No data to analyse.\n".format(opts.zhorizon))
            exit()
    else:
        events = readdata.read_event(opts.event_class, opts.data, opts.event)

    model = opts.model

    if opts.out_dir is None:
        output = opts.data+"/EVENT_1%03d/"%(opts.event+1)
    else:
        output = opts.out_dir
    
    C = CosmologicalModel(model,
                          events,
                          em_selection  = em_selection,
                          snr_threshold = opts.snr_threshold,
                          z_threshold   = opts.zhorizon,
                          event_class   = opts.event_class)

    #FIXME: postprocess doesn't work when events are randomly selected, since 'events' in C are different from the ones read in the chain.txt file
    if opts.postprocess == 0:
        work=cpnest.CPNest(C,
                           verbose      = 2,
                           poolsize     = opts.poolsize,
                           nthreads     = opts.threads,
                           nlive        = opts.nlive,
                           maxmcmc      = opts.maxmcmc,
                           output       = output,
                           nhamiltonian = 0)

        work.run()
        print('log Evidence {0}'.format(work.NS.logZ))
        x = work.posterior_samples.ravel()

        # Save git info
        with open("{}/git_info.txt".format(opts.out_dir), "w+") as fileout:
            subprocess.call(["git", "diff"], stdout=fileout);

    else:
        x = np.genfromtxt(os.path.join(output,"chain_"+str(opts.nlive)+"_1234.txt"), names=True)
        from cpnest import nest2pos
        x = nest2pos.draw_posterior_many([x], [opts.nlive], verbose=False)

    # Make plots
    if opts.event_class == "EMRI":
        for e in C.data:
            fig = plt.figure()
            ax  = fig.add_subplot(111)
            z   = np.linspace(e.zmin,e.zmax, 100)

            if em_selection:
                ax2 = ax.twinx()
                
                if model == "DE": normalisation = matplotlib.colors.Normalize(vmin=np.min(x['w0']), vmax=np.max(x['w0']))
                else:             normalisation = matplotlib.colors.Normalize(vmin=np.min(x['h']), vmax=np.max(x['h']))
                # choose a colormap
                c_m = matplotlib.cm.cool
                # create a ScalarMappable and initialize a data structure
                s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=normalisation)
                s_m.set_array([])
                for i in range(x.shape[0])[::10]:
                    if model == "LambdaCDM": O = cs.CosmologicalParameters(x['h'][i],x['om'][i],1.0-x['om'][i],truths['w0'],truths['w1'])
                    elif model == "CLambdaCDM": O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],truths['w0'],truths['w1'])
                    elif model == "LambdaCDMDE": O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],x['w0'][i],x['w1'][i])
                    elif model == "DE": O = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],x['w0'][i],x['w1'][i])
                    distances = np.array([O.LuminosityDistance(zi) for zi in z])
                    if model == "DE":  ax2.plot(z, [lk.em_selection_function(d) for d in distances], lw = 0.15, color=s_m.to_rgba(x['w0'][i]), alpha = 0.5)
                    else: ax2.plot(z, [lk.em_selection_function(d) for d in distances], lw = 0.15, color=s_m.to_rgba(x['h'][i]), alpha = 0.5)
                    O.DestroyCosmologicalParameters()
                CB = plt.colorbar(s_m, orientation='vertical', pad=0.15)
                if model == "DE": CB.set_label('w_0')
                else: CB.set_label('h')
                ax2.set_ylim(0.0,1.0)
                ax2.set_ylabel('selection function')
            ax.axvline(e.z_true, linestyle='dotted', lw=0.5, color='k')
            ax.hist(x['z%d'%e.ID], bins=z, density=True, alpha = 0.5, facecolor="green")
            ax.hist(x['z%d'%e.ID], bins=z, density=True, alpha = 0.5, histtype='step', edgecolor="k")

            for g in e.potential_galaxy_hosts:
                zg = np.linspace(g.redshift - 5*g.dredshift, g.redshift+5*g.dredshift, 100)
                pg = norm.pdf(zg, g.redshift, g.dredshift*(1+g.redshift))*g.weight
                ax.plot(zg, pg, lw=0.5,color='k')
            ax.set_xlabel('$z_{%d}$'%e.ID)
            ax.set_ylabel('probability density')
            plt.savefig(os.path.join(output,'redshift_%d'%e.ID+'.png'), bbox_inches='tight')
            plt.close()
    
    if opts.event_class == "MBH":
        dl = [e.dl/1e3 for e in C.data]
        ztrue = [e.potential_galaxy_hosts[0].redshift for e in C.data]
        dztrue = np.squeeze([[ztrue[i]-e.zmin,e.zmax-ztrue[i]] for i,e in enumerate(C.data)]).T
        deltadl = [np.sqrt((e.sigma/1e3)**2+(lk.sigma_weak_lensing(e.potential_galaxy_hosts[0].redshift,e.dl)/1e3)**2) for e in C.data]
        z = [np.median(x['z%d'%e.ID]) for e in C.data]
        deltaz = [2*np.std(x['z%d'%e.ID]) for e in C.data]
        
        # injected cosmology
        omega_true = cs.CosmologicalParameters(0.73,0.25,0.75,-1,0)
        redshift = np.logspace(-3,1.0,100)
        
        # loop over the posterior samples to get all models to then average
        # for the plot
        
        models = []
        
        for k in range(x.shape[0]):
            if opts.model == "LambdaCDM":
                omega = cs.CosmologicalParameters(x['h'][k],
                                               x['om'][k],
                                               1.0-x['om'][k],
                                               -1.0,
                                               0.0)
            elif opts.model == "CLambdaCDM":
                omega = cs.CosmologicalParameters(x['h'][k],
                                               x['om'][k],
                                               x['ol'][k],
                                               -1.0,
                                               0.0)
            elif opts.model == "LambdaCDMDE":
                omega = cs.CosmologicalParameters(x['h'][k],
                                               x['om'][k],
                                               x['ol'][k],
                                               x['w0'][k],
                                               x['w1'][k])
            elif opts.model == "DE":
                omega = cs.CosmologicalParameters(0.73,
                                               0.25,
                                               0.75,
                                               x['w0'][k],
                                               x['w1'][k])
            else:
                print(opts.model,"is unknown")
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
    
    
    if model == "LambdaCDM":
        samps = np.column_stack((x['h'],x['om']))
        fig = corner.corner(samps,
               labels= [r'$h$',
                        r'$\Omega_m$'],
               quantiles=[0.05, 0.5, 0.95],
               show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
               use_math_text=True, truths=[0.73,0.25],
               filename=os.path.join(output,'joint_posterior.pdf'))
#        axes = fig.get_axes()
#        axes[0].set_xlim(0.69, 0.77)
#        axes[2].set_xlim(0.69, 0.77)
#        axes[3].set_xlim(0.04, 0.5)
#        axes[2].set_ylim(0.04, 0.5)
    
    if model == "CLambdaCDM":
        samps = np.column_stack((x['h'],x['om'],x['ol'],1.0-x['om']-x['ol']))
        fig = corner.corner(samps,
               labels= [r'$h$',
                        r'$\Omega_m$',
                        r'$\Omega_\Lambda$',
                        r'$\Omega_k$'],
               quantiles=[0.05, 0.5, 0.95],
               show_titles=True, title_kwargs={"fontsize": 12},
               use_math_text=True, truths=[0.73,0.25,0.75,0.0],
               filename=os.path.join(output,'joint_posterior.pdf'))
               
    if model == "LambdaCDMDE":
        samps = np.column_stack((x['h'],x['om'],x['ol'],x['w0'],x['w1']))
        fig = corner.corner(samps,
                        labels= [r'$h$',
                                 r'$\Omega_m$',
                                 r'$\Omega_\Lambda$',
                                 r'$w_0$',
                                 r'$w_a$'],
                        quantiles=[0.05, 0.5, 0.95],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        use_math_text=True, truths=[0.73,0.25,0.75,-1.0,0.0],
                        filename=os.path.join(output,'joint_posterior.pdf'))

    if model == "DE":
        samps = np.column_stack((x['w0'],x['w1']))
        fig = corner.corner(samps,
                        labels= [r'$w_0$',
                                 r'$w_a$'],
                        quantiles=[0.05, 0.5, 0.95],
                        show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                        use_math_text=True, truths=[-1.0,0.0],
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

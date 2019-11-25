#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import unittest
import numpy as np
import cpnest.model
import sys
import os
from optparse import OptionParser
import itertools as it
import cosmology as cs
import readdata
from scipy.special import logsumexp
import likelihood as lk
from functools import reduce

"""
G = the GW is in a galaxy that i see
N = the GW is in a galaxy that i do not see
D = a GW
I = i see only GW with SNR > 20

p(H|D(G+N)I) \propto p(H|I)p(D(G+N)|HI)
p(D(G+N)|HI) = p(DG+DN|HI) = p(DG|HI)+p(DN|HI) = p(D|GHI)p(G|HI)+p(D|NHI)p(N|HI) = p(D|HI)(p(G|HI)+p(N|HI))
"""
class CosmologicalModel(cpnest.model.Model):

    names=[]#'h','om','ol','w0','w1']
    bounds=[]#[0.5,1.0],[0.04,1.0],[0.0,1.0],[-2.0,0.0],[-3.0,3.0]]
    
    def __init__(self, model, data, *args, **kwargs):

        super(CosmologicalModel,self).__init__()
        # Set up the data
        self.data           = data
        self.N              = len(self.data)
        self.model          = model
        self.gw_selection   = kwargs['gw_selection']
        self.em_selection   = kwargs['em_selection']
        self.z_threshold    = kwargs['z_threshold']
        self.snr_threshold  = kwargs['snr_threshold']
        self.O              = None
        
        if self.model == "LambdaCDM":
            
            self.names  = ['h','om']
            self.bounds = [[0.5,1.0],[0.04,0.5]]
        
        elif self.model == "LambdaCDMDE":
            
            self.names  = ['h','om','ol','w0','w1']
            self.bounds = [[0.5,1.0],[0.04,0.5],[0.0,1.0],[-2.0,0.0],[-3.0,3.0]]
        
        elif self.model == "DE":
            
            self.names  = ['w0','w1']
            self.bounds = [[-2.0,0.0],[-3.0,3.0]]
        
        else:
            
            print("Cosmological model %s not supported. exiting..\n"%self.model)
            exit()
        
        for e in self.data:
            self.bounds.append([e.zmin,e.zmax])
            self.names.append('z%d'%e.ID)
            
        self._initialise_galaxy_hosts()
        
        print("==========================================")
        print("cpnest model initialised with:")
        print("Cosmological model: {0}".format(self.model))
        print("Number of events: {0}".format(len(self.data)))
        print("EM correction: {0}".format(self.em_selection))
        print("GW correction {0}".format(self.gw_selection))
        print("==========================================")

    def _initialise_galaxy_hosts(self):
        self.hosts = {e.ID:np.array([(g.redshift,g.dredshift,g.weight) for g in e.potential_galaxy_hosts]) for e in self.data}

    def find_max_distance(self):
        dlmax = -1.0
        for _ in range(1000):
            if self.model == "LambdaCDM":
                z_idx = 2
                h  = np.random.uniform(self.bounds[0][0],self.bounds[0][1])
                om = np.random.uniform(self.bounds[1][0],self.bounds[1][1])
                ol = 1.0-om
                O = cs.CosmologicalParameters(h,om,ol,-1.0,0.0)
            elif self.model == "LambdaCDMDE":
                z_idx = 5
                h  = np.random.uniform(self.bounds[0][0],self.bounds[0][1])
                om = np.random.uniform(self.bounds[1][0],self.bounds[1][1])
                ol = np.random.uniform(self.bounds[2][0],self.bounds[2][1])
                w0 = np.random.uniform(self.bounds[3][0],self.bounds[3][1])
                w1 = np.random.uniform(self.bounds[4][0],self.bounds[4][1])
                O = cs.CosmologicalParameters(h,om,ol,w0,w1)
            elif self.model == "DE":
                z_idx = 5
                w0 = np.random.uniform(self.bounds[0][0],self.bounds[0][1])
                w1 = np.random.uniform(self.bounds[1][0],self.bounds[1][1])
                O = cs.CosmologicalParameters(0.73,0.25,0.75,w0,w1)
            dl = O.LuminosityDistance(self.bounds[z_idx][1])
            if dl > dlmax: dlmax = dl
            O.DestroyCosmologicalParameters()
            
        return dlmax
        
    def log_prior(self,x):
        logP = super(CosmologicalModel,self).log_prior(x)
        
        if np.isfinite(logP):
            """
            apply a uniform in comoving volume density redshift prior
            """
            if self.model == "LambdaCDM":
                self.O = cs.CosmologicalParameters(x['h'],x['om'],1.0-x['om'],-1.0,0.0)
            elif self.model == "LambdaCDMDE":
                self.O = cs.CosmologicalParameters(x['h'],x['om'],x['ol'],x['w0'],x['w1'])
            elif self.model == "DE":
                self.O = cs.CosmologicalParameters(0.7,0.25,0.75,x['w0'],x['w1'])

            logP += np.sum([np.log(self.O.ComovingVolumeElement(x['z%d'%e.ID])) for e in self.data])
        return logP
    
    def log_likelihood(self,x):
        
        # compute the p(GW|G\Omega)p(G|\Omega)+p(GW|~G\Omega)p(~G|\Omega)
        logL = np.sum([lk.logLikelihood_single_event(self.hosts[e.ID], e.dl, e.sigma, e.VC, self.O, x['z%d'%e.ID],
                                em_selection = self.em_selection, zmin = self.bounds[-1][0], zmax = self.bounds[-1][1]) for e in self.data])

        self.O.DestroyCosmologicalParameters()

        return logL

usage=""" %prog (options)"""

if __name__=='__main__':

    parser=OptionParser(usage)
    parser.add_option('-o','--out-dir', default=None,type='string',metavar='DIR',help='Directory for output')
    parser.add_option('-t','--threads', default=None,type='int',metavar='threads',help='Number of threads (default = 1/core)')
    parser.add_option('-d','--data',    default=None,type='string',metavar='data',help='galaxy data location')
    parser.add_option('-e','--event',   default=None,type='int',metavar='event',help='event number')
    parser.add_option('-c','--event-class',default=None,type='string',metavar='event_class',help='class of the event(s) [MBH, EMRI, sBH]')
    parser.add_option('-m','--model',   default='LambdaCDM',type='string',metavar='model',help='cosmological model to assume for the analysis (default LambdaCDM). Supports LambdaCDM and LambdaCDMDE')
    parser.add_option('-j','--joint',   default=0, type='int',metavar='joint',help='run a joint analysis for N events, randomly selected. (EMRI only)')
    parser.add_option('-s','--seed',   default=0, type='int', metavar='seed',help='rando seed initialisation')
    parser.add_option('--snr_threshold',    default=0, type='float',metavar='snr_threshold',help='SNR detection threshold')
    parser.add_option('--zhorizon',     default=1.0, type='float',metavar='zhorizon',help='Horizon redshift corresponding to the SNR threshold')
    parser.add_option('--gw_selection', default=0, type='int',metavar='gw_selection',help='use GW selection function')
    parser.add_option('--em_selection', default=0, type='int',metavar='em_selection',help='use EM selection function')
    parser.add_option('--nlive',        default=1000, type='int',metavar='nlive',help='number of live points')
    parser.add_option('--poolsize',     default=100, type='int',metavar='poolsize',help='poolsize for the samplers')
    parser.add_option('--maxmcmc',      default=1000, type='int',metavar='maxmcmc',help='maximum number of mcmc steps')
    (opts,args)=parser.parse_args()
    
    gw_selection = opts.gw_selection
    em_selection = opts.em_selection
    
    if opts.event_class == "SMBH":
        # if running on SMBH override the selection functions
        gw_selection = 0
        em_selection = 0
#    if opts.event_class == "EMRI" or opts.event_class == "sBH":
#        gw_selection = True
#    if opts.event_class == "EMRI" or opts.event_class == "sBH":
#        em_selection = True

    if opts.event_class == "EMRI" and opts.joint !=0:
        np.random.seed(opts.seed)
        events = readdata.read_event(opts.event_class, opts.data, None)
        N = opts.joint#np.int(np.random.poisson(len(events)*4./10.))
        print("Will run a random catalog selection of {0} events:".format(N))
        print("==================================================")
        selected_events  = []
        count = 0
        if 0:
            while len(selected_events) < N-count and not(len(events) == 0):

                while True:
                    if len(events) > 0:
                        idx = np.random.randint(len(events))
                        selected_event = events.pop(idx)
                    else:
                        break
                    if len(selected_event.potential_galaxy_hosts) < 100:
                        selected_events.append(selected_event)
                        count += 1
                        break
            
            events = np.copy(selected_events)
        else: events = np.random.choice(events, size = N, replace = False)
        for e in events:
            print("event {0}: distance {1} \pm {2} Mpc, z \in [{3},{4}] galaxies {5}".format(e.ID,e.dl,e.sigma,e.zmin,e.zmax,len(e.potential_galaxy_hosts)))
        print("==================================================")
    else:
        events = readdata.read_event(opts.event_class, opts.data, opts.event)

    model = opts.model

    if opts.out_dir is None:
        output = opts.data+"/EVENT_1%03d/"%(opts.event+1)
    else:
        output = opts.out_dir
    
    C = CosmologicalModel(model,
                          events,
                          gw_selection = gw_selection,
                          em_selection = em_selection,
                          snr_threshold= opts.snr_threshold,
                          z_threshold  = opts.zhorizon)
    
    if 1:
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
    else:
        x = np.genfromtxt(os.path.join(output,"chain_"+str(opts.nlive)+"_1234.txt"), names=True)
        from cpnest import nest2pos
        x = nest2pos.draw_posterior_many([x], [opts.nlive], verbose=False)

    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    for e in C.data:
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        z = np.linspace(e.zmin,e.zmax, 100)
        
        ax2 = ax.twinx()
        
        normalisation = matplotlib.colors.Normalize(vmin=np.min(x['h']), vmax=np.max(x['h']))

        # choose a colormap
        c_m = matplotlib.cm.cool

        # create a ScalarMappable and initialize a data structure
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=normalisation)
        s_m.set_array([])
        ax.axvline(e.z_true, linestyle='dotted', lw=0.5, color='k')
        for i in range(x.shape[0])[::10]:
            O = cs.CosmologicalParameters(x['h'][i],x['om'][i],1.0-x['om'][i],-1.0,0.0)
            distances = np.array([O.LuminosityDistance(zi) for zi in z])
            ax2.plot(z, [lk.em_selection_function(d) for d in distances], lw = 0.15, color=s_m.to_rgba(x['h'][i]), alpha = 0.5)
#            ax2.plot(z, np.exp(-0.5*(distances-e.dl)*(distances-e.dl)/e.sigma**2), lw = 0.01, color='k', alpha = 0.5)

            O.DestroyCosmologicalParameters()
        CB = plt.colorbar(s_m, orientation='vertical', pad=0.15)
        CB.set_label('h')
        ax2.set_ylim(0.0,1.0)
        ax2.set_ylabel('selection function')
        ax.hist(x['z%d'%e.ID], bins=z, density=True, alpha = 0.5, facecolor="green")
        ax.hist(x['z%d'%e.ID], bins=z, density=True, alpha = 0.5, histtype='step', edgecolor="k")

        for g in e.potential_galaxy_hosts:
            zg = np.linspace(g.redshift - 5*g.dredshift, g.redshift+5*g.dredshift, 100)
            pg = norm.pdf(zg, g.redshift, g.dredshift*(1+g.redshift))*g.weight
            ax.plot(zg, pg, lw=0.5,color='k')
        ax.set_xlabel('$z_{%d}$'%e.ID)
        ax.set_ylabel('probability density')
        plt.savefig(os.path.join(output,'redshift_%d'%e.ID+'.pdf'), bbox_inches='tight')
        plt.close()
    
    import corner
    if model == "LambdaCDM":
        samps = np.column_stack((x['h'],x['om']))
        fig = corner.corner(samps,
               labels= [r'$h$',
                        r'$\Omega_m$'],
               quantiles=[0.05, 0.5, 0.95],
               show_titles=True, title_kwargs={"fontsize": 12},
               use_math_text=True, truths=[0.73,0.25],
               filename=os.path.join(output,'joint_posterior.pdf'))
    if model == "LambdaCDMDE":
        samps = np.column_stack((x['h'],x['om'],x['ol'],x['w0'],x['w1']))
        fig = corner.corner(samps,
                        labels= [r'$h$',
                                 r'$\Omega_m$',
                                 r'$\Omega_\Lambda$',
                                 r'$w_0$',
                                 r'$w_1$'],
                        quantiles=[0.05, 0.5, 0.95],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        use_math_text=True, truths=[0.73,0.25,0.75,-1.0,0.0],
                        filename=os.path.join(output,'joint_posterior.pdf'))
    if model == "DE":
        samps = np.column_stack((x['w0'],x['w1']))
        fig = corner.corner(samps,
                        labels= [r'$w_0$',
                                 r'$w_1$'],
                        quantiles=[0.05, 0.5, 0.95],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        use_math_text=True, truths=[-1.0,0.0],
                        filename=os.path.join(output,'joint_posterior.pdf'))
    fig.savefig(os.path.join(output,'joint_posterior.pdf'), bbox_inches='tight')

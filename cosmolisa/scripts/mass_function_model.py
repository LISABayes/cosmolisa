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
import cosmolisa.readdata
import matplotlib
import corner
import subprocess
import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt
import cosmolisa.cosmology as cs
import cosmolisa.likelihood as lk
import cosmolisa.prior as pr
import numpy as np
import cosmolisa.galaxy as gal
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

truths = {'h':0.73,'om':0.25,'ol':0.75,'w0':-1.0,'w1':0.0,'r0':5e-13,'Q':3.1,'W':45.,'R':4.1} # 0.73, 0.25, 0.75, -1.0, 0.0

class MassFunctionModel(cpnest.model.Model):

    names  = [] #'h','om','ol','w0','w1']
    bounds = [] #[0.5,1.0],[0.04,0.5],[0.0,1.0],[-3.0,-0.3],[-1.0,1.0]]
    
    def __init__(self, model, data, *args, **kwargs):

        super(MassFunctionModel,self).__init__()
        # Set up the data
        self.data             = data
        self.model            = model
        self.N                = len(self.data)
        self.mass_model       = model
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
        
        self.names.append('phistar')
        self.bounds.append([1e-10,1.0])
        self.names.append('phistar_exponent')
        self.bounds.append([-5.0,5.0])
        self.names.append('logMstar')
        self.bounds.append([9.0,12.0])
        self.names.append('logMstar_exponent')
        self.bounds.append([-2.0,2.0])
        self.names.append('alpha')
        self.bounds.append([-2.0,2.0])
        self.names.append('alpha_exponent')
        self.bounds.append([-2.0,2.0])

    def log_prior(self,x):
    
        logP = super(MassFunctionModel,self).log_prior(x)
        
        if np.isfinite(logP):
        
            if   (self.model == "LambdaCDM"):
                self.O = cs.CosmologicalParameters(x['h'],x['om'],1.0-x['om'],truths['w0'],truths['w1'])
            elif (self.model == "CLambdaCDM"):
                self.O = cs.CosmologicalParameters(x['h'],x['om'],x['ol'],truths['w0'],truths['w1'])
            elif (self.model == "LambdaCDMDE"):
                self.O = cs.CosmologicalParameters(x['h'],x['om'],x['ol'],x['w0'],x['w1'])
            elif (self.model == "DE"):
                self.O = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],x['w0'],x['w1'])
#            self.O = cs.CosmologicalParameters(truths['h'],truths['om'],truths['ol'],truths['w0'],truths['w1'])
            self.mass_model = gal.GalaxyMassDistribution(self.O,
                                                         x['phistar'],
                                                         x['phistar_exponent'],
                                                         x['logMstar'],
                                                         x['logMstar_exponent'],
                                                         x['alpha'],
                                                         x['alpha_exponent'],
                                                         mmin,
                                                         mmax,
                                                         self.data[:,1].min(),
                                                         self.data[:,1].max(),
                                                         ramin,
                                                         ramax,
                                                         decmin,
                                                         decmax,
                                                         threshold,
                                                         slope_model_choice,
                                                         cutoff_model_choice,
                                                         density_model_choice)
        return logP

    def log_likelihood(self,x):
        
        logL = self.mass_model.loglikelihood(self.data)

        self.O.DestroyCosmologicalParameters()
        if np.isfinite(logL):
            return logL
        else:
            return -np.inf

if __name__ == "__main__":
    
    Mstar_exponent = 0
    alpha_exponent = 0
    phistar_exponent = 0
    mmin = 9.0
    mmax = 14.0
    zmin = 0.0
    zmax = 2.0
    ramin = 0.0
    ramax = 2.0*np.pi
    decmin = -np.pi/2.0
    decmax = np.pi/2.0
    threshold = 10.7
    slope_model_choice = 1
    cutoff_model_choice = 1
    density_model_choice = 1
    selection = 0 # 0:full, 1:detected, 2:non detected

    plot_labels = {0:'full',1:'detected',2:'non_detected'}
    
    for i in range(1,260):
        print("reading %d"%i)
        path = '/Users/wdp/repositories/cosmolisa/data/EMRI_SAMPLE_MODEL106_TTOT10yr_SIG2_GAUSS/EVENT_1%03d'%i
        if i == 1:
            d = np.loadtxt(os.path.join(path,'ERRORBOX.dat'))
        else:
#            if np.random.uniform(0,1) < 0.1:
            d = np.row_stack((d,np.loadtxt(os.path.join(path,'ERRORBOX.dat'))))
    
    data = np.column_stack((d[:,3],d[:,1]))
    idx = np.random.choice(np.arange(data.shape[0]), size = 10000, replace=False)
    data = data[idx,:]
    output_folder = "./full_run_random/"
    nlive = 2000
    print("total number of galaxies = ", data.shape[0])

    if 1:
        C = MassFunctionModel("LambdaCDM", data)
        work=cpnest.CPNest(C,
                           verbose      = 2,
                           poolsize     = 32,
                           nthreads     = 6,
                           nlive        = nlive,
                           maxmcmc      = 1000,
                           output       = output_folder,
                           nhamiltonian = 0)
        work.run()
        print('log Evidence {0}'.format(work.NS.logZ))
        x = work.posterior_samples.ravel()
    else:
        print("Reading the chain...")
        x = np.genfromtxt(os.path.join(output_folder,"chain_"+str(nlive)+"_1234.txt"), names=True)
        from cpnest import nest2pos
        print("Drawing posterior samples...")
        x = nest2pos.draw_posterior_many([x], [nlive], verbose=False)
    
#    x = x[:100]
    lM  = np.linspace(9.0,13.0,100)
    Z   = np.linspace(data[:,1].min(),data[:,1].max(),100)
    PMZ = np.zeros((x.shape[0],lM.shape[0],Z.shape[0]))
    PM  = np.zeros((x.shape[0],lM.shape[0]))
    PZ  = np.zeros((x.shape[0],Z.shape[0]))
    
    alpha_law = np.zeros((x.shape[0], Z.shape[0]))
    logMstar_law = np.zeros((x.shape[0], Z.shape[0]))
    phistar_law = np.zeros((x.shape[0], Z.shape[0]))
    
    OM, OZ = np.meshgrid(lM,Z)
    for i,xi in enumerate(x):
        sys.stderr.write("processing sample {0} of {1}\r".format(i+1,x.shape[0]))
        mass_model = gal.GalaxyMassDistribution(cs.CosmologicalParameters(xi['h'],xi['om'],1.0-xi['om'],truths['w0'],truths['w1']),
                                         xi['phistar'],
                                         xi['phistar_exponent'],
                                         xi['logMstar'],
                                         xi['logMstar_exponent'],
                                         xi['alpha'],
                                         xi['alpha_exponent'],
                                         mmin,
                                         mmax,
                                         data[:,1].min(),
                                         data[:,1].max(),
                                         ramin,
                                         ramax,
                                         decmin,
                                         decmax,
                                         threshold,
                                         slope_model_choice,
                                         cutoff_model_choice,
                                         density_model_choice)
        PMZ[i,:,:] = np.array([mass_model.pdf(lMi, Zj, selection) for Zj in Z for lMi in lM]).reshape(lM.shape[0],Z.shape[0])
        
        if slope_model_choice == 1:
            alpha_law[i,:] = xi['alpha']*(1+Z)**xi['alpha_exponent']
        if cutoff_model_choice == 1:
            logMstar_law[i,:] = xi['logMstar']*(1+Z)**xi['logMstar_exponent']
        if density_model_choice == 1:
            phistar_law[i,:] = xi['phistar']*(1+Z)**xi['phistar_exponent']
        
    def marginalise(p, x, axis):
        return np.sum(p*np.diff(x)[0], axis = axis)
        
    sys.stderr.write("\n")
    PMZlow, PMZmean, PMZhigh = np.percentile(PMZ, [5,50,95], axis = 0)
    PMmean = marginalise(PMZmean, Z, axis=0)
    PZmean = marginalise(PMZmean, lM, axis=1)
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = '3d')
    S   = ax.plot_surface(OM, OZ, PMZmean, cmap = cm.rainbow,
                      norm=colors.SymLogNorm(linthresh=1e-15, linscale=1e-15, vmin=0.0, vmax=PMZ.max(), base = 10.0))
    ax.set_xlabel("$\log_{10} (M/M_\odot)$")
    ax.set_ylabel("$z$")
    ax.set_zlabel("$d^2N/dzd\log_{10}(M/M_\odot)$")
    cb = plt.colorbar(S, orientation="horizontal", pad=0.2)
    cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation='vertical')
    fig.savefig(os.path.join(output_folder,'mass_redshift_distribution_{0}.pdf'.format(plot_labels[selection])),bbox_inches='tight')
    
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    S   = ax.plot(lM, PMmean, color='k', lw=1.2)
    ax.fill_between(lM, marginalise(PMZlow, Z, axis=0), marginalise(PMZhigh, Z, axis=0), facecolor = "0.5",alpha=0.3)
    ax.hist(data[:,0], bins = 100, density = True, facecolor="0.55", alpha=0.5)
    ax.set_xlabel("$\log_{10} (M/M_\odot)$")
    ax.set_ylabel("$z$")
    ax.set_ylabel("$dN/d\log_{10}(M/M_\odot)$")
    fig.savefig(os.path.join(output_folder,'mass_function_{0}.pdf'.format(plot_labels[selection])),bbox_inches='tight')
    
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    S   = ax.plot(Z, PZmean, color='k', lw=1.2)
    ax.fill_between(Z, marginalise(PMZlow, lM, axis=1), marginalise(PMZhigh, lM, axis=1), facecolor = "0.5",alpha=0.3)
    ax.hist(data[:,1], bins = 100, density = True, facecolor="0.75", alpha=0.5)
    ax.set_xlabel("$z$")
    ax.set_ylabel("$dN/dz$")
    fig.savefig(os.path.join(output_folder,'redshift_distribution_{0}.pdf'.format(plot_labels[selection])),bbox_inches='tight')

    if slope_model_choice == 1:
        l,m,h = np.percentile(alpha_law, [5,50,95], axis = 0)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        S   = ax.plot(Z, m, color='k', lw=1.2)
        ax.fill_between(Z, l, h, facecolor = "0.5")
        ax.set_xlabel("$z$")
        ax.set_ylabel("$alpha$")
        fig.savefig(os.path.join(output_folder,'low_mass_slope_{0}.pdf'.format(plot_labels[selection])),bbox_inches='tight')
    if cutoff_model_choice == 1:
        l,m,h = np.percentile(logMstar_law, [5,50,95], axis = 0)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        S   = ax.plot(Z, m, color='k', lw=1.2)
        ax.fill_between(Z, l, h, facecolor = "0.5")
        ax.set_xlabel("$z$")
        ax.set_ylabel("$\log(M/M_\odot)$")
        fig.savefig(os.path.join(output_folder,'cutoff_mass_{0}.pdf'.format(plot_labels[selection])),bbox_inches='tight')
    if density_model_choice == 1:
        l,m,h = np.percentile(phistar_law, [5,50,95], axis = 0)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        S   = ax.plot(Z, m, color='k', lw=1.2)
        ax.fill_between(Z, l, h, facecolor = "0.5")
        ax.set_xlabel("$z$")
        ax.set_ylabel("$\phi^*$")
        fig.savefig(os.path.join(output_folder,'density_{0}.pdf'.format(plot_labels[selection])),bbox_inches='tight')

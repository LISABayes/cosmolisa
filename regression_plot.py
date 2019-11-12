import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from optparse import OptionParser
import sys
import readdata


import multiprocessing as mp
from scipy.misc import logsumexp
from cosmology import *
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib
import cPickle as pickle
from optparse import OptionParser

def init_plotting():
    plt.rcParams['figure.figsize'] = (2*3.4, 2*3.4)
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

if __name__=="__main__":
    parser=OptionParser()
    parser.add_option('-o','--out-dir',default=None,type='string',metavar='DIR',help='Directory for output')
    parser.add_option('-d','--data',default=None,type='string',metavar='data',help='galaxy data location')
    parser.add_option('-p','--posteriors',default=None,type='string',metavar='posterior_file',help='posterior file from cpnest')
    parser.add_option('-m','--model',default='LAMBDACDM',type='string',metavar='model',help='model (default: LAMBDACDM)')
    parser.add_option('-s','--source',default=None,type='string',metavar='source',help='source class')
    (opts,args)=parser.parse_args()
    init_plotting()
    # read in the events
    events = readdata.read_event(opts.source, opts.data)
    # read in the posterior samples
    try:
        fuck 
        posteriors = np.genfromtxt(opts.posteriors,names=True)
    except:
        from cpnest import nest2pos
        x = np.genfromtxt('test_logP_dv/chain_1000_1234.txt',names=True)
        posteriors = nest2pos.draw_posterior_many([x], [1000], verbose=False)
        names = ''
        for n in posteriors.dtype.names: names += n+ '\t'
        np.savetxt('test_logP_dv/posterior.dat', posteriors, header = names)

    dl = [e.dl/1e3 for e in events]
    ztrue = [e.potential_galaxy_hosts[0].redshift for e in events]
    deltadl = [2*e.sigma/1e3 for e in events]
    z = [np.median(posteriors['z%d'%e.ID]) for e in events]
    deltaz = [2*np.std(posteriors['z%d'%e.ID]) for e in events]
    
    # injected cosmology
    omega_true = CosmologicalParameters(0.73,0.25,0.75,-1,0)
    redshift = np.logspace(-3,1.0,100)
    
    # loop over the posterior samples to get all models to then average
    # for the plot
    
    models = []
    
    for k in range(posteriors.shape[0]):
        if opts.model == "LAMBDACDM":
            omega = CosmologicalParameters(posteriors['h'][k],
                                           posteriors['om'][k],
                                           1.0-posteriors['om'][k],
                                           -1,
                                           0.0)

        elif opts.model == "LAMBDACDMDE":
            omega = CosmologicalParameters(posteriors['h'][k],
                                           posteriors['om'][k],
                                           posteriors['ol'][k],
                                           posteriors['w0'][k],
                                           posteriors['w1'][k])
        else:
            print(opts.model,"is unknown")
            exit()
        models.append([omega.LuminosityDistance(zi)/1e3 for zi in redshift])
        omega.DestroyCosmologicalParameters()
    
    models = np.array(models)
    model2p5,model16,model50,model84,model97p5 = np.percentile(models,[2.7,16.0,50.0,84.0,97.5],axis = 0)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(z,dl,xerr=deltaz,yerr=deltadl,markersize=2,linewidth=2,color='k',fmt='o')
    ax.plot(redshift,[omega_true.LuminosityDistance(zi)/1e3 for zi in redshift],linestyle='dashed',color='r')
    ax.plot(redshift,model50,color='k')
    ax.errorbar(ztrue, dl, yerr=deltadl, markersize=8,linewidth=2,color='r',fmt='o')
    ax.fill_between(redshift,model2p5,model97p5,facecolor='turquoise')
    ax.fill_between(redshift,model16,model84,facecolor='cyan')
    ax.set_xlabel(r"z")
    ax.set_ylabel(r"$D_L$/Gpc")
    ax.set_xlim(np.min(redshift),0.8)
    ax.set_ylim(0.0,4.0)
    fig.savefig("test_logP_dv/regression.pdf",bbox_inches='tight')
#    plt.close(fig)

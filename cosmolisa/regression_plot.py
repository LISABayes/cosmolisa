import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from optparse import OptionParser
import sys, os
import readdata

import multiprocessing as mp
from scipy.special import logsumexp
from cosmology import *
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib
import _pickle as pickle
from optparse import OptionParser

def init_plotting():
    plt.rcParams['figure.figsize'] = (4*3.4, 2*3.4)
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

def twod_kde(x,y):
    X, Y = np.mgrid[x.min()*0.9:x.max()*1.1:100j, y.min()*0.9:y.max()*1.1:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    return X, Y, np.reshape(kernel(positions).T, X.shape)

def FindHeightForLevel(inArr, adLevels):
    # flatten the array
    oldshape = np.shape(inArr)
    adInput= np.reshape(inArr,oldshape[0]*oldshape[1])
    # GET ARRAY SPECIFICS
    nLength = np.size(adInput)

    # CREATE REVERSED SORTED LIST
    adTemp = -1.0 * adInput
    adSorted = np.sort(adTemp)
    adSorted = -1.0 * adSorted

    # CREATE NORMALISED CUMULATIVE DISTRIBUTION
    adCum = np.zeros(nLength)
    adCum[0] = adSorted[0]
    for i in range(1,nLength):
        adCum[i] = np.logaddexp(adCum[i-1], adSorted[i])
    adCum = adCum - adCum[-1]

    # FIND VALUE CLOSEST TO LEVELS
    adHeights = []
    for item in adLevels:
        idx=(np.abs(adCum-np.log(item))).argmin()
        adHeights.append(adSorted[idx])

    adHeights = np.array(adHeights)
    return np.sort(adHeights)

if __name__=="__main__":
    parser=OptionParser()
    parser.add_option('-o','--outdir',default=None,type='string',metavar='DIR',help='Directory for output')
    parser.add_option('-d','--data',default=None,type='string',metavar='data',help='galaxy data location')
    parser.add_option('-p','--posteriors',default=None,type='string',metavar='DIR',help='posterior location from cpnest')
    parser.add_option('-m','--model',default='LAMBDACDM',type='string',metavar='model',help='model (default: LAMBDACDM)')
    parser.add_option('-s','--source',default=None,type='string',metavar='source',help='source class')
    (opts,args)=parser.parse_args()
    init_plotting()
    # read in the events
    events = readdata.read_event(opts.source, opts.data, None)
    # read in the posterior samples
    try:
        posteriors = np.genfromtxt(os.path.join(opts.posteriors,'posterior.dat'),names=True)
    except:
        sys.stderr.write("{0} not found. Generating posteriors from the chain ...\n".format(os.path.join(opts.posteriors,'posterior.dat')))
        from cpnest import nest2pos
        x = np.genfromtxt(os.path.join(opts.posteriors,'chain_5000_1234.txt'),names=True)
        posteriors = nest2pos.draw_posterior_many([x], [5000], verbose=False)
        names = ''
        for n in posteriors.dtype.names: names += n+ '\t'
        np.savetxt(os.path.join(opts.posteriors,'posterior.dat'), posteriors, header = names)
    
    # injected cosmology
    omega_true = CosmologicalParameters(0.73,0.25,0.75,-1,0)
    
    dlmeasured = []
    zmeasured = []
    dzmeasured = []
    ddlmeasured = []
    redshift_galaxies = []
    distance_galaxies = []
    weight_galaxies = []
    redshift_posteriors = []
    distance_posteriors = []
    # FIXME:
    for e in events:
        try:
            redshift_posteriors.append(posteriors['z%d'%e.ID])
            distance_posteriors.append(np.random.normal(e.dl/1e3,e.sigma/1e3, size=len(posteriors['z%d'%e.ID])))
            redshift_galaxies.append([g.redshift for g in e.potential_galaxy_hosts])
            weight_galaxies.append([g.weight for g in e.potential_galaxy_hosts])
            distance_galaxies.append([e.dl/1e3 for g in e.potential_galaxy_hosts])
            dlmeasured.append(e.dl/1e3)
            ddlmeasured.append(2*e.sigma/1e3)
            zmeasured.append(np.mean([g.redshift for g in e.potential_galaxy_hosts]))
            dzmeasured.append(2*np.std([g.redshift for g in e.potential_galaxy_hosts]))
        except:
            pass

    redshift   = np.logspace(-3,0.4,100)
    # loop over the posterior samples to get all models to then average
    # for the plot
    
    models = []
    
    for k in range(posteriors.shape[0]):
        if opts.model == "LAMBDACDM":
            omega = CosmologicalParameters(posteriors['h'][k],
                                           posteriors['om'][k],
                                           1.0-posteriors['om'][k],
                                           omega_true.w0,
                                           omega_true.w1)

        elif opts.model == "LAMBDACDMDE":
            omega = CosmologicalParameters(posteriors['h'][k],
                                           posteriors['om'][k],
                                           posteriors['ol'][k],
                                           posteriors['w0'][k],
                                           posteriors['w1'][k])
        elif opts.model == "DE":
            omega = CosmologicalParameters(omega_true.h,
                                           omega_true.om,
                                           omega_true.ol,
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
    
    for z,d,w in zip(redshift_galaxies, distance_galaxies, weight_galaxies):
##        print(z,d,w)
        ax.scatter(z,d,c=w,s=2,alpha=0.35,zorder=0, edgecolors='face')
#    plt.show()
#    exit()
    
    for i,z,d in zip(range(len(redshift_posteriors)),redshift_posteriors,distance_posteriors):
        sys.stderr.write("processing event {0} of {1}\r".format(i+1,len(redshift_posteriors)))
        X, Y, Z = twod_kde(np.array(z)[::100],np.array(d)[::100])
        levels = FindHeightForLevel(np.log(Z), [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        C = ax.contourf(X, Y, np.log(Z), levels, alpha = 0.4, cmap=matplotlib.cm.hot)
        C2 = ax.contour(C, levels, alpha = 0.25, colors = 'k', linewidths=(0.2,))
    sys.stderr.write("\n")
    cbar = fig.colorbar(C)
    cbar.ax.set_ylabel(r'credible level', fontsize=15)
    # Add the contour line levels to the colorbar
    cbar.add_lines(C2)
    cbar.ax.set_yticklabels([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1], fontsize=13)

#    # raw redshift and distance of GW for galaxies mean
#    ax.errorbar(zmeasured,dlmeasured,xerr=dzmeasured,yerr=ddlmeasured,
#                markersize=2,linewidth=0.5,color='r',fmt='o',zorder=0)
    # true cosmology
    ax.plot(redshift,[omega_true.LuminosityDistance(zi)/1e3 for zi in redshift],linestyle='dashed',color='r', linewidth=.7)
    ax.plot(redshift,model50,color='k', linewidth=.7)
    # true redshift and distance of GW
#    ax.errorbar(ztrue, dl, yerr=deltadl, markersize=4,linewidth=1,color='r',fmt='o')
    ax.fill_between(redshift,model2p5,model97p5,facecolor='lightgray')
    ax.fill_between(redshift,model16,model84,facecolor='lightseagreen')
    ax.set_xlabel(r"z", fontsize=15)
    ax.set_ylabel(r"$d_L$/Gpc", fontsize=15)
    ax.set_xlim(np.min(redshift)*0.95,0.7)
    ax.set_ylim(0.0,2.5)
    ax.tick_params(labelsize=13)
    fig.savefig(os.path.join(opts.outdir,'regression.pdf'),bbox_inches='tight')
#    plt.close(fig)

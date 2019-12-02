#!/usr/bin/env python

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from optparse import OptionParser
import sys
import readdata
from dpgmm import *
import multiprocessing as mp
from scipy.special import logsumexp
from cosmology import *
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import dill as pickle

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

def init_plotting():
    plt.rcParams['figure.figsize'] = (3.4, 3.4)
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

    return adHeights

def initialise_dpgmm(dims,posterior_samples):
    model = DPGMM(dims)
    for point in posterior_samples:
        model.add(point)

    model.setPrior()
    model.setThreshold(1e-4)
    model.setConcGamma(10.0,0.01)
    return model

def compute_dpgmm(model,max_sticks=16):
    solve_args = [(nc, model) for nc in range(1, max_sticks+1)]
    solve_results = pool.map(solve_dpgmm, solve_args)
    scores = np.array([r[1] for r in solve_results])
    model = (solve_results[scores.argmax()][-1])
    print("best model has ",scores.argmax()+1,"components")
    return model.intMixture()

def evaluate_grid(density,x,y):
    d = len(x)*len(y)
    sys.stderr.write("computing log posterior for %d grid points\n"%d)
    sample_args = ((density,xi,yi) for xi in x for yi in y)
    results = pool.map(sample_dpgmm, sample_args, 16)
    return np.array([r for r in results]).reshape(len(x),len(y))

def sample_dpgmm(args):
    (dpgmm,x,y) = args
    logPs = [prob.logProb([x,y]) for ind,prob in enumerate(dpgmm[1])]
    return logsumexp(logPs,b=dpgmm[0])

def solve_dpgmm(args):
    (nc, model) = args
    for _ in range(nc-1): model.incStickCap()
    try:
        it = model.solve(iterCap=1024)
        return (model.stickCap, model.nllData(), model)
    except:
        return (model.stickCap, -np.inf, model)

def logit(x,xm,xM):
    return np.log((x-xm)/(xM-x))

def logjacobian(x,xm,xM):
    j = 1./(x-xm)+1./(xM-x)
    return np.log(np.abs(j))

def renormalise(logpdf,dx,dy):
    pdf = np.exp(logpdf)
    return pdf/(pdf*dx*dy).sum()

def marginalise(pdf,dx,axis):
    return np.sum(pdf*dx,axis=axis)

def deceleration_parameter(om):
    return 0.5*om-(1.0-om)

truths = {'h':0.73,'om':0.25,'ol':0.75,'w0':-1,'w1':0}

if __name__=="__main__":
    parser=OptionParser()
    parser.add_option('-o','--out',action='store',type='string',default=None,help='Output folder', dest='output')
    parser.add_option('-d',action='store',type='string',default=None,help='data folder', dest='data')
    parser.add_option('-c',action='store',type='string',default=None,help='source class (MBH, EMRI, sBH)', dest='source_class')
    parser.add_option('-m',action='store',type='string',default='LambdaCDM',help='model (LambdaCDM, LambdaCDMDE)', dest='model')
    parser.add_option('-r',action='store',type='int',default=0,help='realisation number', dest='realisation')
    parser.add_option('-p',action='store',type='int',default=0,help='unpickle precomputed posteriors', dest='pickle')
    parser.add_option('-N',action='store',type='int',default=None,help='Number of bins for the grid sampling', dest='N')
    parser.add_option('-e',action='store',type='int',default=None,help='Number of events to combine', dest='Nevents')
    parser.add_option('--nlive',action='store',type='int',default=5000,help='Number of live points', dest='nlive')
    (options,args)=parser.parse_args()

    out_folder = os.path.join(options.output,str(options.realisation))
    os.system("mkdir -p %s"%out_folder)
    np.random.seed(options.realisation)

    events = readdata.read_event(options.source_class, options.data, None)

    N = len(events)
#    if options.source_class == "EMRI":
#        N = np.int(np.random.poisson(len(events)*4./10.))#len(events)
#        events = np.random.choice(events,size = N,replace=False)

    omega_true = CosmologicalParameters(truths['h'],
                                        truths['om'],
                                        truths['ol'],
                                        truths['w0'],
                                        truths['w1'])

    Nbins = options.N
    joint_posterior = np.zeros((Nbins,Nbins),dtype=np.float64)#-2.0*np.log(Nbins)
    
    cls = []
    cls_om = []
    h_joint_cls = []
    om_joint_cls = []
    h_cdfs = []
    h_pdfs = []
    om_cdfs = []
    all_posteriors = []
    redshift_posteriors = {}
    pool = mp.Pool(mp.cpu_count())

    om_min, om_max = 0.04, 0.5
    h_min, h_max   = 0.5, 1.0
    dx   = (h_max-h_min)/Nbins
    dy   = (om_max-om_min)/Nbins
    x_flat = np.linspace(h_min+dx/2,h_max-dx/2,Nbins)#
    y_flat = np.linspace(om_min+dy/2,om_max-dy/2,Nbins)#
    X,Y = np.meshgrid(x_flat,y_flat)
    lX,lY = np.meshgrid(logit(x_flat,h_min,h_max),logit(y_flat,om_min,om_max))
    logjacobian_factor = logjacobian(X,h_min,h_max)+logjacobian(Y,om_min,om_max)
    
    from cpnest import nest2pos
    init_plotting()
    # sort the events by increasing ID
    if options.realisation == 0: events = sorted(events, key=lambda x: x.ID)
    if options.Nevents is None: options.Nevents = len(events)
    sys.stderr.write("Selected %d events for combination analysis\n"%options.Nevents)
    for k,e in enumerate(events):
    
        folder_name = "EVENT_1%03d"%e.ID
        print("processing ",folder_name)
        posteriors = nest2pos.draw_posterior_many([np.genfromtxt(os.path.join(options.data,folder_name+"/chain_{}_1234.txt".format(options.nlive)),
                                                   names=True)],
                                                   [options.nlive],
                                                   verbose=False)

        if options.pickle == False:
            
            redshift_posteriors['z%d'%e.ID] = posteriors['z%d'%e.ID]
            rvs = np.column_stack((logit(posteriors['h'],h_min,h_max),logit(posteriors['om'],om_min,om_max)))
            cls.append(np.percentile(posteriors['h'],[5,50,95]))
            cls_om.append(np.percentile(posteriors['om'],[5,50,95]))
            model = initialise_dpgmm(2,rvs)
            logdensity = compute_dpgmm(model,max_sticks=16)
            single_posterior  = evaluate_grid(logdensity,logit(x_flat,h_min,h_max),logit(y_flat,om_min,om_max))
            #single_posterior += logjacobian_factor

            joint_posterior += single_posterior
            joint_posterior  = np.log(renormalise(joint_posterior,dx,dy))
            
            all_posteriors.append(joint_posterior)

            pickle.dump(logdensity,open(os.path.join(out_folder,"model_log_post_%d.p"%k),"wb"))
            pickle.dump(joint_posterior,open(os.path.join(out_folder,"joint_log_post_%d.p"%k),"wb"))
            pickle.dump(single_posterior,open(os.path.join(out_folder,"log_post_%d.p"%k),"wb"))
        else:
            joint_posterior = pickle.load(open(os.path.join(out_folder,"joint_log_post_%d.p"%k),"rb"))
            single_posterior = pickle.load(open(os.path.join(out_folder,"log_post_%d.p"%k),"rb"))

        f = plt.figure(dpi=256)
        ax = f.add_subplot(111)
        C = ax.contourf(lX, lY, single_posterior.T, 100, cmap = matplotlib.cm.seismic, zorder=1)
        plt.colorbar(C)
        levs = np.sort(FindHeightForLevel(joint_posterior.T,[0.68,0.95]))
        ax.contour(lX,lY,joint_posterior.T, levs, colors='w', linewidths = 0.5, zorder=2)
        levs = np.sort(FindHeightForLevel(single_posterior.T,[0.68,0.95]))
        ax.contour(lX,lY,single_posterior.T, levs, colors='k', linewidths = 0.5, linestyles = 'solid', zorder=3)
        
        if False:
            ax.scatter(logit(posteriors['h'],h_min,h_max),logit(posteriors['om'],om_min,om_max), c='k', s=0.01, marker='.', alpha = 0.5)
        
        ax.axvline(logit(0.73,h_min,h_max),color='k',linestyle='dashed',lw=0.5)
        ax.axhline(logit(0.25,om_min,om_max),color='k',linestyle='dashed',lw=0.5)
        ax.set_xlabel(r"$\mathrm{logit}(h)$")
        ax.set_ylabel(r"$\mathrm{logit}(\Omega_m)$")
        plt.title(r'Event {0}'.format(k))
        f.savefig(os.path.join(out_folder,"log_post_%004d.png"%k),bbox_inches='tight')
        plt.close()
        
        # plot the marginals
        normalised_pdf = renormalise(joint_posterior+logjacobian_factor,dx,dy)
        
        f = plt.figure(dpi=256)
        ax = f.add_subplot(111)
        p  = marginalise(normalised_pdf,dy,1)
        ax.plot(x_flat, p, color='k',linestyle='solid',lw=1.5)
        p  = marginalise(normalised_pdf,dy,1)
        ax.plot(x_flat, p, color='k',linestyle='dashed',lw=0.25)
        ax.hist(posteriors['h'], density = True, alpha = 0.5, bins = 50)
        ax.axvline(0.73, color='r', linestyle='dotted')
        ax.set_xlim(0.5,1.0)
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(r"$p(h|DI)$")
        plt.title(r'Event {0}'.format(k))
        f.savefig(os.path.join(out_folder,"post_h_%004d.png"%k),bbox_inches='tight')
        plt.close()

        f = plt.figure(dpi=256)
        ax = f.add_subplot(111)
        p  = marginalise(normalised_pdf,dx,0)
        ax.plot(y_flat, p, color='k',linestyle='solid',lw=1.5)
        p  = marginalise(normalised_pdf,dx,0)
        ax.plot(y_flat, p, color='k',linestyle='dashed',lw=0.25)
        ax.hist(posteriors['om'], density = True, alpha = 0.5, bins = 50)
        ax.axvline(0.25, color='r', linestyle='dotted')
        ax.set_xlim(0.04,0.5)
        ax.set_xlabel(r"$\Omega_m$")
        ax.set_ylabel(r"$p(\Omega_m|DI)$")
        plt.title(r'Event {0}'.format(k))
        f.savefig(os.path.join(out_folder,"post_om_%004d.png"%k),bbox_inches='tight')
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        levs = np.sort(FindHeightForLevel(joint_posterior.T+logjacobian_factor,[0.68,0.95]))
        C = ax.contour(X,Y,joint_posterior.T+logjacobian_factor,levs,linewidths=0.75,colors='black')
        ax.grid(alpha=0.5,linestyle='dotted')
        ax.axvline(0.73,color='k',linestyle='dashed',lw=0.5)
        ax.axhline(0.25,color='k',linestyle='dashed',lw=0.5)
        ax.set_xlabel(r"$H_0/100\,km\,s^{-1}\,Mpc^{-1}$",fontsize=18)
        ax.set_ylabel(r"$\Omega_m$",fontsize=18)
        plt.savefig(os.path.join(out_folder,"joint_posterior_{0}.pdf".format(k)),bbox_inches='tight')
        plt.close()
        if k==options.Nevents-1: break

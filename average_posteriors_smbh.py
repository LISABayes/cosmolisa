import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from optparse import OptionParser
import sys
import readdata
from dpgmm import *
import multiprocessing as mp
from scipy.special import logsumexp
from cosmology import *
import matplotlib
import matplotlib.pyplot as plt
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
    model.setConcGamma(1.0,1.0)
    return model

def compute_dpgmm(model,max_sticks=16):
    solve_args = [(nc, model) for nc in range(1, max_sticks+1)]
    solve_results = pool.map(solve_dpgmm, solve_args)
    scores = np.array([r[1] for r in solve_results])
    model = (solve_results[scores.argmax()][-1])
    print("best model has ",scores.argmax()+1,"components")
    return model.intMixture()

def evaluate_grid(density,x,y):
    sys.stderr.write("computing log posterior for %d grid points\n"%(len(x)*len(y)))
    sample_args = ((density,xi,yi) for xi in x for yi in y)
    results = pool.map(sample_dpgmm, sample_args)
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

def rescaled_om(om,min_om,max_om):
    return (om - min_om)/(max_om-min_om)

def logit(x,xm,xM):
    return np.log((x-xm)/(xM-x))

def logjacobian(x,xm,xM):
    y = logit(x,xm,xM)
    j = np.abs(1./(x-xm)+1./(xM-x))
    return np.log(np.abs(j))

def renormalise(logpdf,dx,dy):
    pdf = np.exp(logpdf)
    return pdf/(pdf*dx*dy).sum()

def marginalise(pdf,dx,axis):
    return np.sum(pdf*dx,axis=axis)


if __name__=="__main__":
    parser=OptionParser()
    parser.add_option('-o','--out',action='store',type='string',default=None,help='Output folder', dest='output')
    parser.add_option('-d',action='store',type='string',default=None,help='data folder', dest='data')
    parser.add_option('-m',action='store',type='string',default='LambdaCDM',help='model (LambdaCDM, LambdaCDMDE, DE, CLambdaCDM)', dest='model', metavar='model')
    parser.add_option('-N',action='store',type='int',default=None,help='Number of bins for the grid sampling', dest='N')
    (options,args)=parser.parse_args()

    out_folder = options.output
    os.system("mkdir -p %s"%out_folder)

    catalogs = [c for c in os.listdir(options.data) if 'cat' in c]

    omega_true = CosmologicalParameters(0.73, 0.25, 0.75, -1.0, 0.0)
    omega_injected = (0.73, 0.25, 0.75, -1.0, 0.0)

    Nbins = options.N
    pool = mp.Pool(mp.cpu_count())

    if options.model == "LambdaCDM":
        x_flat = np.linspace(0.5,1.0,Nbins)
        y_flat = np.linspace(0.04,0.5,Nbins)
        dx = np.diff(x_flat)[0]
        dy = np.diff(y_flat)[0]
        X,Y = np.meshgrid(x_flat,y_flat)
    elif options.model == "DE":
        x_flat = np.linspace(-3.0,-0.3,Nbins)
        y_flat = np.linspace(-1.0,1.0,Nbins)
        dx = np.diff(x_flat)[0]
        dy = np.diff(y_flat)[0]
        X,Y = np.meshgrid(x_flat,y_flat)

    from cpnest import nest2pos
    for i,c in enumerate(catalogs):
        print("\nprocessing",c)
        samples = np.genfromtxt(os.path.join(options.data,c+"/chain_5000_1234.txt"),names=True)
        posteriors = nest2pos.draw_posterior_many([samples], [5000], verbose=False)
        if options.model == "LambdaCDM":
            if i==0:
                p1 = posteriors['h'][::5]
                p2 = posteriors['om'][::5]
            else:
                p1 = np.concatenate((p1,posteriors['h'][::5]))
                p2 = np.concatenate((p2,posteriors['om'][::5]))
            print('elements {0} {1}'.format(len(p1),len(posteriors['h'])))
        elif options.model == "DE":
            if i==0:
                p1 = posteriors['w0'][::5]
                p2 = posteriors['w1'][::5]
            else:
                p1 = np.concatenate((p1,posteriors['w0'][::5]))
                p2 = np.concatenate((p2,posteriors['w1'][::5]))
            print('elements {0} {1}'.format(len(p1),len(posteriors['w0'])))
    model = initialise_dpgmm(2,np.column_stack((p1,p2)))
    logdensity = compute_dpgmm(model,max_sticks=8)
    single_posterior = evaluate_grid(logdensity,x_flat,y_flat)
    pickle.dump(single_posterior,open(os.path.join(options.output,"average_posterior_{0}.p".format(model)),"wb"))
    init_plotting()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    levs = np.sort(FindHeightForLevel(single_posterior.T,[0.68,0.95]))
    ax.contourf(X,Y,single_posterior.T,100, cmap = matplotlib.cm.gray_r, alpha = 0.5, zorder=1)
    C = ax.contour(X,Y,single_posterior.T,levs,linewidths=0.75,colors='white', zorder = 22, linestyles = 'dashed')
    C = ax.contour(X,Y,single_posterior.T,levs,linewidths=1.0,colors='black')
    ax.grid(alpha=0.5,linestyle='dotted')
    if options.model == "LambdaCDM":
        ax.axvline(omega_injected[0],color='k',linestyle='dashed',lw=0.5)
        ax.axhline(omega_injected[1],color='k',linestyle='dashed',lw=0.5)
        ax.set_xlabel(r"$H_0/100\,km\,s^{-1}\,Mpc^{-1}$",fontsize=18)
        ax.set_ylabel(r"$\Omega_m$",fontsize=18)
    elif options.model == "DE":
        ax.axvline(omega_injected[3],color='k',linestyle='dashed',lw=0.5)
        ax.axhline(omega_injected[4],color='k',linestyle='dashed',lw=0.5)
        ax.set_xlabel(r"$w_0$",fontsize=18)
        ax.set_ylabel(r"$w_1$",fontsize=18)
    plt.savefig(os.path.join(options.output,"average_posterior_{0}.pdf".format(options.model)),bbox_inches='tight')
    plt.close()

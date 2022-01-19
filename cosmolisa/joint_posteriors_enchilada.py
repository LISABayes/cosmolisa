import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from optparse import OptionParser
import sys
import readdata
from dpgmm import *
import multiprocessing as mp
from scipy.special import logsumexp
from cosmolisa.cosmology import *
import matplotlib
import matplotlib.pyplot as plt
import dill as pickle
import corner

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
    parser.add_option('-o','--out', action='store', type='string', default=None,                 help='Output folder',                                  dest='output')
    parser.add_option('-d',         action='store', type='string', default=None,                 help='data folder_MBHB',                               dest='posteriors_MBHB')
    parser.add_option('-e',         action='store', type='string', default=None,                 help='data folder_EMRI',                               dest='posteriors_EMRI')
    parser.add_option('-m',         action='store', type='string', default='LambdaCDM',          help='model (LambdaCDM, DE)', dest='model', metavar='model')
    parser.add_option('-N',         action='store', type='int',    default=None,                 help='Number of bins for the grid sampling',           dest='N')
    parser.add_option('--name',     action='store', type='string', default='averaged_posterior', help='name of the averaged posterior file',            dest='name')
    parser.add_option('--corner',   action='store', type='int',    default=False,                help='Corner plot',                                    dest='corner')
    (options,args)=parser.parse_args()

    Nbins = options.N
    posteriors_MBHB = options.posteriors_MBHB
    posteriors_EMRI = options.posteriors_EMRI
    model = options.model
    pool = mp.Pool(mp.cpu_count())
    out_folder = options.output
    os.system("mkdir -p %s"%out_folder)

    truths = {'h':0.73,'om':0.25,'ol':0.75,'w0':-1.0,'w1':0.0}
    omega_true = CosmologicalParameters(0.73, 0.25, 0.75, -1.0, 0.0)

    Nbins = 256

    model         = 'DE' # 'LambdaCDM','DE'
    MBHB_model    = 'heavy_no_delays' # 'heavy_Q3','heavy_no_delays', 'popIII'
    EMRI_model    = 'M105' # 'M101', 'M105', 'M106'
    EMRI_duration = '4yrs' # '4yrs'

    data_folder_MBHB = '/home/laghi/data1-laghi/cosmolisa/Results/MBHB/MBHB_June2021/'
    data_folder_EMRI = '/home/laghi/data1-laghi/cosmolisa/Results/new_EMRI/'
    out_folder = 'enchilada_plots'
    os.system("mkdir -p %s"%out_folder)

    posteriors_MBHB = np.genfromtxt(os.path.join(data_folder_MBHB,'{MBHB_model}/{model}/averaged/averaged_posterior_{model}.dat'.format(MBHB_model=MBHB_model, model=model)),names=True)
    if EMRI_duration == '4yrs':
        posteriors_EMRI = np.genfromtxt(os.path.join(data_folder_EMRI,'{model}_SNR_100_reduced/{emri}_averaged/averaged_posterior.dat'.format(model=model,emri=EMRI_model)),names=True)
    else:
        posteriors_EMRI = np.genfromtxt(os.path.join(data_folder_EMRI,'{model}_SNR_100/{emri}_averaged/averaged_posterior.dat'.format(model=model,emri=EMRI_model)),names=True)

    if model == "LambdaCDM":
        x_flat = np.linspace(0.6,0.86,Nbins)
        y_flat = np.linspace(0.04,0.5,Nbins)
        p1 = 'h'
        p2 = 'om'
        posteriors_SOBH = np.array(Nbins*[norm(0.734,0.035).logpdf(x_flat)]).T # From Del Pozzo-Sesana-Klein, Tab. 1, run A2_50, LISA design N2A2M5L6
    elif model == "DE":
        x_flat = np.linspace(-3.0,-0.3,Nbins)
        y_flat = np.linspace(-1.0,1.0,Nbins)
        p1 = 'w0'
        p2 = 'w1'
    dx = np.diff(x_flat)[0]
    dy = np.diff(y_flat)[0]
    X,Y = np.meshgrid(x_flat,y_flat)

    single_posterior = []
    for post in [posteriors_MBHB,posteriors_EMRI]:

        model_dpgmm = initialise_dpgmm(2,np.column_stack((post[p1],post[p2])))
        logdensity = compute_dpgmm(model_dpgmm,max_sticks=8)
        single_posterior.append(evaluate_grid(logdensity,x_flat,y_flat))
        # pickle.dump(single_posterior,open(os.path.join(options.output,"average_posterior_{0}.p".format(model)),"wb"))
    colors = [matplotlib.cm.Greens,matplotlib.cm.Reds]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    joint_posterior = np.zeros((Nbins,Nbins))
    for p,c in zip(single_posterior,colors):
        levs = np.sort(FindHeightForLevel(p.T,[0.0, 0.68, 0.95]))
        # ax.contourf(X,Y,p.T,100, cmap = matplotlib.cm.gray_r, alpha = 0.5, zorder=1)
        # C = ax.contour(X,Y,p.T,levs,linewidths=0.75,colors=c, zorder = 22, linestyles = 'dashed')
        C = ax.contourf(X,Y,p.T,levs, alpha=0.5, cmap = c)
        C = ax.contour(X,Y,p.T,levs, linewidths=0.5, colors='k', alpha=0.4)

    joint_posterior = np.add(single_posterior[0],single_posterior[1])
    if model == 'LambdaCDM':
        levs = np.sort(FindHeightForLevel(posteriors_SOBH.T,[0.0, 0.68,0.95]))
        C = ax.contourf(X,Y,posteriors_SOBH.T,levs, alpha=0.5, cmap = matplotlib.cm.Blues, zorder=0)
        C = ax.contour(X,Y,posteriors_SOBH.T,levs, linewidths=0.5, colors='k', alpha=0.4, zorder=0)
        joint_posterior = np.add(joint_posterior,posteriors_SOBH)
        
    levs_joint = np.sort(FindHeightForLevel(joint_posterior.T,[0.0, 0.68,0.95]))
    C = ax.contourf(X,Y,joint_posterior.T,levs_joint, alpha=0.75, cmap = matplotlib.cm.gray_r, zorder=12)
    C = ax.contour(X,Y,joint_posterior.T,levs_joint, linewidths=0.5, colors='k', zorder=12, alpha=0.4)


    ax.grid(alpha=0.5,linestyle='dotted')
    if model == "LambdaCDM":
        ax.axvline(truths['h'],color='k',linestyle='dashed',lw=0.8, zorder=15)
        ax.axhline(truths['om'],color='k',linestyle='dashed',lw=0.8, zorder=15)
        ax.set_xlabel(r"$H_0/100\,km\,s^{-1}\,Mpc^{-1}$",fontsize=18)
        ax.set_ylabel(r"$\Omega_m$",fontsize=18)
    elif model == "DE":
        ax.axvline(truths['w0'],color='k',linestyle='dashed',lw=0.8, zorder=15)
        ax.axhline(truths['w1'],color='k',linestyle='dashed',lw=0.8, zorder=15)
        ax.set_xlabel(r"$w_0$",fontsize=18)
        ax.set_ylabel(r"$w_a$",fontsize=18)
        ax.set_xlim(-1.5,-0.5)
    plt.savefig(os.path.join(out_folder,"joint_posteriors_enchilada_{model}_{mbhb}_4yrs_{emri}_{dur}.pdf".format(model=model,mbhb=MBHB_model,emri=EMRI_model,dur=EMRI_duration)),bbox_inches='tight')
    plt.close()

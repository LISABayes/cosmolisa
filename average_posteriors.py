import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from optparse import OptionParser
import sys
import readdata
from dpgmm import *
import multiprocessing as mp
from scipy.misc import logsumexp
from cosmology import *
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib
import matplotlib.cm as cm
import cPickle as pickle

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

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

if __name__ == "__main__":
    parser=OptionParser()
    parser.add_option('-o','--out',action='store',type='string',default=None,help='Output folder', dest='output')
    parser.add_option('--d-emri',action='store',type='string',default=None,help='EMRI data folder', dest='data_emri')
    parser.add_option('--d-sBH',action='store',type='string',default=None,help='sBH data folder', dest='data_sbh')
    parser.add_option('--d-SMBH',action='store',type='string',default=None,help='SMBH data folder', dest='data_smbh')
    parser.add_option('--r-emri',action='store',type='int',default=None,help='number of EMRI realisations', dest='r_emri')
    parser.add_option('--r-sBH',action='store',type='int',default=None,help='number of sBH realisations', dest='r_sbh')
    parser.add_option('--r-SMBH',action='store',type='int',default=None,help='number of SMBH realisations', dest='r_smbh')
    parser.add_option('-m',action='store',type='string',default='LambdaCDM',help='model (LambdaCDM, LambdaCDMDE)', dest='model')
    (options,args)=parser.parse_args()

    eps = 1e-3
    x_flat = np.linspace(0.5+eps,1.0-eps,Nbins)
    y_flat = np.linspace(0.04+eps,1.0-eps,Nbins)
    dx = np.diff(x_flat)[0]
    dy = np.diff(y_flat)[0]
    X,Y = np.meshgrid(x_flat,y_flat)
    
    # the list of posteriors
    logpEMRI = []
    logpSMBH = []
    logpSBH  = []

    # read in the EMRI posteriors
    in_folder = options.data_emri
    for realisation in range(1,options.r_emri):
        print("searching realisation %d"%realisation)
        all_files = os.listdir(os.path.join(in_folder,"%d"%realisation))
        posfiles = [f for f in all_files if "joint_log_post" in f]
        idlast = np.argmax([int((f.split("_")[-1]).split(".")[0]) for f in posfiles])
        print("loading %s"%posfiles[idlast])
        logpEMRI.append(pickle.load(open(os.path.join(in_folder,"%d/%s"%(realisation,posfiles[idlast])),"r")))

    logpEMRI = np.squeeze(np.percentile(np.array(logpEMRI),[50],axis=0))

    # read in the EMRI posteriors
    in_folder = options.data_sbh
    for realisation in range(1,options.r_sb):
        print("searching realisation %d"%realisation)
        all_files = os.listdir(os.path.join(in_folder,"%d"%realisation))
        posfiles = [f for f in all_files if "joint_log_post" in f]
        idlast = np.argmax([int((f.split("_")[-1]).split(".")[0]) for f in posfiles])
        print("loading %s"%posfiles[idlast])
        logpSBH.append(pickle.load(open(os.path.join(in_folder,"%d/%s"%(realisation,posfiles[idlast])),"r")))

    logpSBH = np.squeeze(np.percentile(np.array(logpSBH),[50],axis=0))
    
    # read in the SMBH average posteriors
    in_folder = options.data_smbh
    all_files = os.listdir(in_folder)
    posfiles = [f for f in all_files if "log_post" in f]

    for p in posfiles:
        print("loading %s"%p)
        logpSMBH.append(pickle.load(open(os.path.join(in_folder,"%s"%p),"r")))
    logpSMBH = np.squeeze(np.percentile(np.array(logpSMBH),[50],axis=0))

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.axvline(0.73,linestyle='dashed',color='0.5')
    ax.axhline(0.25,linestyle='dashed',color='0.5')
    levs = FindHeightForLevel(logpEMRI, [0.68,0.95])
    C = ax.contour(X,Y,logpEMRI.T,levels=np.sort(levs),colors='r',linewidths=1.5)
    levs = FindHeightForLevel(logpSBH, [0.68,0.95])
    C = ax.contour(X,Y,logpSBH.T,levels=np.sort(levs),colors='g',linewidths=1.5)
    levs = FindHeightForLevel(logpSMBH, [0.68,0.95])
    C = ax.contour(X,Y,logpSMBH.T,levels=np.sort(levs),colors='b',linewidths=1.5)

    logPall=logpEMRI+logpSMBH+logpSBH
    levs = FindHeightForLevel(logPall, [0.68,0.95])
    C = ax.contour(X,Y,logPall.T,levels=np.sort(levs),colors='k',linewidths=2.0)
    plt.grid(linestyle='dotted',alpha=0.5)

    red_line = mlines.Line2D([], [], color = 'r', marker = '.', markersize=1, label=r"$\mathrm{EMRI}$" )
    green_line = mlines.Line2D([], [], color = 'g', marker = '.', markersize=1, label=r"$\mathrm{sMBH}$" )
    blue_line = mlines.Line2D([], [], color = 'b', marker = '.', markersize=1, label=r"$\mathrm{SMBH}$" )
    black_line = mlines.Line2D([], [], color = 'k', marker = '.', markersize=1, label=r"$\mathrm{total}$" )
    ax.legend(handles = [red_line,green_line,blue_line], loc='upper right')#
    ax.set_ylabel(r"$\Omega_m$",fontsize=18)
    plt.xlim(0.65,0.8)
    plt.ylim(0.1,0.6)
    ax.set_xlabel(r"$H_0/100\,km\,s^{-1}\,Mpc^{-1}$",fontsize=18)
    plt.savefig('best_posteriors_for_all.pdf',bbox_inches='tight')

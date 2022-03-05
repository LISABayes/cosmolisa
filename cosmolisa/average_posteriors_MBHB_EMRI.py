import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import corner
import h5py
from optparse import OptionParser

COSMOLISA_PATH = os.getcwd()
sys.path.insert(1, os.path.join(COSMOLISA_PATH,'DPGMM'))
from dpgmm import *
sys.path.insert(1, COSMOLISA_PATH)

import multiprocessing as mp
from scipy.special import logsumexp

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

if __name__=="__main__":
    parser=OptionParser()
    parser.add_option('-o','--out',      action='store', type='string', default=None,        help='Output folder',                        dest='output')
    parser.add_option('-m',              action='store', type='string', default='LambdaCDM', help='model (LambdaCDM, DE)',                dest='model')
    parser.add_option('-N',              action='store', type='int',    default=256,         help='Number of bins for the grid sampling', dest='N')
    parser.add_option('--d_EMRI',        action='store', type='string', default=None,        help='EMRI data folder',                     dest='data_EMRI')
    parser.add_option('--d_MBHB',        action='store', type='string', default=None,        help='MBHB data folder',                     dest='data_MBHB')
    parser.add_option('--cat_name_EMRI', action='store', type='string', default=False,       help='EMRI catalog name',                    dest='cat_name_EMRI')
    parser.add_option('--cat_name_MBHB', action='store', type='string', default=False,       help='MBHB catalog name',                    dest='cat_name_MBHB')
    (options,args)=parser.parse_args()

    out_folder            = options.output
    model                 = options.model
    Nbins                 = options.N
    EMRI_path             = options.data_EMRI
    MBHB_path             = options.data_MBHB
    catalog_name_EMRI     = options.cat_name_EMRI
    catalog_name_MBHB     = options.cat_name_MBHB
    pool                  = mp.Pool(mp.cpu_count())

    # Read or not reduced (in years of observation) catalogs
    if 'reduced' in EMRI_path:
        reduced_string = '_reduced'
    else:
        reduced_string = ''

    os.system("mkdir -p %s"%out_folder)

    truths      = {'h':0.73,'om':0.25,'ol':0.75,'w0':-1.0,'w1':0.0}
    all_sources = {
                   "EMRI": {'p1': p1_EMRI, 'p2': p2_EMRI}, 
                   "MBHB": {'p1': p1_MBHB, 'p2': p2_MBHB}, 
                   }
    colors_dict = {"EMRI": {'68%': 'orangered', '90%': 'orangered'}, #orangered lightsalmon
                   "MBHB": {'68%': 'dodgerblue', '90%': 'dodgerblue'}, #dodgerblue lightblue
                   "combined": {'68%': "black", '90%': "black"}
                   }

    # First read average posteriors already averaged over different realisations. Do it for each source. 
    for source in all_sources.keys():
        if (source == 'EMRI'):
            print("\nReading {} averaged posterior stored in {}".format(source, os.path.join(EMRI_path,catalog_name_EMRI+"_averaged/averaged_posterior_{}_{}.dat".format(model,catalog_name_EMRI))))
            posteriors = np.genfromtxt(os.path.join(EMRI_path,catalog_name_EMRI+"_averaged/averaged_posterior_{}_{}.dat".format(model,catalog_name_EMRI)),names=True)
            if model == "LambdaCDM":
                p1_EMRI = posteriors['h']
                p2_EMRI = posteriors['om']
            elif model == "DE":
                p1_EMRI = posteriors['w0']
                p2_EMRI = posteriors['w1']
            
        elif (source == 'MBHB'):
            print("\nReading {} averaged posterior stored in {}".format(source, os.path.join(MBHB_path,catalog_name_MBHB,"{cat}_averaged/averaged_posterior_{mod}_{cat}.dat".format(mod=model,cat=catalog_name_MBHB))))
            posteriors = np.genfromtxt(os.path.join(MBHB_path,catalog_name_MBHB,"{cat}_averaged/averaged_posterior_{mod}_{cat}.dat".format(mod=model,cat=catalog_name_MBHB)),names=True)
            if model == "LambdaCDM":
                p1_MBHB = posteriors['h']
                p2_MBHB = posteriors['om']
            elif model == "DE":
                p1_MBHB = posteriors['w0']
                p2_MBHB = posteriors['w1']

    joint_posterior = np.zeros((Nbins,Nbins),dtype=np.float64)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(alpha=0.5,linestyle='dotted')

    if model == "LambdaCDM":
        x_flat = np.linspace(0.6,0.86,Nbins)
        y_flat = np.linspace(0.04,0.5,Nbins)
    elif model == "DE":
        x_flat = np.linspace(-1.5,-0.3,Nbins)
        y_flat = np.linspace(-1.0,1.0,Nbins)
    else:
        sys.exit("DPGMM only accepts 2D models (LambdaCDM, DE), Exiting.")

    dx = np.diff(x_flat)[0]
    dy = np.diff(y_flat)[0]
    X,Y = np.meshgrid(x_flat, y_flat)

    # Print statistics and prepare samples to be combined
    for source,source_samp in all_sources.items():
        print("\n\n"+source)
        p1 = source_samp['p1'][::10]
        p2 = source_samp['p2'][::10]
        # Compute and save .05, .16, .5, .84, .95 quantiles for each source.
        if model == "LambdaCDM":
            p1_name, p2_name = 'h','om'
        elif model == "DE":
            p1_name, p2_name = 'w0','w1'

        # file_path = os.path.join(options.output,'quantiles_{}_{}{}.txt'.format(model, catalog_name, reduced_string))
        # print("Will save .5, .16, .50, .84, .95 quantiles in {}".format(output_dir))
        # sys.stdout = open(file_path, "w+")

        p1_ll,p1_l,p1_median,p1_h,p1_hh = np.percentile(p1,[5.0,16.0,50.0,84.0,95.0],axis = 0)
        p2_ll,p2_l,p2_median,p2_h,p2_hh = np.percentile(p2,[5.0,16.0,50.0,84.0,95.0],axis = 0)

        p1_inf_68, p1_sup_68 = p1_median - p1_l,  p1_h - p1_median
        p1_inf_90, p1_sup_90 = p1_median - p1_ll, p1_hh - p1_median
        p2_inf_68, p2_sup_68 = p2_median - p2_l,  p2_h - p2_median
        p2_inf_90, p2_sup_90 = p2_median - p2_ll, p2_hh - p2_median

        p1_credible68 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(p1_inf_68, p1_median, p1_sup_68)
        p1_credible90 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(p1_inf_90, p1_median, p1_sup_90)
        p2_credible68 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(p2_inf_68, p2_median, p2_sup_68)
        p2_credible90 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(p2_inf_90, p2_median, p2_sup_90)

        print("{} 68% CI:".format(p1_name))
        print(p1_credible68)
        print("{} 90% CI:".format(p1_name))
        print(p1_credible90)
        print("\n{} 68% CI:".format(p2_name))
        print(p2_credible68)
        print("{} 90% CI:".format(p2_name))
        print(p2_credible90)

        print("\nAdding source", source, "to the plot.")

        model_dp = initialise_dpgmm(2,np.column_stack((p1,p2)))
        logdensity = compute_dpgmm(model_dp,max_sticks=8)
        single_posterior = evaluate_grid(logdensity,x_flat,y_flat)
        joint_posterior += single_posterior
        # pickle.dump(single_posterior,open(os.path.join(options.output,"average_posterior_dpgmm_{0}.p".format(model_dp)),"wb"))
        levs = np.sort(FindHeightForLevel(single_posterior.T,[0.68,0.90]))
        C = ax.contour(X,Y,single_posterior.T,levs,linewidths=1.5,colors=[colors_dict[source]['68%'],colors_dict[source]['90%']], zorder = 5, linestyles = 'solid')

    levs = np.sort(FindHeightForLevel(joint_posterior.T,[0.68,0.90]))
    C = ax.contour(X,Y,joint_posterior.T,levs,linewidths=2.0,colors=[colors_dict['combined']['68%'],colors_dict['combined']['90%']], zorder = 10, linestyles = 'solid')

    if model == "LambdaCDM":
        ax.axvline(truths['h'],color='k',linestyle='dashed',lw=0.5)
        ax.axhline(truths['om'],color='k',linestyle='dashed',lw=0.5)
        ax.set_xlabel(r"$h$",fontsize=18)
        ax.set_ylabel(r"$\Omega_m$",fontsize=18)
    elif model == "DE":
        ax.axvline(truths['w0'],color='k',linestyle='dashed',lw=0.5)
        ax.axhline(truths['w1'],color='k',linestyle='dashed',lw=0.5)
        ax.set_xlabel(r"$w_0$",fontsize=18)
        ax.set_ylabel(r"$w_a$",fontsize=18)
    plt.savefig(os.path.join(out_folder,"average_posterior_combined_{0}{1}_{2}_{3}_10samp.pdf".format(model, reduced_string, catalog_name_EMRI, catalog_name_MBHB)),bbox_inches='tight')
    plt.close()

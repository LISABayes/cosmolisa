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
from scipy.stats import rv_discrete

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

def marginalise(pdf, dx, axis):
    return np.sum(pdf*dx, axis=axis)

def renormalise(pdf, dx):
    return pdf / (pdf*dx).sum()

def par_dic(a1, a2):
    return {'p1': a1, 'p2': a2}

if __name__=="__main__":

    out_dir = './'
    model = 'LambdaCDM'
    Nbins = 1024
    pool = mp.Pool(mp.cpu_count())

    truths = {'h':0.73,'om':0.25,'ol':0.75,'w0':-1.0,'wa':0.0}
    fmt = "{{0:{0}}}".format('.3f').format
    colors_dict = {
                   "EMRI": 'red', #orangered lightsalmon
                   "MBHB": 'dodgerblue', #dodgerblue lightblue
                   "combined": "black"
                  }

    emri_catalog_list = ['M105'] #['M105', 'M101', 'M106']
    mbhb_catalog_list = ['popIII'] #['popIII', 'heavy_Q3', 'heavy_no_delays']

    for catalog_emri in emri_catalog_list:
        for catalog_mbhb in mbhb_catalog_list:

            catalog_name_EMRI = catalog_emri
            catalog_name_MBHB = catalog_mbhb
            EMRI_data_path = 
            MBHB_data_path = 

            # Read or not reduced (in years of observation) catalogs
            if 'reduced' in EMRI_data_path:
                reduced_string = '_reduced'
            else:
                reduced_string = ''

            out_folder = os.path.join(out_dir,"{0}{1}_{2}_{3}".format(model, reduced_string, catalog_name_EMRI, catalog_name_MBHB))
            os.system("mkdir -p %s"%out_folder)

            # First read average posteriors already averaged over different realisations. Do it for each source. 
            for source in ["EMRI", "MBHB"]:
                if (source == 'EMRI'):
                    print("\nReading {} averaged posterior stored in {}".format(source, os.path.join(EMRI_data_path, catalog_name_EMRI+"_averaged/averaged_posterior_{}_{}.dat".format(model, catalog_name_EMRI))))
                    posteriors = np.genfromtxt(os.path.join(EMRI_data_path, catalog_name_EMRI+"_averaged/averaged_posterior_{}_{}.dat".format(model, catalog_name_EMRI)), names=True)
                    if model == "LambdaCDM":
                        p1_EMRI = posteriors['h']
                        p2_EMRI = posteriors['om']
                    elif model == "DE":
                        p1_EMRI = posteriors['w0']
                        p2_EMRI = posteriors['w1']
                    
                elif (source == 'MBHB'):
                    print("\nReading {} averaged posterior stored in {}".format(source, os.path.join(MBHB_data_path, catalog_name_MBHB, "{cat}_averaged/averaged_posterior_{mod}_{cat}.dat".format(mod=model, cat=catalog_name_MBHB))))
                    posteriors = np.genfromtxt(os.path.join(MBHB_data_path,catalog_name_MBHB,"{cat}_averaged/averaged_posterior_{mod}_{cat}.dat".format(mod=model, cat=catalog_name_MBHB)), names=True)
                    if model == "LambdaCDM":
                        p1_MBHB = posteriors['h']
                        p2_MBHB = posteriors['om']
                    elif model == "DE":
                        p1_MBHB = posteriors['w0']
                        p2_MBHB = posteriors['w1']

            EMRI_dic = par_dic(p1_EMRI, p2_EMRI)
            MBHB_dic = par_dic(p1_MBHB, p2_MBHB)

            all_sources = {
                           "EMRI": EMRI_dic, 
                           "MBHB": MBHB_dic, 
                          }

            joint_posterior = np.zeros((Nbins,Nbins), dtype=np.float64)

            # Plot - Compute single and joint posteriors
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            ax.grid(alpha=0.5, linestyle='dotted')

            if model == "LambdaCDM":
                x_flat = np.linspace(0.6, 0.86, Nbins)
                y_flat = np.linspace(0.04, 0.5, Nbins)
            elif model == "DE":
                x_flat = np.linspace(-1.5, -0.3, Nbins)
                y_flat = np.linspace(-1.0, 1.0, Nbins)
            else:
                sys.exit("DPGMM only accepts 2D models (LambdaCDM, DE), Exiting.")

            dx = np.diff(x_flat)[0]
            dy = np.diff(y_flat)[0]
            X,Y = np.meshgrid(x_flat, y_flat)

            # Print statistics and prepare samples to be combined
            file_path = os.path.join(out_folder,'quantiles.txt')
            sys.stdout = open(file_path, "w+")
            print("{} {} {} {}".format(model, catalog_name_EMRI, catalog_name_MBHB, reduced_string))
            print("Will save .5, .16, .50, .84, .95 quantiles in {}".format(out_folder))

            for source,source_samp in all_sources.items():
                p1 = source_samp['p1'][::10]
                p2 = source_samp['p2'][::10]
                # Compute and save .05, .16, .5, .84, .95 quantiles for each source.
                if model == "LambdaCDM":
                    p1_name, p2_name = 'h','om'
                    p1_name_string, p2_name_string = r"$h$", r"$\Omega_m$"
                elif model == "DE":
                    p1_name, p2_name = 'w0','wa'
                    p1_name_string, p2_name_string = r"$w_0$", r"$w_a$"

                print("\n\n"+source)

                p1_ll, p1_l, p1_median, p1_h, p1_hh = np.percentile(p1, [5.0,16.0,50.0,84.0,95.0], axis=0)
                p2_ll, p2_l, p2_median, p2_h, p2_hh = np.percentile(p2, [5.0,16.0,50.0,84.0,95.0], axis=0)

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
                print(p2_credible90, "\n")

                model_dp = initialise_dpgmm(2,np.column_stack((p1,p2)))
                logdensity = compute_dpgmm(model_dp, max_sticks=8)
                single_posterior = evaluate_grid(logdensity, x_flat, y_flat)
                joint_posterior += single_posterior
                levs = np.sort(FindHeightForLevel(single_posterior.T, [0.0,0.68,0.90]))
                C = ax.contourf(X, Y, single_posterior.T, levs[:-1], colors=colors_dict[source], zorder=5, linestyles='solid', alpha=0.1)
                C = ax.contourf(X, Y, single_posterior.T, levs[1:], colors=colors_dict[source], zorder=5, linestyles='solid', alpha=0.3)
                C = ax.contour(X, Y, single_posterior.T, levs[:-1], linewidths=1., colors=colors_dict[source], zorder=6, linestyles='solid')
                C.collections[0].set_label(source)

            levs = np.sort(FindHeightForLevel(joint_posterior.T,[0.0,0.68,0.90]))
            C = ax.contourf(X, Y, joint_posterior.T, levs[:-1], colors='whitesmoke', zorder=10, linestyles='solid', alpha=0.85)
            C = ax.contourf(X, Y, joint_posterior.T, levs[1:], colors=colors_dict['combined'], zorder=10, linestyles='solid', alpha=0.3)
            C = ax.contour(X, Y, joint_posterior.T, levs[:-1], linewidths=2.0, colors=colors_dict['combined'], zorder=11, linestyles='solid')
            C.collections[0].set_label('combined')
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.legend(fontsize=18)

            if model == "LambdaCDM":
                ax.axvline(truths['h'], color='k', linestyle='dashed', lw=0.5)
                ax.axhline(truths['om'], color='k', linestyle='dashed', lw=0.5)
                ax.set_xlabel(p1_name_string,fontsize=18)
                ax.set_ylabel(p2_name_string,fontsize=18)

            elif model == "DE":
                ax.axvline(truths['w0'], color='k', linestyle='dashed', lw=0.5)
                ax.axhline(truths['wa'], color='k', linestyle='dashed', lw=0.5)
                ax.set_xlabel(p1_name_string, fontsize=18)
                ax.set_ylabel(p2_name_string, fontsize=18)
            plt.savefig(os.path.join(out_folder,"jp_combined{0}_10samp.pdf".format(reduced_string)), bbox_inches='tight')

            # Plot - Level contours of joint distribution
            fig_j = plt.figure(figsize=(10,8))
            ax_j = plt.axes()
            hs, lgs = [], []
            levs_long = np.concatenate(([0.05],[k for k in np.arange(0.1,1.0,0.1)]))
            colors_long = np.flip(['k', 'darkgrey', 'crimson', 'darkorange', 'gold', 'limegreen', 'darkturquoise', 'royalblue', 'mediumorchid', 'magenta'])
            for l,c in zip(levs_long,colors_long):
                cntr = ax_j.contour(X, Y, joint_posterior.T, levels = np.sort(FindHeightForLevel(joint_posterior.T, [l])), colors = c, linewidths=1.2)
                h,_ = cntr.legend_elements()
                hs.append(h[0])
                lgs.append(r'${0} \% \, CR$'.format(int(l*100.)))
            if model == "LambdaCDM":
                ax_j.axvline(truths['h'], color='k', linestyle='dashed', lw=0.5)
                ax_j.axhline(truths['om'], color='k', linestyle='dashed', lw=0.5)
                ax_j.set_xlabel(p1_name_string, fontsize=18)
                ax_j.set_ylabel(p2_name_string, fontsize=18)
                xlimits = [0.7,0.76]
                ylimits = [0.15,0.35]
                leg_loc = 'upper right'
            elif model == "DE":
                ax_j.axvline(truths['w0'], color='k', linestyle='dashed', lw=0.5)
                ax_j.axhline(truths['wa'], color='k', linestyle='dashed', lw=0.5)
                ax_j.set_xlabel(p1_name_string, fontsize=18)
                ax_j.set_ylabel(p2_name_string, fontsize=18)
                xlimits = [-1.3,-0.7]
                ylimits = [-1.,1.]
                leg_loc = 'lower left'
            plt.legend([x_h for x_h in hs], [lg for lg in lgs], loc=leg_loc, fontsize=14)
            plt.xlim(xlimits)
            plt.ylim(ylimits)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.savefig(os.path.join(out_folder,"jp_contour_levels{0}_10samp.pdf".format(reduced_string)),bbox_inches='tight')

            # Compute marginalised 1D PDFs statistics
            pdf_p1 = renormalise(marginalise(np.exp(joint_posterior.T), dy, axis=0), dx)
            pdf_p2 = renormalise(marginalise(np.exp(joint_posterior.T), dx, axis=1), dy)
            print("\n1D PDF normalisation {}: ".format(p1_name), marginalise(pdf_p1, dx, axis=0))
            print("1D PDF normalisation {}: ".format(p2_name), marginalise(pdf_p2, dy, axis=0))

            custm_p1 = rv_discrete(name='cutsm', values=(x_flat, pdf_p1*dx))
            custm_p2 = rv_discrete(name='cutsm', values=(y_flat, pdf_p2*dy))

            custm_p1_ll, custm_p1_l, custm_p1_median, custm_p1_h, custm_p1_hh = custm_p1.interval(.90)[0], custm_p1.interval(.68)[0], custm_p1.median(), custm_p1.interval(.68)[1], custm_p1.interval(.90)[1]
            custm_p2_ll, custm_p2_l, custm_p2_median, custm_p2_h, custm_p2_hh = custm_p2.interval(.90)[0], custm_p2.interval(.68)[0], custm_p2.median(), custm_p2.interval(.68)[1], custm_p2.interval(.90)[1]

            custm_p1_inf_68, custm_p1_sup_68 = custm_p1_median - custm_p1_l, custm_p1_h - custm_p1_median
            custm_p1_inf_90, custm_p1_sup_90 = custm_p1_median - custm_p1_ll, custm_p1_hh - custm_p1_median
            custm_p2_inf_68, custm_p2_sup_68 = custm_p2_median - custm_p2_l, custm_p2_h - custm_p2_median
            custm_p2_inf_90, custm_p2_sup_90 = custm_p2_median - custm_p2_ll, custm_p2_hh - custm_p2_median

            custm_p1_credible68 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(custm_p1_inf_68, custm_p1_median, custm_p1_sup_68)
            custm_p1_credible90 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(custm_p1_inf_90, custm_p1_median, custm_p1_sup_90)
            custm_p2_credible68 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(custm_p2_inf_68, custm_p2_median, custm_p2_sup_68)
            custm_p2_credible90 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(custm_p2_inf_90, custm_p2_median, custm_p2_sup_90)

            print("\n\njoint posterior")
            print("{} 68% CI:".format(p1_name))
            print(custm_p1_credible68)
            print("{} 90% CI:".format(p1_name))
            print(custm_p1_credible90)
            print("\n{} 68% CI:".format(p2_name))
            print(custm_p2_credible68)
            print("{} 90% CI:".format(p2_name))
            print(custm_p2_credible90)

            # Plot - 1D PDFs
            plt.figure(figsize=(8,8))
            plt.plot(x_flat, pdf_p1, c='k', linewidth=2.0)
            plt.axvline(custm_p1_median, linestyle='--', color='k', zorder=-1)
            plt.axvline(custm_p1_ll, linestyle='--', color='k', zorder=-1)
            plt.axvline(custm_p1_hh, linestyle='--', color='k', zorder=-1)
            plt.axvline(truths[p1_name], color='dodgerblue', linestyle='-', zorder=-1)
            plt.xlabel(p1_name_string, fontsize=18)
            plt.ylabel(r"$PDF$", fontsize=18)
            plt.xlim(xlimits)
            plt.ylim(bottom=0.0)
            plt.xticks(np.linspace(xlimits[0], xlimits[1], 11), fontsize=13)
            plt.yticks(fontsize=13)
            par_n = 'h' if model == 'LambdaCDM' else 'w_0' 
            plt.title(r'${0} = {{{1}}}_{{-{2}}}^{{+{3}}} \,\, (90 \% \, CI)$'.format(par_n, fmt(custm_p1_median), fmt(custm_p1_inf_90), fmt(custm_p1_sup_90)), fontsize=18)
            plt.savefig(os.path.join(out_folder,'{}.png'.format(p1_name)), bbox_inches='tight')

            plt.figure(figsize=(8,8))
            plt.plot(y_flat, pdf_p2, c='k', linewidth=2.0)
            plt.axvline(custm_p2_median, linestyle='--', color='k', zorder=-1)
            plt.axvline(custm_p2_ll, linestyle='--', color='k', zorder=-1)
            plt.axvline(custm_p2_hh, linestyle='--', color='k', zorder=-1)
            plt.axvline(truths[p2_name], color='dodgerblue', linestyle='-', zorder=-1)
            plt.xlabel(p2_name_string, fontsize=18)
            plt.ylabel(r"$PDF$", fontsize=18)
            plt.xlim(ylimits)
            plt.ylim(bottom=0.0)
            plt.xticks(np.linspace(ylimits[0], ylimits[1], 11), fontsize=13)
            plt.yticks(fontsize=13)
            par_n = '\Omega_m' if model == 'LambdaCDM' else 'w_a'
            plt.title(r'${0} = {{{1}}}_{{-{2}}}^{{+{3}}} \, (90 \% \, CI)$'.format(par_n, fmt(custm_p2_median), fmt(custm_p2_inf_90), fmt(custm_p2_sup_90)), fontsize=18)
            plt.savefig(os.path.join(out_folder,'{}.png'.format(p2_name)), bbox_inches='tight')

            plt.close()

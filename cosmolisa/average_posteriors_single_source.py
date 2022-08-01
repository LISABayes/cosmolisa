import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import corner
import h5py
from optparse import OptionParser

if __name__=="__main__":
    parser=OptionParser()
    parser.add_option('-o','--out',              action='store', type='string', default=None,        help='Output folder',                                                                                           dest='output')
    parser.add_option('-d',                      action='store', type='string', default=None,        help='data folder',                                                                                             dest='data')
    parser.add_option('-m',                      action='store', type='string', default='LambdaCDM', help='model (LambdaCDM, LambdaCDMDE, DE, CLambdaCDM)',                                                          dest='model', metavar='model')
    parser.add_option('-c',                      action='store', type='string', default='EMRI',      help='source class (MBHB, EMRI)',                                                                               dest='source')
    parser.add_option('-N',                      action='store', type='int',    default=256,         help='Number of bins for the grid sampling (only used for dpgmm)',                                              dest='N')
    parser.add_option('--dpgmm',                 action='store', type='int',    default=False,       help='DPGMM average plot',                                                                                      dest='dpgmm')
    parser.add_option('--cat_name',              action='store', type='string', default=False,       help='Catalog name',                                                                                            dest='cat_name')
    parser.add_option('--corner_68',             action='store', type='int',    default=False,       help='Corner plot bugged (only showing 68%CI)',                                                                 dest='corner_68')
    parser.add_option('--corner_90',             action='store', type='int',    default=False,       help='Corner plot without bug (90%CI)',                                                                         dest='corner_90')
    parser.add_option('--split_catalog',         action='store', type='int',    default=False,       help='Read samples from analyses where the catalog has been split',                                             dest='split_catalog')
    parser.add_option('--produce_averaged_post', action='store', type='int',    default=1,           help='Read different catalog realisations. If 0, read averaged posterior produced at the time of the analysis', dest='produce_averaged_post')
    (options,args)=parser.parse_args()

    dpgmm_average         = options.dpgmm
    catalog_name          = options.cat_name
    produce_averaged_post = options.produce_averaged_post
    corner_68             = options.corner_68
    corner_90             = options.corner_90
    split_catalog         = options.split_catalog
    out_folder            = os.path.join(options.output,catalog_name+'_averaged')

    # Read or not reduced (in years of observation) catalogs
    if 'reduced' in options.data:
        reduced_string = '_reduced'
    else:
        reduced_string = ''

    final_posterior_name = 'averaged_posterior_'+options.model+'_'+catalog_name+reduced_string+'.dat'

    os.system("mkdir -p %s"%out_folder)

    truths = {'h':0.673,'om':0.315,'ol':0.685,'w0':-1.0,'w1':0.0}
    # truths = {'h':0.73,'om':0.25,'ol':0.75,'w0':-1.0,'w1':0.0}

    # Average posteriors from different runs or read previously averaged posterior 
    if produce_averaged_post:
        if options.source == 'MBHB':
            catalogs = [c for c in os.listdir(options.data) if ('cat' in c and 'averaged' not in c)]
        elif options.source == 'EMRI':
            catalogs = [c for c in os.listdir(options.data) if (catalog_name in c and 'averaged' not in c and 'matrix' not in c)]
        print("Will read", len(catalogs), "catalogs")

        for i,c in enumerate(catalogs):
            print("\nprocessing", options.source, options.model, c)
            if (split_catalog):
                posteriors = np.genfromtxt(os.path.join(options.data,c,"{}_joint/samples.dat".format(options.model)), names=True)
            else:
                try:
                    filename = os.path.join(options.data,c,'CPNest','cpnest.h5')
                    h5_file = h5py.File(filename,'r')
                    posteriors = h5_file['combined'].get('posterior_samples')
                    print("Read .h5 file")
                except:
                    posteriors = np.genfromtxt(os.path.join(options.data,c+"/posterior.dat"), names=True)
                    print("Read .dat file")
            if options.model == "LambdaCDM":
                if i==0:
                    p1 = posteriors['h']
                    p2 = posteriors['om']
                else:
                    p1 = np.concatenate((p1,posteriors['h']))
                    p2 = np.concatenate((p2,posteriors['om']))
                print('Reading {0} samples from catalog {1} (total: {2})'.format(len(posteriors['h']), c, len(p1)))
            elif options.model == "CLambdaCDM":
                if i==0:
                    p1 = posteriors['h']
                    p2 = posteriors['om']
                    p3 = posteriors['ol']
                else:
                    p1 = np.concatenate((p1,posteriors['h']))
                    p2 = np.concatenate((p2,posteriors['om']))
                    p3 = np.concatenate((p3,posteriors['ol']))
                print('Reading {0} samples from catalog {1} (total: {2})'.format(len(posteriors['h']), c, len(p1)))
            elif options.model == "DE":
                if i==0:
                    p1 = posteriors['w0']
                    p2 = posteriors['w1']
                else:
                    p1 = np.concatenate((p1,posteriors['w0']))
                    p2 = np.concatenate((p2,posteriors['w1']))
                print('Reading {0} samples from catalog {1} (total: {2})'.format(len(posteriors['w0']), c, len(p1)))
    else:
        print("\nReading averaged posterior stored in {}".format(os.path.join(options.data,catalog_name+"_averaged/averaged_posterior.dat")))
        posteriors = np.genfromtxt(os.path.join(options.data,catalog_name+"_averaged/averaged_posterior.dat"),names=True)
        if options.model == "LambdaCDM":
            p1 = posteriors['h']
            p2 = posteriors['om']
        elif options.model == "CLambdaCDM":
            p1 = posteriors['h']
            p2 = posteriors['om']
            p3 = posteriors['ol']
        elif options.model == "DE":
            p1 = posteriors['w0']
            p2 = posteriors['w1']

    # Save averaged samples.
    if options.model == "LambdaCDM":
        average_samples = np.column_stack((p1,p2))
        np.savetxt(os.path.join(out_folder,final_posterior_name), average_samples, header='h\tom')  
    elif options.model == "CLambdaCDM":
        average_samples = np.column_stack((p1,p2,p3))
        np.savetxt(os.path.join(out_folder,final_posterior_name), average_samples, header='h\tom\tol')  
    elif options.model == "DE":
        average_samples = np.column_stack((p1,p2))
        np.savetxt(os.path.join(out_folder,final_posterior_name), average_samples, header='w0\tw1')

    # Compute and save .05, .16, .5, .84, .95 quantiles.
    if options.model == "LambdaCDM":
        p1_name, p2_name = 'h','om'
    elif options.model == "CLambdaCDM":
        p1_name, p2_name, p3_name = 'h','om', 'ol'
    elif options.model == "DE":
        p1_name, p2_name = 'w0','w1'

    file_path = os.path.join(out_folder,'quantiles_{}_{}{}.txt'.format(options.model, catalog_name, reduced_string))
    print("Will save .5, .16, .50, .84, .95 quantiles in {}".format(file_path))
    sys.stdout = open(file_path, "w+")

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

    if options.model == "CLambdaCDM":

        p3_ll,p3_l,p3_median,p3_h,p3_hh = np.percentile(p3,[5.0,16.0,50.0,84.0,95.0],axis = 0)

        p3_inf_68, p3_sup_68 = p3_median - p3_l,  p3_h - p3_median
        p3_inf_90, p3_sup_90 = p3_median - p3_ll, p3_hh - p3_median

        p3_credible68 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(p3_inf_68, p3_median, p3_sup_68)
        p3_credible90 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(p3_inf_90, p3_median, p3_sup_90)

        print("\n{} 68% CI:".format(p3_name))
        print(p3_credible68)
        print("{} 90% CI:".format(p3_name))
        print(p3_credible90)

    if corner_90:
        fig = plt.figure()
        if options.model == "LambdaCDM":
            fig = corner.corner(average_samples,
                                labels= [r'$h$',
                                         r'$\Omega_m$'],
                                quantiles=[0.05, 0.5, 0.95],
                                show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                                use_math_text=True, truths=[truths['h'],truths['om']])
        elif options.model == "CLambdaCDM":
            fig = corner.corner(average_samples,
                                labels= [r'$h$',
                                         r'$\Omega_m$',
                                         r'$\Omega_\Lambda$'],
                                quantiles=[0.05, 0.5, 0.95],
                                show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                                use_math_text=True, truths=[truths['h'],truths['om'],truths['ol']])
        elif options.model == "DE":
            fig = corner.corner(average_samples,
                                labels=[r'$w_0$',
                                        r'$w_a$'],
                                quantiles=[0.05, 0.5, 0.95],
                                show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                                use_math_text=True, truths=[truths['w0'],truths['w1']])
        plt.savefig(os.path.join(out_folder,"average_posterior_corner_90CI_{}_{}{}.pdf".format(options.model, catalog_name, reduced_string)),bbox_inches='tight', pad_inches=0.16)
        plt.savefig(os.path.join(out_folder,"average_posterior_corner_90CI_{}_{}{}.png".format(options.model, catalog_name, reduced_string)),bbox_inches='tight', pad_inches=0.16)
        plt.close()

    if corner_68:
        fig = plt.figure()
        if options.model == "LambdaCDM":
            fig = corner.corner(average_samples,
                                labels= [r'$h$',
                                         r'$\Omega_m$'],
                                quantiles=[0.16, 0.5, 0.84],
                                show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                                use_math_text=True, truths=[truths['h'],truths['om']])
        elif options.model == "CLambdaCDM":
            fig = corner.corner(average_samples,
                                labels= [r'$h$',
                                         r'$\Omega_m$',
                                         r'$\Omega_\Lambda$'],
                                quantiles=[0.16, 0.5, 0.84],
                                show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                                use_math_text=True, truths=[truths['h'],truths['om'],truths['ol']])
        elif options.model == "DE":
            fig = corner.corner(average_samples,
                                labels=[r'$w_0$',
                                        r'$w_a$'],
                                quantiles=[0.16, 0.5, 0.84],
                                show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                                use_math_text=True, truths=[truths['w0'],truths['w1']])
        plt.savefig(os.path.join(out_folder,"average_posterior_corner_68CI_{}_{}{}.pdf".format(options.model, catalog_name, reduced_string)),bbox_inches='tight')
        plt.savefig(os.path.join(out_folder,"average_posterior_corner_68CI_{}_{}{}.png".format(options.model, catalog_name, reduced_string)),bbox_inches='tight')
        plt.close()

    # Currently unused, it works for 2D models if the parameter space is well-constrained.
    if dpgmm_average:

        COSMOLISA_PATH = os.getcwd()
        sys.path.insert(1, os.path.join(COSMOLISA_PATH,'DPGMM'))
        from dpgmm import *
        sys.path.insert(1, COSMOLISA_PATH)

        import multiprocessing as mp
        import dill as pickle
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

        Nbins = options.N
        pool = mp.Pool(mp.cpu_count())

        if options.model == "LambdaCDM":
            x_flat = np.linspace(0.5,1.0,Nbins)
            y_flat = np.linspace(0.04,0.5,Nbins)
        elif options.model == "DE":
            x_flat = np.linspace(-3.0,-0.3,Nbins)
            y_flat = np.linspace(-1.0,1.0,Nbins)
        else:
            sys.exit("DPGMM only accepts 2D models (LambdaCDM, DE), Exiting.")
        dx = np.diff(x_flat)[0]
        dy = np.diff(y_flat)[0]
        X,Y = np.meshgrid(x_flat,y_flat)

        model = initialise_dpgmm(2,np.column_stack((p1,p2)))
        logdensity = compute_dpgmm(model,max_sticks=8)
        single_posterior = evaluate_grid(logdensity,x_flat,y_flat)
        pickle.dump(single_posterior,open(os.path.join(out_folder,"average_posterior_dpgmm_{0}.p".format(model)),"wb"))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        levs = np.sort(FindHeightForLevel(single_posterior.T,[0.68,0.90]))
        ax.contourf(X,Y,single_posterior.T,100, cmap = matplotlib.cm.gray_r, alpha = 0.5, zorder=1)
        C = ax.contour(X,Y,single_posterior.T,levs,linewidths=0.75,colors='white', zorder = 22, linestyles = 'dashed')
        C = ax.contour(X,Y,single_posterior.T,levs,linewidths=1.0,colors='black')
        ax.grid(alpha=0.5,linestyle='dotted')
        if options.model == "LambdaCDM":
            ax.axvline(truths['h'],color='k',linestyle='dashed',lw=0.5)
            ax.axhline(truths['om'],color='k',linestyle='dashed',lw=0.5)
            ax.set_xlabel(r"$H_0/100\,km\,s^{-1}\,Mpc^{-1}$",fontsize=18)
            ax.set_ylabel(r"$\Omega_m$",fontsize=18)
        elif options.model == "DE":
            ax.axvline(truths['w0'],color='k',linestyle='dashed',lw=0.5)
            ax.axhline(truths['w1'],color='k',linestyle='dashed',lw=0.5)
            ax.set_xlabel(r"$w_0$",fontsize=18)
            ax.set_ylabel(r"$w_a$",fontsize=18)
        plt.savefig(os.path.join(out_folder,"average_posterior_{0}_{1}{2}.pdf".format(options.model, catalog_name, reduced_string)),bbox_inches='tight')
        plt.close()

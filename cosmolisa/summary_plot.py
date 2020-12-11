import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

def init_plotting():
   plt.rcParams['figure.figsize'] = (2*3.4, 3.4)
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


param_map = {'h':r'$h$', 'om':r'$\Omega_m$', 'w0':r'$w_0$', 'w1':r'$w_a$'}
param_truths = {'h': 0.73, 'om': 0.25, 'w0': -1.0, 'w1': 0.0}

if __name__ == "__main__":
    
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-p', '--parameter', default=None, type='string', metavar='parameter', help='parameter to plot (h, om, w0, wa)')
    parser.add_option('--scenario',        default='LambdaCDM', type='string', metavar='scenario', help='Scenario (LambdaCDM,DE')
    parser.add_option('-N',                default=None, type='string', metavar='EMRIs', help='file containing the number of EMRIs per catalog')
    parser.add_option('--path',            default=None, type='string', metavar='path', help='base_path of the posteriors')
    parser.add_option('--labels',          default=None, type='string', metavar='lables', help='file containing the labels for the catalog')
    parser.add_option('--regions',         default=None, type='string', metavar='regions', help='file the credible regions for the parameter to plot')
    parser.add_option('-o', '--output',    default=None, type='string', metavar='DIR', help='directory for output')
    (opts,args)=parser.parse_args()
    
# OUTDATED: To produce the CR file, use compute_CR.py

# For each run we have an averaged_posterior.dat file with two columns ['h','om'] located in base_path
    base_path = opts.path 
    different_paths = ['SNR_100_reduced', 'SNR_100']
    models = ['M105', 'M101', 'M106']
    scenario = opts.scenario #LambdaCDM, DE
    posteriors_list = []

    for model in models:
        for diff_path in different_paths:
            print('Reading {}_{}'.format(model, diff_path))
            post_path = os.path.join(base_path,scenario+'_'+diff_path,model+'_averaged')
            posteriors_list.append(np.genfromtxt(post_path+'/averaged_posterior.dat',names=True))

    #FIXME: calculate and produce the plots
    medians= []
    inf_68 = []
    sup_68 = []
    inf_90 = []
    sup_90 = []
    credible68 = []
    credible90 = []
    init_plotting()
    for post in posteriors_list:
        print("Reading...\n")
        if opts.parameter == 'h':
            print(post['h'])
            ll,l,median,h,hh = np.percentile(post['h'],[5.0,16.0,50.0,84.0,95.0],axis = 0)
            medians.append(median)
            inf_68.append(median - l)
            sup_68.append(h - median)
            inf_90.append(median - ll)
            sup_90.append(hh - median)
    credible68 = np.column_stack((inf_68, sup_68))
    credible90 = np.column_stack((inf_90, sup_90))
#    labels      = np.genfromtxt(opts.labels,dtype='str')
    labels = [r'M5$_{4\;\,}$', r'M5$_{10}$', r'M1$_{4\;\,}$', r'M1$_{10}$', r'M6$_{4\;\,}$', r'M6$_{10}$']
    N           = opts.N.split(',')
    xaxis       = range(len(labels))

    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    ax.errorbar(xaxis, medians, yerr=credible68.T, fmt='o', color='red', ecolor='red', elinewidth=1.3, capsize=2, ms = 2.7, zorder=1)
    ax.errorbar(xaxis, medians, yerr=credible90.T, fmt='o', color='black', ecolor='black', elinewidth=1, capsize=2, ms = 2, zorder=0)
    ax.axhline(param_truths[opts.parameter], linestyle='dashed', linewidth=1.2, zorder = 0)
    for i,n in enumerate(N):
        plt.text( xaxis[i], medians[i]+credible90[i,1], r'N = {0}'.format(int(n)), rotation=45, fontsize=11)
    plt.yticks(fontsize=13)
    plt.xticks(xaxis, labels, rotation=45, fontsize=15)
    ax.set_ylabel(param_map[opts.parameter], fontsize=15)
    plt.savefig(os.path.join(opts.output,opts.parameter+'_summary.pdf'),bbox_inches='tight')

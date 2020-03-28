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


param_map = {'h':'h', 'om':r'$\Omega_m$', 'w0':r'$w_0$', 'w1':r'$w_a$'}

if __name__ == "__main__":
    
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-p', '--parameter',   default=None, type='string', metavar='parameter', help='parameter to plot (h, om, w0, wa)')
    parser.add_option('--galaxies',          default=None, type='string', metavar='galaxies', help='file containing the number of galaxies per catalog')
    parser.add_option('--labels',            default=None, type='string', metavar='lables', help='file containing the labels for the catalog')
    parser.add_option('--regions',           default=None, type='string', metavar='regions', help='file the credible regions for the parameter to plot')
    parser.add_option('-o', '--output',      default=None, type='string', metavar='DIR', help='directory for output')
    (opts,args)=parser.parse_args()
    
    init_plotting()
    ll,l,medians,h,hh = np.loadtxt(opts.regions, unpack = True)
    credible68  = np.array([(medians[i]-l[i], h[i]-medians[i]) for i in range(len(medians))])
    credible90  = np.array([(medians[i]-ll[i], hh[i]-medians[i]) for i in range(len(medians))])
    labels      = np.genfromtxt(opts.labels,dtype='str')
    N           = np.loadtxt(opts.galaxies)
    xaxis       = range(len(N))
    
    print(credible90)
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    ax.errorbar(xaxis, medians, yerr=credible68.T, fmt='o', color='red', ecolor='red', elinewidth=1.5, capsize=2, ms = 4,zorder=1)
    ax.errorbar(xaxis, medians, yerr=credible90.T, fmt='o', color='black', ecolor='black', elinewidth=1, capsize=2, ms = 4, zorder=0)
    for i,n in enumerate(N):
        plt.text( xaxis[i], medians[i]+credible90[i,1], r'N = {0}'.format(int(n)), rotation=45, fontsize=8)
    plt.xticks(xaxis, labels, rotation=45)
    ax.set_ylabel(param_map[opts.parameter])
    plt.savefig(os.path.join(opts.output,opts.parameter+'_summary.pdf'),bbox_inches='tight')

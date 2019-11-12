import numpy as np
import os
import matplotlib.pyplot as plt
from optparse import OptionParser

if __name__=="__main__":
    parser=OptionParser()
    parser.add_option('-o','--out',action='store',type='string',default=None,help='Output folder', dest='output')
    parser.add_option('-d',action='store',type='string',default=None,help='data folder', dest='data')
    parser.add_option('-c',action='store',type='string',default=None,help='source class (MBH, EMRI, sBH)', dest='source_class')
    parser.add_option('-f',action='store',type='string',default=None,help='Formation model', dest='formation_model')
    parser.add_option('-m',action='store',type='string',default='LambdaCDM',help='model (LambdaCDM, LambdaCDMDE)', dest='model')
    (options,args)=parser.parse_args()

    if options.formation_model is not None:
        work_folder = os.path.join(options.data,options.formation_model)
    else:
        work_folder = options.data

    if options.model == "LambdaCDM":
        p = ['h','om']
        formats = ['f8','f8']
    elif options.model == "LambdaCDMDE":
        p = ['h','om','ol','w0','w1']
        formats = ['f8','f8','f8','f8','f8']
    else:
        print("model not known, exiting.\n")

    confidence_levels = {pi:[] for pi in p}

    all_files = os.listdir(work_folder)
    catalogs = [a for a in all_files if 'cat' in a]
    for c in catalogs:
        posteriors = np.genfromtxt(os.path.join(work_folder,c+"/posterior.dat"),names=True)
        for pi in p:
            confidence_levels[pi].append(np.percentile(posteriors[pi],[5,50,95]))
            print(c,pi, confidence_levels[pi][-1])
    import matplotlib.pyplot as plt

    def average_confidence(confidence_levels, p):
        y = np.array(confidence_levels[p])
        m = np.average(y[:,1])
        lm = m-np.average(y[:,0])
        hm = np.average(y[:,2])-m
        dlm = np.std(lm-y[:,0])
        dhm = np.std(y[:,2]-hm)
        print("%s = %.2f_{-%.2f\pm %.2f}^{+%.2f\pm %.2f}\n"%(p,m,lm,dlm,hm,dhm))

    for pi in p: average_confidence(confidence_levels,pi)
    exit()
    x = range(len(catalogs))
    f = plt.figure(1)
    ax = f.add_subplot(211)
    ym = np.array(confidence_levels['h'])
    ax.errorbar(x, ym[:,1],yerr=ym[:,2]-ym[:,0])
    ax.axhline(0.73)
    ax.set_ylabel('h')
    ax = f.add_subplot(212)
    ym = np.array(confidence_levels['om'])
    ax.errorbar(x, ym[:,1],yerr=ym[:,2]-ym[:,0])
    ax.axhline(0.25)
    ax.set_ylabel('om')
    plt.tight_layout()
    plt.show()

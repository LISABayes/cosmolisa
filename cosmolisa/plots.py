#!/usr/bin/env python
import matplotlib.pyplot as plt
import corner


def par_histogram(x, par_name, truth=None, outdir):
    fig = plt.figure()
    plt.hist(x, density=True, alpha = 1.0, histtype='step', edgecolor="black")
    plt.axvline(truth, linestyle='dashed', color='r')
    quantiles = np.quantile(x, [0.05, 0.5, 0.95])
    plt.title(r'${par} = {med:.3f}({low:.3f},+{up:.3f})$'.format(par=par_name, med=quantiles[1], low=quantiles[0]-quantiles[1], up=quantiles[2]-quantiles[1]), size = 16)
    plt.xlabel(r'${par}$'.format(par=par_name))
    plt.savefig(os.path.join(outdir,'Plots','{}_histogram.pdf'.format(par_name)), bbox_inches='tight')
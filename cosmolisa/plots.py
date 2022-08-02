import numpy as np
import corner
import os
import matplotlib.pyplot as plt

# Mathematical labels used by the different models
labels_plot = {'LambdaCDM_h':  ['h'],
               'LambdaCDM_om': ['\Omega_m'],
               'LambdaCDM':    [r'$h$', r'$\Omega_m$'],
               'CLambdaCDM':   [r'$h$', r'$\Omega_m$', r'$\Omega_\Lambda$'],
               'LambdaCDMDE':  [r'$h$', r'$\Omega_m$', r'$\Omega_\Lambda$', r'$w_0$', r'$w_a$'],
               'DE':           [r'$w_0$', r'$w_a$'],
               'Rate':         [r'$h$', r'$\Omega_m$', r'$\log_{10} r_0$', r'$W$', r'$R$', r'$Q$'],
               'Luminosity':   [r'$\phi^{*}/Mpc^{3}$', r'$a$', r'$M^{*}$', r'$b$', r'$\alpha$', r'$c$'],
              }

def par_hist(model, samples, outdir, name, bins=20, truths=None):
    """
    Histogram for single-parameter inference
    """
    fmt = "{{0:{0}}}".format('.3f').format
    fig = plt.figure()
    plt.hist(samples, density=True, bins=bins, alpha = 1.0, histtype='step', edgecolor="black", lw=1.2)
    quantiles = np.quantile(samples, [0.05, 0.5, 0.95])
    plt.axvline(quantiles[0], linestyle='dashed', color='k', lw=1.5)
    plt.axvline(quantiles[1], linestyle='dashed', color='k', lw=1.5)
    plt.axvline(quantiles[2], linestyle='dashed', color='k', lw=1.5)
    plt.axvline(truths, linestyle='dashed', color='#4682b4', lw=1.5)
    med, low, up = quantiles[1], quantiles[0]-quantiles[1], quantiles[2]-quantiles[1]
    plt.title(r"${par} = {{{med}}}_{{{low}}}^{{+{up}}}$".format(par=labels_plot[model][0], med=fmt(med), low=fmt(low), up=fmt(up)), size = 16)
    plt.xlabel(r'${}$'.format(labels_plot[model][0]), fontsize=16)
    fig.savefig(os.path.join(outdir,'Plots', name+'.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(outdir,'Plots', name+'.png'), bbox_inches='tight')

def histogram(x, **kwargs):

    if (kwargs['model'] == "LambdaCDM_h"):
        par_hist(model=kwargs['model'],
                 samples=x['h'],
                 truths=kwargs['truths']['h'],
                 outdir=kwargs['outdir'],
                 name='histogram_h_90CI')

    elif (kwargs['model'] == "LambdaCDM_om"):
        par_hist(model=kwargs['model'],
                 samples=x['om'],
                 truths=kwargs['truths']['om'],
                 outdir=kwargs['outdir'],
                 name='histogram_om_90CI')        


def corner_config(model, samps_tuple, quantiles_plot, outdir, name, truths=None):
    """
    Instructions used to make corner plots.
    'title_quantiles' is not specified, hence plotted quantiles coincide with 'quantiles'.
    This holds for the version of corner.py indicated in the README file. 
    """
    samps = np.column_stack(samps_tuple)
    fig = corner.corner(samps,
                        labels = labels_plot[model],
                        quantiles = quantiles_plot,
                        show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
                        use_math_text=True, 
                        truths=truths)
    fig.savefig(os.path.join(outdir,'Plots', name+'.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(outdir,'Plots', name+'.png'), bbox_inches='tight')


def corner_plot(x, **kwargs):

    if (kwargs['model'] == "LambdaCDM"): 
        corner_config(model=kwargs['model'], 
                    samps_tuple=(x['h'],x['om']),
                    quantiles_plot=[0.16, 0.5, 0.84], 
                    truths=[kwargs['truths']['h'],kwargs['truths']['om']],
                    outdir=kwargs['outdir'],
                    name='corner_plot_68CI')
        corner_config(model=kwargs['model'], 
                    samps_tuple=(x['h'],x['om']),
                    quantiles_plot=[0.05, 0.5, 0.95], 
                    truths=[kwargs['truths']['h'],kwargs['truths']['om']],
                    outdir=kwargs['outdir'],
                    name='corner_plot_90CI')

    elif (kwargs['model'] == 'CLambdaCDM'):
        corner_config(model=kwargs['model'], 
                    samps_tuple=(x['h'],x['om'],x['ol']),
                    quantiles_plot=[0.05, 0.5, 0.95], 
                    truths=[truths['h'],truths['om'],truths['ol']],
                    outdir=outdir,
                    name='corner_plot_90CI')

    elif (kwargs['model'] == 'LambdaCDMDE'):    
        corner_config(model=kwargs['model'], 
                    samps_tuple=(x['h'],x['om'],x['ol'],x['w0'],x['w1']),
                    quantiles_plot=[0.05, 0.5, 0.95], 
                    truths=[truths['h'],truths['om'],truths['ol'],truths['w0'],truths['w1']],
                    outdir=outdir,
                    name='corner_plot_90CI')

    elif (kwargs['model'] == 'DE'):
        corner_config(model=kwargs['model'], 
                    samps_tuple=(x['w0'],x['w1']),
                    quantiles_plot=[0.05, 0.5, 0.95], 
                    truths=[truths['w0'],truths['w1']],
                    outdir=outdir,
                    name='corner_plot_90CI')

    elif (kwargs['model'] == 'Rate'):
        corner_config(model=kwargs['model'], 
                    samps_tuple=(x['h'],x['om'],x['log10r0'],x['W'],x['R'],x['Q']),
                    quantiles_plot=[0.05, 0.5, 0.95], 
                    outdir=outdir,
                    name='corner_plot_rate_90CI')       

    elif (kwargs['model'] == 'Luminosity'):
        corner_config(model=kwargs['model'], 
                    samps_tuple=(x['phistar0'],x['phistar_exponent'],x['Mstar0'],x['Mstar_exponent'],x['alpha0'],x['alpha_exponent']),
                    quantiles_plot=[0.05, 0.5, 0.95], 
                    outdir=outdir,
                    name='corner_plot_luminosity_90CI')         



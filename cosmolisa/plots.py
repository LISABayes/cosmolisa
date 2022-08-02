import numpy as np
import corner
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

from cosmolisa import cosmology as cs
from cosmolisa import likelihood as lk


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


def redshift_ev_plot(x, **kwargs):

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    z   = np.linspace(kwargs['event'].zmin, kwargs['event'].zmax, 100)

    #FIXME: Fix positions of colorbar and axes 
    if (kwargs['em_sel']):
        ax3 = ax.twinx()
        
        if ("DE" in kwargs['model']): normalisation = matplotlib.colors.Normalize(vmin=np.min(x['w0']), vmax=np.max(x['w0']))
        else:                         normalisation = matplotlib.colors.Normalize(vmin=np.min(x['h']), vmax=np.max(x['h']))
        # choose a colormap
        c_m = matplotlib.cm.cool
        # create a ScalarMappable and initialize a data structure
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=normalisation)
        s_m.set_array([])
        for i in range(x.shape[0])[::10]:
            if ("LambdaCDM_h" in kwargs['model']): O = cs.CosmologicalParameters(x['h'][i],kwargs['truths']['om'],kwargs['truths']['ol'],kwargs['truths']['w0'],kwargs['truths']['w1'])
            elif ("LambdaCDM_om" in kwargs['model']): O = cs.CosmologicalParameters(kwargs['truths']['h'], x['om'][i], kwargs['truths']['ol'], kwargs['truths']['w0'], kwargs['truths']['w1'])
            elif ("LambdaCDM" in kwargs['model']): O = cs.CosmologicalParameters(x['h'][i],x['om'][i],1.0-x['om'][i],kwargs['truths']['w0'],kwargs['truths']['w1'])
            elif ("CLambdaCDM" in kwargs['model']): O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],kwargs['truths']['w0'],kwargs['truths']['w1'])
            elif ("LambdaCDMDE" in kwargs['model']): O = cs.CosmologicalParameters(x['h'][i],x['om'][i],x['ol'][i],x['w0'][i],x['w1'][i])
            elif ("DE" in kwargs['model']): O = cs.CosmologicalParameters(kwargs['truths']['h'],kwargs['truths']['om'],kwargs['truths']['ol'],x['w0'][i],x['w1'][i])
            distances = np.array([O.LuminosityDistance(zi) for zi in z])
            if ("DE" in kwargs['model']):  
                ax3.plot(z, [lk.em_selection_function(d) for d in distances], lw = 0.15, color=s_m.to_rgba(x['w0'][i]), alpha = 0.5)
            else: 
                ax3.plot(z, [lk.em_selection_function(d) for d in distances], lw = 0.15, color=s_m.to_rgba(x['h'][i]), alpha = 0.5)
            O.DestroyCosmologicalParameters()
        CB = plt.colorbar(s_m, orientation='vertical', pad=0.15)
        if ("DE" in kwargs['model']): CB.set_label('w_0')
        else: CB.set_label('h')
        ax3.set_ylim(0.0, 1.0)
        ax3.set_ylabel('selection function')

    # Plot the likelihood  
    distance_likelihood = []
    print("Making redshift plot of event", kwargs['event'].ID)
    for i in range(x.shape[0])[::10]:
        if   ("LambdaCDM_h" in kwargs['model']):  O = cs.CosmologicalParameters(x['h'][i], kwargs['truths']['om'], kwargs['truths']['ol'], kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ("LambdaCDM_om" in kwargs['model']): O = cs.CosmologicalParameters(kwargs['truths']['h'], x['om'][i], kwargs['truths']['ol'], kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ("LambdaCDM" in kwargs['model']):    O = cs.CosmologicalParameters(x['h'][i], x['om'][i], 1.0-x['om'][i], kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ("CLambdaCDM" in kwargs['model']):   O = cs.CosmologicalParameters(x['h'][i], x['om'][i], x['ol'][i], kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ("LambdaCDMDE" in kwargs['model']):  O = cs.CosmologicalParameters(x['h'][i], x['om'][i], x['ol'][i], x['w0'][i], x['w1'][i])
        elif ("DE" in kwargs['model']):           O = cs.CosmologicalParameters(kwargs['truths']['h'], kwargs['truths']['om'], kwargs['truths']['ol'], x['w0'][i], x['w1'][i])
        # distance_likelihood.append(np.array([lk.logLikelihood_single_event(C.hosts[kwargs['event'].ID], kwargs['event'].dl, kwargs['event'].sigma, O, zi) for zi in z]))
        distance_likelihood.append(np.array([-0.5*((O.LuminosityDistance(zi) - kwargs['event'].dl)/kwargs['event'].sigma)**2 for zi in z]))
        O.DestroyCosmologicalParameters()
    distance_likelihood = np.exp(np.array(distance_likelihood))
    l, m, h = np.percentile(distance_likelihood,[5,50,95], axis = 0)

    ax2 = ax.twinx()
    ax2.plot(z, m, linestyle='dashed', color='k', lw=0.75)
    ax2.fill_between(z, l, h,facecolor='magenta', alpha=0.5)
    ax2.plot(z, np.exp(np.array([-0.5*((kwargs['omega_true'].LuminosityDistance(zi)-kwargs['event'].dl)/kwargs['event'].sigma)**2 for zi in z])), linestyle = 'dashed', color='gold', lw=1.5)
    ax.axvline(lk.find_redshift(kwargs['omega_true'],kwargs['event'].dl), linestyle='dotted', lw=0.8, color='red')
    ax.axvline(kwargs['event'].z_true, linestyle='dotted', lw=0.8, color='k')
    ax.hist(x['z%d'%kwargs['event'].ID], bins=z, density=True, alpha = 0.5, facecolor="green")
    ax.hist(x['z%d'%kwargs['event'].ID], bins=z, density=True, alpha = 0.5, histtype='step', edgecolor="k")

    for g in kwargs['event'].potential_galaxy_hosts:
        zg = np.linspace(g.redshift - 5*g.dredshift, g.redshift+5*g.dredshift, 100)
        pg = norm.pdf(zg, g.redshift, g.dredshift*(1+g.redshift))*g.weight
        ax.plot(zg, pg, lw=0.5, color='k')
    ax.set_xlabel('$z_{%d}$'%kwargs['event'].ID, fontsize=16)
    ax.set_ylabel('probability density', fontsize=16)
    plt.savefig(os.path.join(kwargs['outdir'], 'Plots', 'redshift_{}'.format(kwargs['event'].ID)+'.png'), bbox_inches='tight')
    plt.close()


def MBHB_regression(x, **kwargs):

    dl = [e.dl/1e3 for e in kwargs['data']]
    ztrue = [e.potential_galaxy_hosts[0].redshift for e in kwargs['data']]
    if not len(kwargs['data']) == 1:
        dztrue = np.squeeze([[ztrue[i]-e.zmin, e.zmax-ztrue[i]] for i,e in enumerate(kwargs['data'])]).T
    else:
        dztrue = np.squeeze([[ztrue[i]-e.zmin, e.zmax-ztrue[i]] for i,e in enumerate(kwargs['data'])]).reshape(2,1)
    deltadl = [np.sqrt((e.sigma/1e3)**2 + (lk.sigma_weak_lensing(e.potential_galaxy_hosts[0].redshift, e.dl)/1e3)**2) for e in kwargs['data']]
    z = [np.median(x['z{}'.format(e.ID)]) for e in kwargs['data']]
    deltaz = [2*np.std(x['z{}'.format(e.ID)]) for e in kwargs['data']]
    redshift = np.logspace(-3, 1.0, 100)

    # loop over the posterior samples to get all models to then average for the plot
    models = []
    for k in range(x.shape[0]):
        if   ("LambdaCDM_h" in kwargs['model']): omega = cs.CosmologicalParameters(x['h'][k], kwargs['truths']['om'], kwargs['truths']['ol'], kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ("LambdaCDM_om" in kwargs['data']): omega = cs.CosmologicalParameters(kwargs['truths']['h'], x['om'][k], 1.0-x['om'][k], kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ("LambdaCDM" in kwargs['data']):    omega = cs.CosmologicalParameters(x['h'][k], x['om'][k], 1.0-x['om'][k], kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ("CLambdaCDM" in kwargs['data']):   omega = cs.CosmologicalParameters(x['h'][k], x['om'][k], x['ol'][k], kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ("LambdaCDMDE" in kwargs['data']):  omega = cs.CosmologicalParameters(x['h'][k], x['om'][k], x['ol'][k], x['w0'][k], x['w1'][k])
        elif ("DE" in kwargs['data']):           omega = cs.CosmologicalParameters(kwargs['truths']['h'], kwargs['truths']['om'], kwargs['truths']['ol'], x['w0'][k], x['w1'][k])
        models.append([omega.LuminosityDistance(zi)/1e3 for zi in redshift])
        omega.DestroyCosmologicalParameters()

    models = np.array(models)
    model2p5,model16,model50,model84,model97p5 = np.percentile(models,[2.5,16.0,50.0,84.0,97.5],axis = 0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(z, dl, xerr=deltaz, yerr=deltadl, markersize=1, linewidth=2, color='k', fmt='o')
    ax.plot(redshift, [kwargs['omega_true'].LuminosityDistance(zi)/1e3 for zi in redshift], linestyle='dashed', color='red', zorder=22)
    ax.plot(redshift, model50, color='k')
    ax.errorbar(ztrue, dl, xerr=dztrue, yerr=deltadl, markersize=2, linewidth=1, color='r', fmt='o')
    ax.fill_between(redshift, model2p5, model97p5, facecolor='turquoise')
    ax.fill_between(redshift, model16, model84, facecolor='cyan')
    ax.set_xlabel(r"z", fontsize=16)
    ax.set_ylabel(r"$D_L$/Gpc", fontsize=16)
    fig.savefig(os.path.join(kwargs['outdir'], 'Plots', 'MBHB_regression_68_95CI.pdf'), bbox_inches='tight')
    plt.close()
import numpy as np
import corner
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

from cosmolisa import cosmology as cs
from cosmolisa import likelihood as lk
from cosmolisa import galaxy as gal
from cosmolisa import astrophysics as astro

truth_color = "#4682b4"

# Mathematical labels used by the different models.
labels_plot = {
    'LambdaCDM_h': ['h'],
    'LambdaCDM_om': ['\Omega_m'],
    'LambdaCDM': [r'$h$', r'$\Omega_m$'],
    'CLambdaCDM': [r'$h$', r'$\Omega_m$', r'$\Omega_\Lambda$'],
    'LambdaCDMDE': [r'$h$', r'$\Omega_m$', r'$\Omega_\Lambda$', 
                    r'$w_0$', r'$w_a$'],
    'DE': [r'$w_0$', r'$w_a$'],
    'RatePW': [r'$h$', r'$\Omega_m$', r'$\log_{10} r_0$', r'$p_1$'],
    'Rate': [r'$h$', r'$\Omega_m$', r'$\log_{10} r_0$', r'$\log_{10} p_1$',
             r'$p_2$', r'$p_3$'],
    'Luminosity': [r'$\phi^{*}/Mpc^{3}$', r'$a$', r'$M^{*}$', r'$b$',
                   r'$\alpha$', r'$c$'],
    }


def par_hist(model, samples, outdir, name, bins=20, truths=None):
    """Histogram for single-parameter inference."""
    fmt = "{{0:{0}}}".format('.3f').format
    fig = plt.figure()
    plt.hist(samples, density=True, bins=bins, alpha = 1.0, 
             histtype='step', edgecolor="black", lw=1.2)
    quantiles = np.quantile(samples, [0.05, 0.5, 0.95])
    plt.axvline(quantiles[0], linestyle='dashed', color='k', lw=1.5)
    plt.axvline(quantiles[1], linestyle='dashed', color='k', lw=1.5)
    plt.axvline(quantiles[2], linestyle='dashed', color='k', lw=1.5)
    plt.axvline(truths, linestyle='dashed', color=truth_color, lw=1.5)
    med, low, up = (quantiles[1], quantiles[0]-quantiles[1], 
                    quantiles[2]-quantiles[1])
    plt.title(r"${par} = {{{med}}}_{{{low}}}^{{+{up}}}$".format(
        par=labels_plot[model][0], med=fmt(med),
        low=fmt(low), up=fmt(up)), size=16)
    plt.xlabel(r'${}$'.format(labels_plot[model][0]), fontsize=16)
    fig.savefig(os.path.join(outdir,'Plots', name+'.pdf'),
        bbox_inches='tight')
    fig.savefig(os.path.join(outdir,'Plots', name+'.png'),
        bbox_inches='tight')


def histogram(x, **kwargs):
    """Function to call par_hist with the appropriate model."""
    if (kwargs['model'] == 'LambdaCDM_h'):
        par_hist(model=kwargs['model'],
                 samples=x['h'],
                 truths=kwargs['truths']['h'],
                 outdir=kwargs['outdir'],
                 name="histogram_h_90CI")

    elif (kwargs['model'] == 'LambdaCDM_om'):
        par_hist(model=kwargs['model'],
                 samples=x['om'],
                 truths=kwargs['truths']['om'],
                 outdir=kwargs['outdir'],
                 name="histogram_om_90CI")


def corner_config(model, samps_tuple, quantiles_plot, 
                  outdir, name, truths=None, **kwargs):
    """Instructions used to make corner plots.
    'title_quantiles' is not specified, hence plotted quantiles
    coincide with 'quantiles'. This holds for the version of corner.py
    indicated in the README file. 
    """
    samps = np.column_stack(samps_tuple)
    fig = corner.corner(samps,
                        labels=labels_plot[model],
                        quantiles=quantiles_plot,
                        show_titles=True, 
                        title_fmt='.3f',
                        title_kwargs={'fontsize': 16},
                        label_kwargs={'fontsize': 16},
                        use_math_text=True,
                        truths=truths)
    fig.savefig(os.path.join(outdir,"Plots", name+".pdf"),
                bbox_inches='tight')
    fig.savefig(os.path.join(outdir,"Plots", name+".png"),
                bbox_inches='tight')


def corner_plot(x, **kwargs):
    """Function to call corner_config according to different models."""
    print("Making corner plot...")
    if (kwargs['model'] == 'LambdaCDM'):
        corner_config(model=kwargs['model'],
                      samps_tuple=(x['h'], x['om']),
                      quantiles_plot=[0.16, 0.5, 0.84],
                      truths=[kwargs['truths']['h'], kwargs['truths']['om']],
                      outdir=kwargs['outdir'],
                      name="corner_plot_68CI")
        corner_config(model=kwargs['model'],
                      samps_tuple=(x['h'] ,x['om']),
                      quantiles_plot=[0.05, 0.5, 0.95],
                      truths=[kwargs['truths']['h'], kwargs['truths']['om']],
                      outdir=kwargs['outdir'],
                      name="corner_plot_90CI")

    elif (kwargs['model'] == 'CLambdaCDM'):
        corner_config(model=kwargs['model'], 
                      samps_tuple=(x['h'], x['om'], x['ol']),
                      quantiles_plot=[0.05, 0.5, 0.95],
                      truths=[kwargs['truths']['h'], kwargs['truths']['om'],
                              kwargs['truths']['ol']],
                      outdir=kwargs['outdir'],
                      name="corner_plot_90CI")

    elif (kwargs['model'] == 'LambdaCDMDE'):    
        corner_config(model=kwargs['model'], 
                      samps_tuple=(x['h'], x['om'], x['ol'], x['w0'],
                                   x['w1']),
                      quantiles_plot=[0.05, 0.5, 0.95], 
                      truths=[kwargs['truths']['h'], kwargs['truths']['om'],
                              kwargs['truths']['ol'], kwargs['truths']['w0'],
                              kwargs['truths']['w1']],
                      outdir=kwargs['outdir'],
                      name="corner_plot_90CI")

    elif (kwargs['model'] == 'DE'):
        corner_config(model=kwargs['model'], 
                      samps_tuple=(x['w0'],x['w1']),
                      quantiles_plot=[0.05, 0.5, 0.95], 
                      truths=[kwargs['truths']['w0'],
                              kwargs['truths']['w1']],
                      outdir=kwargs['outdir'],
                      name="corner_plot_90CI")

    elif (kwargs['model'] == 'RatePW'):
        corner_config(model=kwargs['model'], 
                      samps_tuple=(x['h'], x['om'],
                                   x['log10r0'], x['p1']),
                      quantiles_plot=[0.05, 0.5, 0.95],
                      SFRD='powerlaw',
                      outdir=kwargs['outdir'],
                      name="corner_plot_rate_90CI")
    elif (kwargs['model'] == 'Rate'):
        corner_config(model=kwargs['model'], 
                      samps_tuple=(x['h'], x['om'], x['log10r0'],
                                   x['log10p1'], x['p2'], x['p3']),
                      quantiles_plot=[0.05, 0.5, 0.95],
                      outdir=kwargs['outdir'],
                      name="corner_plot_rate_90CI")

    elif (kwargs['model'] == 'Luminosity'):
        corner_config(model=kwargs['model'], 
                      samps_tuple=(x['phistar0'], x['phistar_exponent'],
                                   x['Mstar0'], x['Mstar_exponent'],
                                   x['alpha0'], x['alpha_exponent']),
                      quantiles_plot=[0.05, 0.5, 0.95], 
                      outdir=kwargs['outdir'],
                      name="corner_plot_luminosity_90CI")         


def redshift_ev_plot(x, **kwargs):
    """Plot single-event redshift posterior and 
    single-event likelihood."""
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    z   = np.linspace(kwargs['event'].zmin, kwargs['event'].zmax, 100)

    #FIXME: Fix positions of colorbar and axes.
    if (kwargs['em_sel']):
        ax3 = ax.twinx()
        
        if ("DE" in kwargs['model']):
            normalisation = matplotlib.colors.Normalize(vmin=np.min(x['w0']),
                                                        vmax=np.max(x['w0']))
        else:
            normalisation = matplotlib.colors.Normalize(vmin=np.min(x['h']),
                                                        vmax=np.max(x['h']))
        # Choose a colormap.
        c_m = matplotlib.cm.cool
        # Create a ScalarMappable and initialize a data structure.
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=normalisation)
        s_m.set_array([])
        for i in range(x.shape[0])[::10]:
            if ('LambdaCDM_h' in kwargs['model']):
                O = cs.CosmologicalParameters(
                    x['h'][i], kwargs['truths']['om'], kwargs['truths']['ol'],
                    kwargs['truths']['w0'], kwargs['truths']['w1'])
            elif ('LambdaCDM_om' in kwargs['model']):
                O = cs.CosmologicalParameters(
                    kwargs['truths']['h'], x['om'][i], kwargs['truths']['ol'],
                    kwargs['truths']['w0'], kwargs['truths']['w1'])
            elif ('LambdaCDM' in kwargs['model']): 
                O = cs.CosmologicalParameters(
                    x['h'][i], x['om'][i], 1.0-x['om'][i],
                    kwargs['truths']['w0'], kwargs['truths']['w1'])
            elif ('CLambdaCDM' in kwargs['model']):
                O = cs.CosmologicalParameters(
                    x['h'][i], x['om'][i], x['ol'][i],
                    kwargs['truths']['w0'], kwargs['truths']['w1'])
            elif ('LambdaCDMDE' in kwargs['model']):
                O = cs.CosmologicalParameters(
                    x['h'][i], x['om'][i], x['ol'][i],
                    x['w0'][i], x['w1'][i])
            elif ('DE' in kwargs['model']):
                O = cs.CosmologicalParameters(
                    kwargs['truths']['h'], kwargs['truths']['om'],
                    kwargs['truths']['ol'], x['w0'][i], x['w1'][i])
            distances = np.array([O.LuminosityDistance(zi) for zi in z])

            if ('DE' in kwargs['model']):
                ax3.plot(z, [lk.em_selection_function(d) for d in distances],
                         lw=0.15, color=s_m.to_rgba(x['w0'][i]), alpha=0.5)
            else:
                ax3.plot(z, [lk.em_selection_function(d) for d in distances],
                         lw=0.15, color=s_m.to_rgba(x['h'][i]), alpha=0.5)
            O.DestroyCosmologicalParameters()
        CB = plt.colorbar(s_m, orientation='vertical', pad=0.15)
        if ("DE" in kwargs['model']): CB.set_label("w_0")
        else: CB.set_label("h")
        ax3.set_ylim(0.0, 1.0)
        ax3.set_ylabel("selection function")

    # Plot the likelihood.
    distance_likelihood = []
    print("Making redshift plot of event", kwargs['event'].ID)
    for i in range(x.shape[0])[::10]:
        if ('LambdaCDM_h' in kwargs['model']):
            O = cs.CosmologicalParameters(
                x['h'][i], kwargs['truths']['om'], kwargs['truths']['ol'],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('LambdaCDM_om' in kwargs['model']):
            O = cs.CosmologicalParameters(
                kwargs['truths']['h'], x['om'][i], kwargs['truths']['ol'],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('LambdaCDM' in kwargs['model']):
            O = cs.CosmologicalParameters(
                x['h'][i], x['om'][i], 1.0-x['om'][i],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('CLambdaCDM' in kwargs['model']):
            O = cs.CosmologicalParameters(
                x['h'][i], x['om'][i], x['ol'][i],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('LambdaCDMDE' in kwargs['model']):
            O = cs.CosmologicalParameters(
                x['h'][i], x['om'][i], x['ol'][i], x['w0'][i], x['w1'][i])
        elif ('DE' in kwargs['model']):
            O = cs.CosmologicalParameters(
                kwargs['truths']['h'], kwargs['truths']['om'], 
                kwargs['truths']['ol'], x['w0'][i], x['w1'][i])
        # distance_likelihood.append(np.array([lk.logLikelihood_single_event(
        #    C.hosts[kwargs['event'].ID], kwargs['event'].dl, 
        #    kwargs['event'].sigmadl, O, zi) for zi in z]))
        distance_likelihood.append(
            np.array([-0.5*((O.LuminosityDistance(zi) - kwargs['event'].dl)
            / kwargs['event'].sigmadl)**2 for zi in z]))
        O.DestroyCosmologicalParameters()
    distance_likelihood = np.exp(np.array(distance_likelihood))
    l, m, h = np.percentile(distance_likelihood,[5, 50, 95], axis=0)

    ax2 = ax.twinx()
    ax2.plot(z, m, linestyle='dashed', color='k', lw=0.75)
    ax2.fill_between(z, l, h,facecolor='magenta', alpha=0.5)
    ax2.plot(z, np.exp(np.array([-0.5*(
        (kwargs['omega_true'].LuminosityDistance(zi)-kwargs['event'].dl)
        /kwargs['event'].sigmadl)**2 for zi in z])),
        linestyle = 'dashed', color='gold', lw=1.5)
    ax.axvline(lk.find_redshift(kwargs['omega_true'], kwargs['event'].dl),
        linestyle='dotted', lw=0.8, color='red')
    ax.axvline(kwargs['event'].z_true, linestyle='dotted', lw=0.8, color='k')
    ax.hist(x['z%d'%kwargs['event'].ID],
        bins=z, density=True, alpha=0.5, facecolor='green')
    ax.hist(x['z%d'%kwargs['event'].ID],
        bins=z, density=True, alpha=0.5, histtype='step', edgecolor='k')

    for g in kwargs['event'].potential_galaxy_hosts:
        zg = np.linspace(g.redshift - 5*g.dredshift, 
            g.redshift+5*g.dredshift, 100)
        pg = norm.pdf(zg, g.redshift, g.dredshift*(1+g.redshift)) * g.weight
        ax.plot(zg, pg, lw=0.5, color='k')
    ax.set_xlabel('$z_{%d}$'%kwargs['event'].ID, fontsize=16)
    ax.set_ylabel('probability density', fontsize=16)
    plt.savefig(os.path.join(kwargs['outdir'], "Plots",
        "redshift_{}".format(kwargs['event'].ID)+".png"), bbox_inches='tight')
    plt.close()


def MBHB_regression(x, **kwargs):
    """Simple regression plot (dL-z) using MBHB events only."""
    print("Making MBHB regression plot...")
    dl = [e.dl/1e3 for e in kwargs['data']]
    ztrue = [e.potential_galaxy_hosts[0].redshift for e in kwargs['data']]
    if not (len(kwargs['data']) == 1):
        dztrue = np.squeeze([[ztrue[i]-e.zmin, e.zmax-ztrue[i]] 
                            for i, e in enumerate(kwargs['data'])]).T
    else:
        dztrue = np.squeeze([[ztrue[i]-e.zmin, e.zmax-ztrue[i]] for i, e 
                            in enumerate(kwargs['data'])]).reshape(2, 1)
    deltadl = [np.sqrt((e.sigmadl/1e3)**2
        + (lk.sigma_weak_lensing(e.potential_galaxy_hosts[0].redshift,
                                 e.dl)/1e3)**2)
        for e in kwargs['data']]
    z = [np.median(x['z{}'.format(e.ID)]) for e in kwargs['data']]
    deltaz = [2*np.std(x['z{}'.format(e.ID)]) for e in kwargs['data']]
    redshift = np.logspace(-3, 1.0, 100)

    # Loop over the posterior samples to get all models
    # to then average for the plot.
    models = []
    for k in range(x.shape[0]):
        if ('LambdaCDM_h' in kwargs['model']): 
            omega = cs.CosmologicalParameters(
                x['h'][k], kwargs['truths']['om'], kwargs['truths']['ol'],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('LambdaCDM_om' in kwargs['model']):
            omega = cs.CosmologicalParameters(
                kwargs['truths']['h'], x['om'][k], 1.0-x['om'][k],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('LambdaCDM' in kwargs['model']):
            omega = cs.CosmologicalParameters(
                x['h'][k], x['om'][k],1.0-x['om'][k],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('CLambdaCDM' in kwargs['model']):
            omega = cs.CosmologicalParameters(
                x['h'][k], x['om'][k], x['ol'][k],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('LambdaCDMDE' in kwargs['model']):
            omega = cs.CosmologicalParameters(
                x['h'][k], x['om'][k], x['ol'][k], x['w0'][k], x['w1'][k])
        elif ('DE' in kwargs['model']):
            omega = cs.CosmologicalParameters(
                kwargs['truths']['h'], kwargs['truths']['om'],
                kwargs['truths']['ol'], x['w0'][k], x['w1'][k])
        else:
            omega = cs.CosmologicalParameters(
                kwargs['truths']['h'], kwargs['truths']['om'],
                kwargs['truths']['ol'], kwargs['truths']['w0'],
                kwargs['truths']['w1'])
        models.append([omega.LuminosityDistance(zi)/1e3 for zi in redshift])
        omega.DestroyCosmologicalParameters()

    models = np.array(models)
    (model2p5, model16, model50, model84, model97p5) = np.percentile(
        models, [2.5, 16.0, 50.0, 84.0, 97.5], axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(z, dl, xerr=deltaz, yerr=deltadl, markersize=1,
                linewidth=2, color='k', fmt='o')
    ax.plot(redshift, [kwargs['omega_true'].LuminosityDistance(zi)/1e3
            for zi in redshift], linestyle='dashed', color='red', zorder=22)
    ax.plot(redshift, model50, color='k')
    ax.errorbar(ztrue, dl, xerr=dztrue, yerr=deltadl, markersize=2,
                linewidth=1, color='r', fmt='o')
    ax.fill_between(redshift, model2p5, model97p5, facecolor='turquoise')
    ax.fill_between(redshift, model16, model84, facecolor='cyan')
    ax.set_xlabel(r"z", fontsize=16)
    ax.set_ylabel(r"$D_L$/Gpc", fontsize=16)
    fig.savefig(os.path.join(kwargs['outdir'], "Plots", 
                "MBHB_regression_68_95CI.pdf"), bbox_inches='tight')
    plt.close()


def rate_plots(x, **kwargs):
    """Plots when the rate is also estimated."""
    print("\nMaking rate plots...")
    pdf_z = []
    cdf_z = []
    z = np.linspace(0.0, kwargs['cosmo_model'].z_threshold, 500)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Ns = np.zeros(x.shape[0], dtype=np.float64)
    alpha = np.zeros(x.shape[0], dtype=np.float64)
    for i in range(x.shape[0]):
        if ('LambdaCDM_h' in kwargs['cosmo_model'].model):
            O = cs.CosmologicalParameters(
                x['h'][i], kwargs['truths']['om'], kwargs['truths']['ol'],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('LambdaCDM_om' in kwargs['cosmo_model'].model):
            O = cs.CosmologicalParameters(
                kwargs['truths']['h'], x['om'][i], 1.0-x['om'][i],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('LambdaCDM' in kwargs['cosmo_model'].model):
            O = cs.CosmologicalParameters(
                x['h'][i], x['om'][i], 1.0-x['om'][i],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('CLambdaCDM' in kwargs['cosmo_model'].model):
            O = cs.CosmologicalParameters(
                x['h'][i], x['om'][i], x['ol'][i], 
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('LambdaCDMDE' in kwargs['cosmo_model'].model):
            O = cs.CosmologicalParameters(
                x['h'][i], x['om'][i], x['ol'][i], x['w0'][i], x['w1'][i])
        elif ('DE' in kwargs['cosmo_model'].model):
            O = cs.CosmologicalParameters(
                kwargs['truths']['h'], kwargs['truths']['om'],
                kwargs['truths']['ol'], x['w0'][i], x['w1'][i])
        else:
            O = cs.CosmologicalParameters(
                kwargs['truths']['h'], kwargs['truths']['om'],
                kwargs['truths']['ol'], kwargs['truths']['w0'],
                kwargs['truths']['w1'])

        # Compute the expected rate of sources Ns (per year)
        # = R(zmax, lambda, O) integrated to the maximum redshift. 
        # This will also serve as normalisation constant for the 
        # individual dR/dz_i to obtain p(z)_i.
        if (kwargs['cosmo_model'].SFRD == 'powerlaw'):
            pop_model = astro.PopulationModel(
                10**x['log10r0'][i], x['p1'][i], 0.0, 0.0, 0.0,
                O, 1e-5, kwargs['cosmo_model'].z_threshold,
                density_model=kwargs['cosmo_model'].SFRD)
        else:    
            pop_model = astro.PopulationModel(
                10**x['log10r0'][i], 10**x['log10p1'][i], x['p2'][i],
                x['p3'][i], 0.0, O, 1e-5, kwargs['cosmo_model'].z_threshold,
                density_model=kwargs['cosmo_model'].SFRD)
        Ns[i] = pop_model.integrated_rate()
        # Compute the fraction of detectable events: 
        # alpha = = Ns_up_tot / Ns_tot. 
        alpha[i] = lk.number_of_detectable_gw(
            pop_model, kwargs['cosmo_model'].snr_threshold, kwargs['corr']) / Ns[i]
        # Compute events redshift PDF, p(z)_i = (dR/dz)_i / Ns.
        u = np.array([pop_model.pdf(zi) for zi in z])
        # Compute CDF of p(z)_i.
        v = np.array([pop_model.cdf(zi) for zi in z])
        pdf_z.append(u)
        cdf_z.append(v)

    if (kwargs['cosmo_model'].SFRD == 'powerlaw'):
        pop_model_true = astro.PopulationModel(
            kwargs['truths']['r0'], kwargs['truths']['p1'],
            0.0, 0.0, 0.0, kwargs['omega_true'], 1e-5,
            kwargs['cosmo_model'].z_threshold,
            density_model=kwargs['cosmo_model'].SFRD)
    else:    
        pop_model_true = astro.PopulationModel(
            kwargs['truths']['r0'], kwargs['truths']['p1'],
            kwargs['truths']['p2'], kwargs['truths']['p3'], 0.0,
            kwargs['omega_true'], 1e-5, kwargs['cosmo_model'].z_threshold,
            density_model=kwargs['cosmo_model'].SFRD)
    Ns_true = pop_model_true.integrated_rate() 
    alpha_true = lk.number_of_detectable_gw(
        pop_model_true, kwargs['cosmo_model'].snr_threshold,
        kwargs['corr']) / Ns_true
    pdf_z_true = np.array([pop_model_true.pdf(zi) for zi in z])
    cdf_z_true = np.array([pop_model_true.cdf(zi) for zi in z])
    pdf_z = np.array(pdf_z) 
    cdf_z = np.array(cdf_z)
    # Compute the true numbers of total sources happening in T
    # as a function of z, Ns_tot_true(z), by multiplying
    # Ns_true_tot by the CDF of p(z).
    Ns_tot_true_of_z = (kwargs['cosmo_model'].T * cdf_z_true)
    Ns_tot_of_z = kwargs['cosmo_model'].T * cdf_z
    # Compute quantiles over different pop samples.
    l_pdf_z, m_pdf_z, h_pdf_z = np.percentile(pdf_z, [5, 50, 95], axis=0)
    l_Ns_tot_of_z, m_Ns_tot_of_z, h_Ns_tot_of_z = np.percentile(
        Ns_tot_of_z, [5, 50, 95], axis=0)
    print("\nNs (per year) [.5, .50, .95] =", np.percentile(Ns, [5, 50, 95]))
    print("Ns_true (per year) = ", Ns_true)
    print("Observation time T (years) = ", kwargs['cosmo_model'].T,
          "\nNs (during T) [.5, .50, .95] =", np.percentile(Ns, [5, 50, 95])
                                              *  kwargs['cosmo_model'].T)          
    print("alpha = (Ns_up/Ns) [.5, .50, .95] =",
          np.percentile(alpha, [5, 50, 95]))
    print("alpha true = (Ns_up_true/Ns_true) = ", alpha_true)    

    # Plot event redshift distribution p(z).
    ax.plot(z, m_pdf_z, color='k', linewidth=.7)
    ax.fill_between(z, l_pdf_z, h_pdf_z, facecolor='lightgray')
    # ax.plot(z, pdf_z_true, linestyle='dashed', color=truth_color)
    ax.set_xlabel(r"$z$", fontsize=16)
    ax.set_ylabel(r"$p(z|\lambda\,\Omega\,I)$", fontsize=16)
    fig.savefig(os.path.join(kwargs['outdir'], "Plots",
                "redshift_distribution.pdf"), bbox_inches='tight')
    # Plot Ns_tot_of_z.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(z, m_Ns_tot_of_z, color='k', linewidth=.7)
    ax.fill_between(z, l_Ns_tot_of_z, h_Ns_tot_of_z, facecolor='lightgray')
    # ax.plot(z, Ns_tot_true_of_z, color=truth_color, linestyle='dashed')
    plt.yscale('log')
    ax.set_xlabel(r"$z$", fontsize=16)
    ax.set_ylabel(
        r"$R(z_{max},\lambda,\Omega)\cdot T\cdot CDF(z|\lambda\,\Omega\,I)$",
        fontsize=16)
    plt.savefig(os.path.join(kwargs['outdir'], "Plots",
                "total_number_of_events.pdf"), bbox_inches='tight')
    # Plot histogram of Ns for different hyperparams samples.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(Ns, bins=100, histtype='step')
    # ax.axvline(Ns_true, linestyle='dashed', color=truth_color)
    ax.set_xlabel(r"$R(z_{max},\lambda\,\Omega) \quad (yr^{-1})$", fontsize=16)
    ax.set_ylabel("Number of samples", fontsize=16)
    fig.savefig(os.path.join(kwargs['outdir'], "Plots",
                "global_rate.pdf"), bbox_inches='tight')
    # Plot alpha.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(alpha, bins=100, histtype='step')
    # ax.axvline(alpha_true, linestyle='dashed', color=truth_color)
    ax.set_xlabel(r"$\alpha$", fontsize=16)
    ax.set_ylabel("Number of samples", fontsize=16)
    fig.savefig(os.path.join(kwargs['outdir'], "Plots",
                "alpha.pdf"), bbox_inches='tight')


def luminosity_plots(x, **kwargs):
    """Plots when the luminosity function is estimated."""
    print("Making luminosity plots...")
    distributions = []
    luminosity_function_0 = []
    luminosity_function_1 = []
    luminosity_function_2 = []    
    Z = np.linspace(0.0, kwargs['cosmo_model'].z_threshold, 100)
    M = np.linspace(
        kwargs['cosmo_model'].Mmin, kwargs['cosmo_model'].Mmax, 100)

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    for i in range(x.shape[0]):
        sys.stderr.write(
            "Processing {0} out of {1} samples\r".format(i+1, x.shape[0]))
        phistar0 = x['phistar0'][i]
        phistar_exponent = x['phistar_exponent'][i]
        Mstar0 = x['Mstar0'][i]
        Mstar_exponent = x['Mstar_exponent'][i]
        alpha0 = x['alpha0'][i]
        alpha_exponent = x['alpha_exponent'][i]

        if ('LambdaCDM_h' in kwargs['cosmo_model'].model):
            O = cs.CosmologicalParameters(
                x['h'][i], kwargs['truths']['om'], kwargs['truths']['ol'],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('LambdaCDM_om' in kwargs['cosmo_model'].model):
            O = cs.CosmologicalParameters(
                kwargs['truths']['h'], x['om'][i], 1.0-x['om'][i],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('LambdaCDM' in kwargs['cosmo_model'].model):
            O = cs.CosmologicalParameters(
                x['h'][i], x['om'][i], 1.0-x['om'][i],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('CLambdaCDM' in kwargs['cosmo_model'].model):
            O = cs.CosmologicalParameters(
                x['h'][i], x['om'][i], x['ol'][i],
                kwargs['truths']['w0'], kwargs['truths']['w1'])
        elif ('LambdaCDMDE' in kwargs['cosmo_model'].model):
            O = cs.CosmologicalParameters(
                x['h'][i], x['om'][i], x['ol'][i], x['w0'][i], x['w1'][i])
        elif ('DE' in kwargs['cosmo_model'].model):
            O = cs.CosmologicalParameters(
                kwargs['truths']['h'], kwargs['truths']['om'],
                kwargs['truths']['ol'], x['w0'][i], x['w1'][i])
        else:
            O = cs.CosmologicalParameters(
                kwargs['truths']['h'], kwargs['truths']['om'],
                kwargs['truths']['ol'], kwargs['truths']['w0'],
                kwargs['truths']['w1'])

        S = gal.GalaxyDistribution(O,
                                   phistar0,
                                   phistar_exponent,
                                   Mstar0,
                                   Mstar_exponent,
                                   alpha0,
                                   alpha_exponent,
                                   kwargs['cosmo_model'].Mmin,
                                   kwargs['cosmo_model'].Mmax,
                                   0.0,
                                   kwargs['cosmo_model'].z_threshold,
                                   0.0,
                                   2.0*np.pi,
                                   -0.5*np.pi,
                                   0.5*np.pi,
                                   kwargs['cosmo_model'].magnitude_threshold,
                                   4.0*np.pi,
                                   1,
                                   1,
                                   1)

        PMZ = np.array([S.pdf(Mi, Zj, 1) for Mi in M
                       for Zj in Z]).reshape(100,100)
        distributions.append(PMZ)
        luminosity_function_0.append(
            np.array([S.luminosity_function(Mi, 1e-5, 0) for Mi in M]))
        luminosity_function_1.append(
            np.array([S.luminosity_function(Mi, S.zmax/2., 0) for Mi in M]))
        luminosity_function_2.append(
            np.array([S.luminosity_function(Mi, S.zmax, 0) for Mi in M]))

    sys.stderr.write("\n")
    distributions = np.array(distributions)
    pmzl, pmzm, pmzh = np.percentile(distributions, [5, 50, 95], axis=0)
    pl_0, pm_0, ph_0 = np.percentile(luminosity_function_0, 
                                     [5, 50, 95], axis=0)
    ax.fill_between(M, pl_0, ph_0, facecolor='magenta', alpha=0.5)
    ax.plot(M, pm_0, linestyle='dashed', color='r', label="z = 0.0")
    
    pl_1, pm_1, ph_1 = np.percentile(luminosity_function_1,
                                     [5, 50, 95], axis=0)
    ax.fill_between(M, pl_1, ph_1, facecolor='green', alpha=0.5)
    ax.plot(M, pm_1, linestyle='dashed', color='g', 
            label="z = {0:.1f}".format(S.zmax/2.))
    
    pl_2, pm_2, ph_2 = np.percentile(luminosity_function_2,
                                     [5, 50, 95], axis=0)
    ax.fill_between(M, pl_2, ph_2, facecolor='turquoise', alpha=0.5)
    ax.plot(M, pm_2, linestyle='dashed', color='b', 
            label="z = {0:.1f}".format(S.zmax))
    
    St = gal.GalaxyDistribution(cs.CosmologicalParameters(
                                    kwargs['truths']['h'],
                                    kwargs['truths']['om'],
                                    kwargs['truths']['ol'],
                                    kwargs['truths']['w0'],
                                    kwargs['truths']['w1']),
                                kwargs['truths']['phistar0'],
                                kwargs['truths']['phistar_exponent'],
                                kwargs['truths']['Mstar0'],
                                kwargs['truths']['Mstar_exponent'],
                                kwargs['truths']['alpha0'],
                                kwargs['truths']['alpha_exponent'],
                                kwargs['cosmo_model'].Mmin,
                                kwargs['cosmo_model'].Mmax,
                                0.0,
                                kwargs['cosmo_model'].z_threshold,
                                0.0,
                                2.0*np.pi,
                                -0.5*np.pi,
                                0.5*np.pi,
                                kwargs['cosmo_model'].magnitude_threshold,
                                4.0*np.pi,
                                1,
                                1,
                                1)
    
    ax.plot(M, np.array([St.luminosity_function(Mi, 1e-5, 0) for Mi in M]),
            linestyle='solid', color='k', lw=1.5, zorder=0)
    plt.legend(fancybox=True)
    ax.set_xlabel("magnitude", fontsize=16)
    ax.set_ylabel("$\phi(M|\Omega,I)$", fontsize=16)
    fig.savefig(os.path.join(kwargs['outdir'], "Plots",
                             "luminosity_function.pdf"), bbox_inches='tight')
    
    magnitude_probability = (np.sum(pmzl * np.diff(Z)[0], axis=1), 
                             np.sum(pmzm * np.diff(Z)[0], axis=1),
                             np.sum(pmzh * np.diff(Z)[0], axis=1))
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.fill_between(M, magnitude_probability[0], magnitude_probability[2],
                    facecolor='lightgray')
    ax.plot(M, magnitude_probability[1], linestyle='dashed', color='k')
    ax.hist(kwargs['cosmo_model'].galaxy_magnitudes, 100, density=True,
            facecolor='turquoise')
    ax.set_xlabel("magnitude", fontsize=16)
    ax.set_ylabel("$\phi(M|\Omega,I)$", fontsize=16)
    fig.savefig(os.path.join(kwargs['outdir'], "Plots",
                "luminosity_probability.pdf"), bbox_inches='tight')

    redshift_probability = (np.sum(pmzl * np.diff(M)[0], axis=0),
                            np.sum(pmzm * np.diff(M)[0], axis=0),
                            np.sum(pmzh * np.diff(M)[0], axis=0))
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.fill_between(Z, redshift_probability[0], redshift_probability[2],
                    facecolor='lightgray')
    ax.plot(Z, redshift_probability[1], linestyle='dashed', color='k')
    ax.hist(kwargs['cosmo_model'].galaxy_redshifts, 100, density=True,
            facecolor='turquoise')
    ax.set_xlabel("redshift", fontsize=16)
    ax.set_ylabel("$\phi(z|\Omega,I)$", fontsize=16)
    fig.savefig(os.path.join(kwargs['outdir'], "Plots",
                "galaxy_redshift_probability.pdf"), bbox_inches='tight')
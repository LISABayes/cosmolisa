import numpy as np
import scipy.stats
import os
import sys
import cosmolisa.cosmology as cs
import cosmolisa.likelihood as lk
import cosmolisa.galaxy as gal
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

from optparse import OptionParser

if __name__=="__main__":
    
    mmin    = -30
    mmax    = -15
    zmin    = 0.0
    zmax    = 1.0
    ramax   = 2.0*np.pi
    ramin   = 0.0
    decmax  = np.pi/2.0
    decmin  = -np.pi/2.0
    h       = 0.73
    om      = 0.25
    Mstar   = -20.5
    Mstar_exponent = -0.08
    cutoff_model_choice = 1
    alpha = -1.0
    alpha_exponent = 0.05
    slope_model_choice = 1
    apparent_magnitude_threshold=17
        
    O = cs.CosmologicalParameters(h, om, 1.0-om, -1.0, 0.0)
    S = gal.GalaxyDistributionLog(O,
                               Mstar,
                               Mstar_exponent,
                               alpha,
                               alpha_exponent,
                               mmin,
                               mmax,
                               zmin,
                               zmax,
                               ramin,
                               ramax,
                               decmin,
                               decmax,
                               apparent_magnitude_threshold,
                               slope_model_choice,
                               cutoff_model_choice)
    
#    galaxies = S.sample(10000)
#    print(galaxies)
    
    M = np.linspace(mmin,mmax,100)
    Z = np.linspace(zmin,zmax,100)
#    junk = []
#    for Zi in Z:
#        junk.append((S.alpha(Zi),S.Mstar(Zi)))
#    junk = np.array(junk)
#    fig = plt.figure()
#    plt.plot(Z,junk[:,0])
#    plt.show()
#    fig = plt.figure()
#    plt.plot(Z,junk[:,1])
#    plt.show()
    OM, OZ = np.meshgrid(M,Z)
    PMZ = np.array([S.pdf(Mi, Zj) for Mi in M for Zj in Z]).reshape(100,100)
    print("N = ",S._norm)
    print("phi = ",S.phistar,"Mpc^{-3}",S.omega.ComovingVolume(zmax))
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    S   = ax.plot_surface(OM, OZ, PMZ.T, cmap = cm.rainbow,
                          norm=colors.SymLogNorm(linthresh=1e-15, linscale=1e-15, vmin=0.0, vmax=PMZ.max(), base = 10.0))
    plt.colorbar(S)

#    n, dm, dz = np.histogram2d(galaxies, bins=100, density = True)
#    x = 0.5*(dm[1:]+dm[:-1])
#    y = 0.5*(dz[1:]+dz[:-1])
#    X, Y = np.meshgrid(x, y)
#    ax.scatter(galaxies[:,0], galaxies[:,1], np.log(galaxies[:,2]), c=np.log(galaxies[:,2]))
#    ax.plot(,np.log(n), color='turquoise')
#    ax.fill_between(0.5*(dm[1:]+dm[:-1]),-10,np.log(n),facecolor='turquoise',alpha=0.5)
#    ax.set_xlabel('absolute magnitude')
#    ax.set_ylabel(r'$\log$ probability')
##    ax.set_ylabel(r'$\log\frac{\phi^*}{\mathrm{Mpc}^{3}}$')
#    ax.set_ylim(-10,0)
    plt.show()
    

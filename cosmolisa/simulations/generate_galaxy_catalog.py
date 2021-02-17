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

def generate_proper_distances(Rmax,Ntot):
    n  = 0
    corr0 = gal.correlation_function([1e-2], 0.7)
    # pick a random position
    x0 = np.array([np.random.uniform(-Rmax,Rmax),
                   np.random.uniform(-Rmax,Rmax),
                   np.random.uniform(-Rmax,Rmax)])
    positions = [x0]
    while n < Ntot:
        # pick a random direction
        xp = np.array([np.random.uniform(-Rmax,Rmax),
                       np.random.uniform(-Rmax,Rmax),
                       np.random.uniform(-Rmax,Rmax)])
        r = np.linalg.norm(x0-xp)
        # check if a galaxy should be there
        # compute the 2-point correlation function for this distance
        corr = gal.correlation_function([r], 0.7)/corr0
        new_gal = int(np.random.poisson(1+corr))

        if new_gal > 0:
            n+=1
            positions.append(xp)
        x0 = xp
        
    return np.array(positions)

if __name__=="__main__":
    """
    @ARTICLE{2017arXiv170703003B,
    author = {{Bacry}, E. and {Bompaire}, M. and {Ga{\"i}ffas}, S. and {Poulsen}, S.},
    title = "{tick: a Python library for statistical learning, with
    a particular emphasis on time-dependent modeling}",
    journal = {ArXiv e-prints},
    eprint = {1707.03003},
    year = 2017,
    month = jul
    }
    """
    mmin    = -30
    mmax    = -15
    zmin    = 0.2
    zmax    = 0.25
#    ramax   = 2.0*np.pi
#    ramin   = 0.0
    ramax   = 1.0
    ramin   = 1.05
#    decmax  = np.pi/2.0
#    decmin  = -np.pi/2.0
    decmax  = -0.15
    decmin  = -0.2
    A       = (ramax-ramin)*(np.cos(decmax+np.pi/2)-np.cos(decmin+np.pi/2))
    h       = 0.73
    om      = 0.25
    n0      = 1e-2
    Mstar   = -20.7
    Mstar_exponent = -0.13
    cutoff_model_choice = 1
    alpha = -1.23
    alpha_exponent = 0.054
    slope_model_choice = 1
    density_model_choice = 1
    phistar_exponent = -0.1
    threshold = 20
    
#    n = 1000
#    positions = generate_proper_distances(1000,n)
###    print(rs.shape)
###    plt.plot(rs,'.')
###    plt.show()
###    exit()
##    phi = np.random.uniform(0,np.pi*2, size=n)
##    costheta = np.random.uniform(-1,1, size=n)
##    theta = np.arccos( costheta )
#    x = positions[:,0]
#    y = positions[:,1]
#    z = positions[:,2]
#    fig = plt.figure()
#    ax  = fig.add_subplot(111, projection='3d')
#    ax.scatter(x,y,z)
#    ax.set_xlabel("x/Mpc")
#    ax.set_ylabel("y/Mpc")
#    ax.set_zlabel("y/Mpc")
#    plt.show()
#    exit()
    
    
    O = cs.CosmologicalParameters(h, om, 1.0-om, -1.0, 0.0)
    S = gal.GalaxyDistribution(O,
                               n0,
                               phistar_exponent,
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
                               threshold,
                               A,
                               slope_model_choice,
                               cutoff_model_choice,
                               density_model_choice)
    
    N = S.get_number_of_galaxies(zmin,zmax,1)
    print("sampling {} galaxies".format(N))
    galaxies = S.sample_correlated(N, zmin, zmax, ramin, ramax, decmin, decmax, selection = 1)

    M = galaxies[:,0]
    z = galaxies[:,1]
    ra = galaxies[:,2]
    dec = galaxies[:,3]
    
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = '3d')
    SC  = ax.scatter(z,ra,dec,c=M,s=2)
    CB = plt.colorbar(SC)
    ax.set_xlabel('z')
    ax.set_ylabel('ra')
    ax.set_zlabel('dec')
    plt.show()
    
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    SC = ax.scatter(ra,dec,c=z)
    CB = plt.colorbar(SC)
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    SC = ax.scatter(z,M)
    CB = plt.colorbar(SC)
    plt.show()
    exit()

    
    
    M = np.linspace(mmin,mmax,256)
    Z = np.linspace(zmin,zmax,256)

    OM, OZ = np.meshgrid(M,Z)
    PMZ = np.array([S.pdf(Mi, Zj, 0) for Mi in M for Zj in Z]).reshape(256,256)
    PMZd = np.array([S.pdf(Mi, Zj, 1) for Mi in M for Zj in Z]).reshape(256,256)

    print("N total = ",S.get_number_of_galaxies(zmin,zmax,0))
    print("N detected = ",S.get_number_of_galaxies(zmin,zmax,1))
    PM = np.sum(PMZ*np.diff(Z)[0], axis = 1)
    PZ = np.sum(PMZ*np.diff(M)[0], axis = 0)
    PMd = np.sum(PMZd*np.diff(Z)[0], axis = 1)
    PZd = np.sum(PMZd*np.diff(M)[0], axis = 0)
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.semilogy(M, PM)
    ax.semilogy(M, PMd, '--r')
    ax.set_xlabel("absolute magnitude")
    ax.set_ylabel(r"$p(M)$")
    
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(Z, PZ)
    ax.plot(Z, PZd, '--r')
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"$p(z)$")
    
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    for z in np.linspace(1e-3,zmax,10):
        ax.plot(M, np.array([S.luminosity_function(Mi, z, 0) for Mi in M]), linestyle='solid')
        ax.plot(M, np.array([S.luminosity_function(Mi, z, 1) for Mi in M]), linestyle='dashed',label = "z = {0:.2f}".format(z))
    ax.set_xlabel("absolute magnitude")
    ax.set_ylabel(r"$\phi(M,z)$")
    plt.legend()
    plt.show()
    

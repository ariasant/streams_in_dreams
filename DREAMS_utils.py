import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import h5py
import math
import numpy as np
import pynbody
import sys
sys.path.append("/mnt/home/asante/ceph/repos/")
from nightmares.reader_funcs import load_zoom_particle_data_pynbody

def rotate_galaxy(dat,r_max=None):

    """
    Transforms coordinates frame of reference such that the z-axis points
    into the same direction as the total angular momentum of the galaxy.

    Inputs
    ------
    x,y,z           array(float)       x/y/z coordinate as read from the simulation output,
                                        i.e. box coordinates, but w.r.t galaxy centre

    vx,vy,vz        array(float)       vx/vy/vz components as read from the simulation output,
                                        i.e. box coordinates, but w.r.t. galaxy centre

    mass            array(float)       mass of the particles.

    r_max           float              maximum distance of a particle to the centre of the galaxy 
                                        to be considered in the total angular momentum estimation.

    Returns
    -------
    xnew,ynew,znew,     array(float)    positions and velocity components of particles in the new 
    vxnew,vynew,vznew                   frame of reference.
    """

    x = dat["x"][...]
    y = dat["y"][...]
    z = dat["z"][...]
    vx = dat["vx"][...]
    vy = dat["vy"][...]
    vz = dat["vz"][...]

    if "Masses" in dat.keys():
        mass = dat["Masses"][...]
    elif "mass" in dat.keys():
        mass = dat["mass"]

    r = np.sqrt(x**2 + y**2 + z**2)

    if r_max is not None:

        disk_stars = np.where((r<0.15*r_max))[0]

    else:

        disk_stars = np.where((r<0.15*r.max()))[0]


    mass = mass[disk_stars]

    vxd = vx[disk_stars]
    vyd = vy[disk_stars]
    vzd = vz[disk_stars]
    xd = x[disk_stars]
    yd = y[disk_stars]
    zd = z[disk_stars]


    #Find velocity centre                                                       
    vxcm = np.sum(vxd*mass)/np.sum(mass)
    vycm = np.sum(vyd*mass)/np.sum(mass)
    vzcm = np.sum(vzd*mass)/np.sum(mass)

    #Express velocity in center frame                                           
    vxd -= vxcm
    vyd -= vycm
    vzd -= vzcm

    #Angular momentum of star particles                                         
    Jx = (yd*vzd - zd*vyd)
    Jy = (zd*vxd - xd*vzd)
    Jz = (xd*vyd - yd*vxd)

    #Mass-weighted mean angular momentum                                        
    Jxtot = np.sum(mass*Jx)/np.sum(mass)
    Jytot = np.sum(mass*Jy)/np.sum(mass)
    Jztot = np.sum(mass*Jz)/np.sum(mass)

    #Find the angles of 3d roatation matrix by finding which combination of alphalpha, beta, gamma gives the highest Jz                                            
    alpha = np.linspace(0,2*np.pi,361)
    beta = np.linspace(0,2*np.pi,361)
    gamma_tmp = np.linspace(0,np.pi,181)

    d0 = []
    Jztotmax = []


    for m in range(180):
        gamma = gamma_tmp[m]
        #Elements of rotation matrix                                            
        a11 = np.cos(beta)*np.cos(alpha)-np.cos(gamma)*np.sin(alpha)*np.sin(beta)
        a12 = np.cos(beta)*np.sin(alpha)+np.cos(gamma)*np.cos(alpha)*np.sin(beta)
        a13 = np.sin(beta)*np.sin(gamma)
        a21 = -np.sin(beta)*np.cos(alpha)-np.cos(gamma)*np.sin(alpha)*np.cos(beta)
        a22 = -np.sin(beta)*np.sin(alpha)+np.cos(gamma)*np.cos(alpha)*np.cos(beta)
        a23 = np.cos(beta)*np.sin(gamma)
        a31 = np.sin(gamma)*np.sin(alpha)
        a32 = -np.sin(gamma)*np.cos(alpha)
        a33 = np.cos(gamma)

        Jxtotnew = a11*Jxtot+a12*Jytot+a13*Jztot
        Jytotnew = a21*Jxtot+a22*Jytot+a23*Jztot
        Jztotnew = a31*Jxtot+a32*Jytot+a33*Jztot

        #Index of which combination of alpha and beta gives the maximum Jz given a certain gamma                                                               
        dn = np.where(Jztotnew==Jztotnew.max())[0]
        Jztotmax0 = Jztotnew.max()
        Jxtotmax0 = Jxtotnew[dn]
        Jytotmax0 = Jytotnew[dn]

        d0.append(dn)
        Jztotmax.append(Jztotmax0)


    d1 = np.where(Jztotmax==np.max(Jztotmax))[0]
    d2 = d0[d1[0]]

    b11 = np.cos(beta[d2])*np.cos(alpha[d2])-np.cos(gamma_tmp[d1])*np.sin(alpha[d2])*np.sin(beta[d2])
    b12 = np.cos(beta[d2])*np.sin(alpha[d2])+np.cos(gamma_tmp[d1])*np.cos(alpha[d2])*np.sin(beta[d2])
    b13 = np.sin(beta[d2])*np.sin(gamma_tmp[d1])
    b21 = -np.sin(beta[d2])*np.cos(alpha[d2])-np.cos(gamma_tmp[d1])*np.sin(alpha[d2])*np.cos(beta[d2])
    b22 = -np.sin(beta[d2])*np.sin(alpha[d2])+np.cos(gamma_tmp[d1])*np.cos(alpha[d2])*np.cos(beta[d2])
    b23 = np.cos(beta[d2])*np.sin(gamma_tmp[d1])
    b31 = np.sin(gamma_tmp[d1])*np.sin(alpha[d2])
    b32 = -np.sin(gamma_tmp[d1])*np.cos(alpha[d2])
    b33 = np.cos(gamma_tmp[d1])

    xnew = b11*x+b12*y+b13*z
    ynew = b21*x+b22*y+b23*z
    znew = b31*x+b32*y+b33*z

    vxnew = b11*vx+b12*vy+b13*vz
    vynew = b21*vx+b22*vy+b23*vz
    vznew = b31*vx+b32*vy+b33*vz

    rotation_matrix = np.array([[b11,b12,b13],
                                [b21,b22,b23],
                                [b31,b32,b33]])
    
    dat["x"] = xnew
    dat["y"] = ynew
    dat["z"] = znew
    dat["vx"]= vxnew
    dat["vy"]= vynew
    dat["vz"]= vznew

    return dat, rotation_matrix[:,:,0]



def get_rotation_matrix(box: int,
                        snap_path: str,
                        group_path: str):
    """
    Returns the rotation matrix to transform the box coordinates
    such that the z-axis is aligned to the total angular momentum
    vector of the stellar disc.
    """

    # Load star particles at z=0
    dat, grp_dat = load_zoom_particle_data_pynbody(snap_path, 
                                                   group_path, 
                                                   box, 
                                                   90, # snap number
                                                   4, # PartType
                                                   subhaloes=False)
    

    # Center the galaxy, convert to physical units, and rotate to face-on
    pynbody.analysis.center(dat, mode='ssc')
    dat.physical_units()
    dat,rotation_matrix = rotate_galaxy(dat=dat) 

    return rotation_matrix



def return_density(logr,weights=1.,rangevals=[-2, 6],bins=500,d2=False):
    """return_density

    simple binned density using logarithmically spaced bins

    inputs
    ---------
    logr        : (array) log radii of particles to bin
    weights     : (float or array) if float, single-mass of particles, otherwise array of particle masses
    rangevals   : (two value list) minimum log r, maximum log r
    bins        : (int) number of bins
    d2          : (bool) if True, compute surface density

    returns
    ---------
    rcentre     : (array) array of sample radii (NOT LOG)
    density     : (array) array of densities sampled at rcentre (NOT LOG)

    """

    # assume evenly spaced logarithmic bins
    dr      = (rangevals[1]-rangevals[0])/bins
    rcentre = np.zeros(bins)
    density = np.zeros(bins)

    # check if single mass, or an array of masses being passed
    # construct array of weights
    if isinstance(weights,float):
        w = weights*np.ones(logr.size)
    else:
        w = weights

    for indx in range(0,bins):

        # compute the centre of the bin (log r)
        rcentre[indx] = rangevals[0] + (indx+0.5)*dr

        # compute dr (not log)
        rmin,rmax = 10.**(rangevals[0] + (indx)*dr),10.**(rangevals[0] + (indx+1)*dr)
        if d2:
            shell = np.pi*(rmax**2-rmin**2)
        else:
            shell = (4./3.)*np.pi*(rmax**3.-rmin**3.)

        # find all particles in bin
        inbin = np.where((logr>=(rangevals[0] + (indx)*dr)) & (logr<(rangevals[0] + (indx+1)*dr)))

        # compute M/V for the bin
        density[indx] = np.nansum(w[inbin])/shell

    # return
    return 10.**rcentre,density


def makemodel_empirical(rvals,dvals,pfile='',plabel = '',verbose=True):
    """make an EXP-compatible spherical basis function table

    inputs
    -------------
    rvals       : (array of floats) radius values to evaluate the density function
    pfile       : (string) the name of the output file. If '', will not print file
    plabel      : (string) comment string, printed to the top of the file
    verbose     : (boolean)

    outputs
    -------------
    R           : (array of floats) the radius values
    D           : (array of floats) the density
    M           : (array of floats) the mass enclosed
    P           : (array of floats) the potential

    """
    M = 1.
    R = np.nanmax(rvals)

    # query out the density values
    #dvals = D#func(rvals,*funcargs)
    #print(R.size,)

    # make the mass and potential arrays
    mvals = np.zeros(dvals.size)
    pvals = np.zeros(dvals.size)
    pwvals = np.zeros(dvals.size)

    # initialise the mass enclosed an potential energy
    mvals[0] = 1.e-15
    pwvals[0] = 0.

    # evaluate mass enclosed and potential energy by recursion
    for indx in range(1,dvals.size):
        mvals[indx] = mvals[indx-1] +\
          2.0*np.pi*(rvals[indx-1]*rvals[indx-1]*dvals[indx-1] +\
                 rvals[indx]*rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
        pwvals[indx] = pwvals[indx-1] + \
          2.0*np.pi*(rvals[indx-1]*dvals[indx-1] + rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);

    # evaluate potential (see theory document)
    pvals = -mvals/(rvals+1.e-10) - (pwvals[dvals.size-1] - pwvals)

    # get the maximum mass and maximum radius
    M0 = mvals[dvals.size-1]
    R0 = rvals[dvals.size-1]

    # compute scaling factors
    Beta = (M/M0) * (R0/R);
    Gamma = np.sqrt((M0*R0)/(M*R)) * (R0/R);
    if verbose:
        print("! Scaling:  R=",R,"  M=",M)

    rfac = np.power(Beta,-0.25) * np.power(Gamma,-0.5);
    dfac = np.power(Beta,1.5) * Gamma;
    mfac = np.power(Beta,0.75) * np.power(Gamma,-0.5);
    pfac = Beta;

    if verbose:
        print(rfac,dfac,mfac,pfac)

    # save file if desired
    if pfile != '':
        f = open(pfile,'w')
        print('! ',plabel,file=f)
        print('! R    D    M    P',file=f)

        print(rvals.size,file=f)

        for indx in range(0,rvals.size):
            print('{0} {1} {2} {3}'.format( rfac*rvals[indx],\
              dfac*dvals[indx],\
              mfac*mvals[indx],\
              pfac*pvals[indx]),file=f)

        f.close()

    return rvals*rfac,dfac*dvals,mfac*mvals,pfac*pvals


def get_cosmology(box,
                  snap_path):

        # Define cosmology of the simulation
        f = h5py.File(f"{snap_path}box_{box}/snap_090.hdf5")

        Om_0 = f["Header"].attrs["Omega0"]
        H0 = f["Header"].attrs["HubbleParam"]*100 * u.km / u.s / u.Mpc

        cosmo = FlatLambdaCDM(H0=H0, Om0=Om_0)

        return cosmo



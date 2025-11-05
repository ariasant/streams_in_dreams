import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import h5py
import numpy as np
import pynbody
import sys
sys.path.append("/mnt/home/asante/ceph/repos/")
from pynbody import units



def get_MW_idx(cat, model: str, param_dict: dict):
    """
    Selects the corrent MW-mass galaxy from each simulation given the group catalog.
    This function only works for z~0
    It selects the least contaminated halo with a mass within current uncertainties of the MW's mass
    
    Inputs 
     - cat - a dictionary containing the 'GroupMassType' field from the FOF catalogs
     
    Returns
     - mw_idx - the index into the group catalog for the target MW-mass galaxy
    """
    
    h = param_dict['H0']/100
    
    masses = cat['GroupMassType'] * 1e10 / h
    
    tot_masses = np.sum(masses,axis=1)
    if model == 'CDM':
        mcut = (tot_masses > 5e11) & (tot_masses < 2.5e12)
    elif model == 'WDM':
        mcut = (tot_masses > 7e11) & (tot_masses < 2.5e12)
    else:
        print('no galaxies with this model yet!')
    if True in np.unique(mcut):
        contamination = masses[:,2] / tot_masses
        idx = np.argmin(contamination[mcut])
        mw_idx = np.arange(len(masses))[mcut][idx]
    else:
        mw_idx = None
    return mw_idx

def load_group_data(path, keys=None):
    """
    Read Group Data from the DREAMS simulations
    
    Inputs
      path - the absolute or relative path to the hdf5 file you want to read from
      keys - the data that you want to read from the simulation 
             see https://www.tng-project.org/data/docs/specifications/ for a list of available data)
      
    Returns
      cat - a dictionary that contains all of the group and subhalo information for the specified keys
    """
    cat = dict()
    file = h5py.File(path)
    if keys is None:
        # Load all keys:
        keys = list(file["Group"].keys()) + list(file["Subhalo"].keys())
    
    for key in keys:
        if 'Group' in key:
            cat[key] = np.array(file[f'Group/{key}'])
        if 'Subhalo' in key:
            cat[key] = np.array(file[f'Subhalo/{key}'])
    file.close()
    
    return cat

def load_particle_data(snap_path, box, snap, part_types):
    """
    revised from original, will use all keys instead of passing in subset 
    Read particle data from the DREAMS simulations
    
    Inputs
      path - the absolute or relative path to the hdf5 file you want to read from
      keys - the data that you want to read from the simulation 
             see https://www.tng-project.org/data/docs/specifications/ for a list of available data)
      part_types - which particle types to load.
                   0 - gas
                   1 - high res dark matter
                   2 - low res dark matter
                   3 - tracers (not used in DREAMS)
                   4 - stars
                   5 - black holes
      
    Returns
      cat - a dictionary that contains all of the particle information for the specified keys and particle types
    """
    
    path = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
    
    cat = dict()
    with h5py.File(path) as ofile:
        
        if type(part_types) == type(0):
            part_types = [part_types]
        
        for part_type in part_types:
            if part_type <= 5:
                keys = ofile[f'PartType{part_type}'].keys()
                for key in keys:
                    if part_type == 1 and key == 'Masses':
                        cat[f'PartType{part_type}/{key}'] = np.ones(ofile['PartType1/ParticleIDs'].shape)*ofile['Header'].attrs['MassTable'][1]
                    else:
                        if f'PartType{part_type}/{key}' in ofile:
                            cat[f'PartType{part_type}/{key}'] = np.array(ofile[f'PartType{part_type}/{key}'])
            else:
                print('Particle type does not exist, try an integer <= 5')
                return
    return cat

def config_pynbody_units(snap_path, box, snap):
    
    #load in to find scale factor:
    path = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
    param_info = np.loadtxt(f'{snap_path}/box_{box}/aux_files/ics_config.txt',  dtype='str', skiprows=20, max_rows=6)
    
    check = h5py.File(path)
    a = 1/(check['Header'].attrs['Redshift']+1)
    check.close()
    
    param_dict = {}
    for i in range(len(param_info)):
        param_dict[param_info[i][0]] = float(param_info[i][2])

    pynbody.config['omegaM0'] = float(param_dict['Omega_m'])
    pynbody.config['omegaL0'] = float(param_dict['Omega_L'])
    pynbody.config['h'] = float(param_dict['H0'])/100 #should be .6909, but file gives 69.09
    pynbody.config['omegaB0'] = float(param_dict['Omega_b'])
    pynbody.config['sigma8'] = float(param_dict['sigma_8'])
    pynbody.config['ns'] = float(param_dict['nspec'])
    pynbody.config['a'] = a
    pynbody.units.a = a
    pynbody.units.h = float(param_dict['H0'])/100
    
    unit_dict = {'BirthPos': units.kpc * units.h**-1,
            'BirthVel': units.a**1/2 * units.km * units.s**-1,
            'Coordinates': units.kpc *units.a * units.h**-1,
            'GFM_InitialMass': 1e10 * units.Msol * units.h**-1,
            'GFM_Metallicity': units.Unit(1),
            'GFM_Metals': units.Unit(1),
            'GFM_MetalsTagged': units.Unit(1),
            'GFM_StellarFormationTime': units.Unit(1),
            'GFM_StellarPhotometrics': units.Unit(1),
            'Masses': 1e10 * units.Msol * units.h**-1,
            'ParticleIDs': units.Unit(1),
            'Potential': units.km**2 * units.s**-2 * units.a**-1,
            'SubfindDMDensity': 1e10 * units.Msol * units.h**2 * units.kpc**-3*units.a**-3,
            'SubfindDensity': 1e10 * units.Msol * units.h**2 * units.kpc**-3*units.a**-3,
            'SubfindHsml': units.kpc * units.a * units.h**-1,
            'SubfindVelDisp': units.km * units.s**-1,
            'Velocities': units.km * units.a**1/2 * units.s**-1,
            'CenterOfMass': units.kpc * units.a * units.h**-1,
            'Density': 1e10 * units.Msol * units.h**2 * units.kpc**-3*units.a**-3,
            'ElectronAbundance': units.Unit(1),
            'GFM_AGNRadiation': units.erg * units.s**-1 * units.cm**-2 * 4 * np.pi,
            'GFM_CoolingRate': units.erg * units.s**-1 * units.cm**3,
            'GFM_Metallicity': units.Unit(1),
            'GFM_Metals': units.Unit(1),
            'GFM_MetalsTagged': units.Unit(1),
            'GFM_WindDMVelDisp': units.km * units.s**-1,
            'GFM_WindHostHaloMass': 1e10 * units.Msol * units.h**-1,
            'InternalEnergy': units.km**2 * units.s**-2,
            'MagneticField': units.h * units.a**-2 * 1e5 * units.Msol**1/2 * units.kpc**-1/2* units.km * units.s**-1 * units.kpc**-1,
            'MagneticFieldDivergence': units.h**3 * 1e5 * units.Msol**1/2 * units.km * units.a**-2 * units.s**-1 * units.kpc**-5/2 * units.a**-5/2,
            'NeutralHydrogenAbundance': units.Unit(1),
            'StarFormationRate': units.Msol * units.yr**-1,
            'InternalEnergy': units.km**2 * units.s**-2,
            'AllowRefinement': units.Unit(1),
            'HighResGasMass': units.Unit(1)} #high res gas mass isn't defined in the tng webpage, gonna just set unit to one
    
    return unit_dict, param_dict


def get_MW_idx_at_snap(snap: int,
                       group_path: str,
                       box: int,
                       param_dict
                       ):
    
    # Get MW-mass halo from z = 0
    fof_path90 = f'{group_path}/box_{box}/fof_subhalo_tab_{90:03}.hdf5'
    grp_cat90 = load_group_data(fof_path90)
    model = group_path.split('/')[-4]
    
    mw_idx = get_MW_idx(grp_cat90, model, param_dict) 
    
    if mw_idx is None:
        raise ValueError('No MW-like mass systems in this box!')

    
    #can't use the z = 0 MW mass idx finder here - need to identify the MW-mass halo at z = 0
    #and then trace it thru time. Will need to load the merger tree to identify the correct halo
    
    tree_cat = h5py.File(group_path+'/box_'+str(int(box))+'/tree_extended.hdf5')
    
    mw_tree_id = tree_cat["SubhaloID"][(tree_cat["SnapNum"][:]==90) &
                                        (tree_cat["SubhaloLen"][:]==grp_cat90["SubhaloLen"][mw_idx])][0]
    subfind_idx = tree_cat["SubfindID"][tree_cat["SubhaloID"][...]==mw_tree_id][0]
    
    main_branch = {90: subfind_idx}

    FirstProgID = tree_cat["FirstProgenitorID"][tree_cat["SubhaloID"][...]==mw_tree_id]

    while FirstProgID!=-1:
        snap_num = tree_cat["SnapNum"][tree_cat["SubhaloID"][...]==FirstProgID][0]
        subfind_idx = tree_cat["SubfindID"][tree_cat["SubhaloID"][...]==FirstProgID][0]
        main_branch[snap_num] = subfind_idx
        
        FirstProgID = tree_cat["FirstProgenitorID"][tree_cat["SubhaloID"][...]==FirstProgID]
        
    return main_branch[snap]
    



def load_zoom_particle_data_pynbody(snap_path: str, 
                                    group_path: str, 
                                    box: int, 
                                    snap: int, 
                                    part_type: int):
    '''take in the snapshot path, the group path, the number box that you want
    the snapshot of, the snapshot number (i.e. what time, here z ~ 0 = 90), 
    the particle type. This will load all keys and port the data into pynbody with the correct cosmology
                    0 - gas
                   1 - high res dark matter
                   2 - low res dark matter
                   3 - tracers (not used in DREAMS)
                   4 - stars
                   5 - black holes
    pass in whether or not you want to load subhaloes (no subhaloes = False, default)
    '''

    
    # Configure pynbody with the cosmology of the simulation
    unit_dict, param_dict = config_pynbody_units(snap_path, box, snap)
    name_map = pynbody.snapshot.namemapper.AdaptiveNameMapper('gadgethdf-name-mapping',return_all_format_names=False)
    

    # Load subfind information
    fof_path = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
    grp_cat = load_group_data(fof_path)


    # Identify the index in the subfind table of the MW mass halo at this snapshot
    mw_idx = get_MW_idx_at_snap(snap,
                                group_path,
                                box,
                                param_dict,
                                )
    
    # If there is a larger halo, ignore the particles belonging to it
    offsets = np.sum(grp_cat['GroupLenType'][:mw_idx],axis=0)
    # Get how many particles of each type are in the MW halo
    num_parts = grp_cat['SubhaloLenType'][grp_cat['GroupFirstSub'][mw_idx]] 

    # Load all the particles in the simulation
    dat = load_particle_data(snap_path, box, snap, [part_type])
    
    # Create output catalog
    if part_type==0:
        out = pynbody.new(gas=int(num_parts[part_type]))
    elif part_type==1:
        out = pynbody.new(dm=int(num_parts[part_type]))
        # Masses are not stored with the other info
        with h5py.File(f'{snap_path}/box_{box}/snap_{snap:03}.hdf5') as ofile:
            out['Masses'] = np.ones(ofile['PartType1/ParticleIDs'][offsets[part_type]:offsets[part_type]+num_parts[part_type]].shape)*ofile['Header'].attrs['MassTable'][1]
            mapped_name = name_map('Masses', reverse=True)
            out[mapped_name].units = unit_dict['Masses']
            out[mapped_name] = np.ones(ofile['PartType1/ParticleIDs'][offsets[part_type]:offsets[part_type]+num_parts[part_type]].shape)*ofile['Header'].attrs['MassTable'][1]
    elif part_type==4:
        out = pynbody.new(star=int(num_parts[part_type]))
    
    # Load all the other fields from 
    for key in dat.keys():
        key = key.split('/')[1]
        mapped_name = name_map(key, reverse=True)
        out[mapped_name] = dat[f'PartType{part_type}/{key}'][offsets[part_type]:offsets[part_type]+num_parts[part_type]]
        out[mapped_name].units = unit_dict[key]
        
    
    # Center the halo using the shrink sphere center method
    pynbody.analysis.center(out, 
                            mode='ssc',
                            with_velocity=True, # Correct for the motion of the center of mass of the halo
                            cen_size="5 kpc")
    

    # Convert to physical units
    out.physical_units()

    return out


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
    dat = load_zoom_particle_data_pynbody(snap_path, 
                                          group_path, 
                                          box, 
                                          90, # snap number
                                          4, # PartType
                                          )
    
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


def makemodel(func,M,funcargs,rvals = 10.**np.linspace(-2.,4.,2000),pfile='',plabel = '',verbose=True):
    """make an EXP-compatible spherical basis function table
    
    inputs
    -------------
    func        : (function) the callable functional form of the density
    M           : (float) the total mass of the model, sets normalisations
    funcargs    : (list) a list of arguments for the density function.
    rvals       : (array of floats) radius values to evaluate the density function
    pfile       : (string) the name of the output file. If '', will not print file
    plabel      : (string) comment string
    verbose     : (boolean)
    outputs
    -------------
    R           : (array of floats) the radius values
    D           : (array of floats) the density
    M           : (array of floats) the mass enclosed
    P           : (array of floats) the potential
    
    """
    
    R = np.nanmax(rvals)
    
    # query out the density values
    dvals = func(rvals,*funcargs)

    # make the mass and potential arrays
    mvals = np.zeros(dvals.size)
    pvals = np.zeros(dvals.size)
    pwvals = np.zeros(dvals.size)

    # initialise the mass enclosed an potential energy
    mvals[0] = 1.e-15
    pwvals[0] = 0.

    # evaluate mass enclosed and potential energy by recursion
    for indx in range(1,dvals.size):
        mvals[indx] = mvals[indx-1] +          2.0*np.pi*(rvals[indx-1]*rvals[indx-1]*dvals[indx-1] +                 rvals[indx]*rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
        pwvals[indx] = pwvals[indx-1] +           2.0*np.pi*(rvals[indx-1]*dvals[indx-1] + rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
    
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
            print('{0} {1} {2} {3}'.format( rfac*rvals[indx],              dfac*dvals[indx],              mfac*mvals[indx],              pfac*pvals[indx]),file=f)
    
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




def characteristic_density(c):
    delta = 200/3 * c**3 / (np.log(1+c) - c/(1+c))
    return delta

def NFW_profile(r, c, R_vir, rho_crit):
    
    x = r/(R_vir/c)

    return  rho_crit * characteristic_density(c) / (x * (1+x)**2)
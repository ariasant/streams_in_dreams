import sys
sys.path.append("/mnt/home/asante/ceph/streams_in_dreams")

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import math
import numpy as np
import os
import pynbody
import DREAMS_utils
import pyEXP
import yaml
import h5py
import k3d
from gala.units import SimulationUnitSystem
import gala.potential as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit



import os




class DREAMSMW():

    def __init__(self, 
                 box: int,
                 snap_path: str,
                 group_path: str):
        
        self.__box__ = box
        self.__snap_path__ = snap_path
        self.__group_path__ = group_path

        ## Set coordinates frame of reference
        self.rotation_matrix = DREAMS_utils.get_rotation_matrix(box=self.__box__,
                                                                snap_path=self.__snap_path__,
                                                                group_path=self.__group_path__)
        
        ## Define the cosmology of the simulation
        self.cosmo = DREAMS_utils.get_cosmology(box=box,
                                                snap_path=self.__snap_path__)
        
        # Calculate characteristic scales of the galaxy
        self.r_scale, self.r_vir, self.M_vir = self.__fit_nfw__(snap=90)
        
        dat = self.__load_part_data__(snap=90, PartType=4)
        disc_star_ids = self.select_disc_stars(dat, 
                                               k_threshold=0.7, 
                                               r_max=30,
                                               z_max=10)
        disc_dat = dat[np.isin(dat["iord"], disc_star_ids)]
        self.r_scale_disc = self.__fit_scale_radius__(disc_dat)
        self.z_scale_disc = self.__fit_scale_height__(disc_dat)
        
        print(f"""Galaxy {box}:\r
               Scale Radius (DM): \t\t{self.r_scale:.1f} kpc\r
               Virial Radius: \t\t{self.r_vir:.1f} kpc\r
               Virial Mass: \t\t{self.M_vir:.1g} M_sun\r
               Disc Scale Radius: \t\t{self.r_scale_disc:.1f} kpc\r
               Disc Scale Height: \t\t{self.z_scale_disc:.1f} kpc
               """)

    def __load_part_data__(self,
                           snap: int, 
                           PartType: int,
                           rotate: bool = True,
                           ):
    

        # Get raw simulation data
        dat = DREAMS_utils.load_zoom_particle_data_pynbody(self.__snap_path__, 
                                                           self.__group_path__, 
                                                           self.__box__, 
                                                           snap, 
                                                           PartType
                                                           )

        if rotate:
            # Rotate to align total angular momentum vector to z-axis
            r_sim = np.stack([dat["x"], dat["y"], dat["z"]])
            v_sim = np.stack([dat["vx"], dat["vy"], dat["vz"]])
            # Apply rotation
            r_rot = self.rotation_matrix @ r_sim
            v_rot = self.rotation_matrix @ v_sim

            for i,xi in enumerate(["x","y","z"]):
                dat[f"{xi}"] = r_rot[i]
                dat[f"v{xi}"] = v_rot[i]


        return dat
    
    
    def select_disc_stars(self,
                          dat,
                          k_threshold=0.5, 
                          r_max=1000,
                          z_max=1000
                          ):
        
        dat['R'] = np.sqrt(dat['x']**2 + dat['y']**2)

        # Calculate velocity in cylindrical coordinates
        dat['vr'] = (dat['vx']*dat['x'] + dat['vy']*dat['y']) / dat['R'] # Radial velocity
        dat['vtheta'] = (dat['vy']*dat['x']-dat['vx']*dat['y']) / dat['R'] # Rotational velocity

        # Calculate the kinetic energy of the star particles
        K_tot = (dat["vx"]**2 + dat["vy"]**2 + dat["vz"]**2)
        
        # Select disc stars based on the amount of kinetic energy in rotation
        K_rot = dat['vtheta']**2
        
        # Optionally also based on distance to centre of the galaxy
        disc_stars_ids = dat["iord"][(K_rot/K_tot>k_threshold) & (dat["r"]<r_max) & (dat["z"]**2<z_max**2)]
        
        return disc_stars_ids
    
    

    def plot_center_L_evolution(self, 
                                snapshots: list[int]):
        
        centre_pos = [] # box coordinates
        L_tilts = [] # cos(alpha)
        times = [] # Gyr

        for snap in snapshots:

            # Read age of the universe at snapshot
            f = h5py.File(f"{self.__snap_path__}box_{self.__box__}/snap_{snap:03}.hdf5")
            times.append(self.cosmo.age(f["Header"].attrs["Redshift"]).value)

            # Get raw simulation data
            dat, grp_dat = DREAMS_utils.load_zoom_particle_data_pynbody(self.__snap_path__, 
                                                                        self.__group_path__, 
                                                                        self.__box__, 
                                                                        snap, 
                                                                        4, # stars
                                                                        )
            
            # Save position of centre in box coordinates
            centre_pos.append(pynbody.analysis.halo.shrink_sphere_center(dat))

            # Select only stars in the inner galaxy
            dat = dat[dat["r"]<20]

            # Rotate to align total angular momentum vector to z-axis
            r_sim = np.stack([dat["x"], dat["y"], dat["z"]])
            v_sim = np.stack([dat["vx"], dat["vy"], dat["vz"]])
            # Apply rotation
            r_rot = self.rotation_matrix @ r_sim
            v_rot = self.rotation_matrix @ v_sim

            for i,xi in enumerate(["x","y","z"]):
                dat[f"{xi}"] = r_rot[i]
                dat[f"v{xi}"] = v_rot[i]

            # Calculate "tilt" of total angular momentum
            Lz_tot = np.sum(dat["x"]*dat["vy"] - dat["y"]*dat["vx"])
            L = np.cross(np.vstack([dat["x"],dat["y"],dat["z"]]).T, 
                         np.vstack([dat["vx"], dat["vy"], dat["vz"]]).T)
            L_tot = np.sqrt(np.sum(np.sum(L,axis=0)**2))
            L_tilts.append(Lz_tot / L_tot)

        centre_pos = np.vstack(centre_pos)
        
        # make plot
        fig,axs = plt.subplots(1,3, 
                               figsize=(8,3),
                               layout="constrained")

        axs[0].plot(times, L_tilts)
        axs[0].set_xlabel("Time [Gyr]")
        axs[0].set_ylabel("Lz/L")
        axs[0].set_xlim([math.floor(min(times)),
                         math.ceil(max(times))])
        axs[0].set_ylim([min(L_tilts),max(L_tilts)])
        axs[0].set_aspect((math.ceil(max(times))-math.floor(min(times)))/
                          (max(L_tilts)-min(L_tilts)))

        lim = np.max(centre_pos**2)**0.5

        axs[1].scatter(centre_pos[:,0], centre_pos[:,1])
        axs[1].set_xlabel("x [kpc]")
        axs[1].set_ylabel("y [kpc]")
        axs[1].set_xlim([-lim,lim])
        axs[1].set_ylim([-lim,lim])
        axs[1].set_aspect(1)

        axs[2].scatter(centre_pos[:,0], centre_pos[:,2])
        axs[2].set_xlabel("x [kpc]")
        axs[2].set_ylabel("z [kpc]")
        axs[2].set_xlim([-lim,lim])
        axs[2].set_ylim([-lim,lim])
        axs[2].set_aspect(1)

        return fig

    
    def __get_subhalo_property__(self,
                                 SubhaloID: int,
                                 prop: str,
                                 tree):
        
        # start with last position
        prop_list = [tree[prop][tree["SubhaloID"][...]==SubhaloID][0]]
        snap = [tree["SnapNum"][tree["SubhaloID"][...]==SubhaloID][0]]

        FirstProgID = tree["FirstProgenitorID"][tree["SubhaloID"][...]==SubhaloID]

        while FirstProgID!=-1:
            prop_list.append(tree[prop][tree["SubhaloID"][...]==FirstProgID][0])
            snap.append(tree["SnapNum"][tree["SubhaloID"][...]==FirstProgID][0])
            FirstProgID = tree["FirstProgenitorID"][tree["SubhaloID"][...]==FirstProgID]

        return dict(zip(snap,prop_list))
    
    def __get_SubhaloID_progenitors__(self,
                                      SubhaloID: int,
                                      tree):
        
        progenitors = []
        FirstProgID = tree["FirstProgenitorID"][tree["SubhaloID"][...]==SubhaloID]

        while FirstProgID!=-1:
            progenitors.append(FirstProgID)
            FirstProgID = tree["FirstProgenitorID"][tree["SubhaloID"][...]==FirstProgID]

        return progenitors
    

    def __get_cosmology__(self):

        # Define cosmology of the simulation
        f = h5py.File(f"{self.__snap_path__}box_{self.__box__}/snap_090.hdf5")

        Om_0 = f["Header"].attrs["Omega0"]
        H0 = f["Header"].attrs["HubbleParam"]*100 * u.km / u.s / u.Mpc

        cosmo = FlatLambdaCDM(H0=H0, Om0=Om_0)

        return cosmo
    
    def __get_r90__(self, snap: int):

        # Define r90 as the radius that encloses 90% of the total DM mass
        dat = self.__load_part_data__(snap=snap, PartType=1)

        # Sort particles based on distance from the centre
        order = np.argsort(dat["r"])
        r_sorted = dat["r"][order]
        m_sorted = dat["mass"][order]

        # Get cumulative distribution of mass
        mass_cdf = np.cumsum(m_sorted) / np.sum(m_sorted)

        r90 = r_sorted[np.argmin((mass_cdf-0.9)**2)]

        return r90
    
    def __fit_nfw__(self, snap: int):

        # Load particles from z=0 snapshot
        dat = self.__load_part_data__(snap=snap, PartType=1)

        # Calculate critical density of the universe at snap
        f = h5py.File(f"{self.__snap_path__}box_{self.__box__}/snap_{snap:03}.hdf5")
        z = f["Header"].attrs["Redshift"]
        rho_crit = self.cosmo.critical_density(z).to(u.Msun/u.kpc**3).value

        # Calculate density profile of the DM halo
        rbins, dvals = DREAMS_utils.return_density(r=dat["r"],
                                                   weights=dat["mass"], 
                                                   bins=100,
                                                   rangevals=[1,300])
        # Fit NFW profile to the density profile
        popt, pcov = curve_fit(lambda r, c, R_vir : DREAMS_utils.NFW_profile(r, c, R_vir, rho_crit),
                       rbins, dvals,
                       bounds=([1e-5, 1e-5], [1000, 1000]),
                       )
        
        r_scale = popt[1]/popt[0]
        r_vir = popt[1]
        M_vir = np.sum(dat["mass"][dat["r"]<r_vir])

        return r_scale, r_vir, M_vir
    
    def __fit_scale_radius__(self, dat):
        
        R = dat["rxy"]
        Rbins, dvals_R = DREAMS_utils.return_density(r=R,
                                                     weights=dat["mass"],
                                                     rangevals=[0.5,max(R)],
                                                     bins=200,
                                                     log=False,
                                                     smooth=True)

        R_s = DREAMS_utils.get_scale_factor(Rbins, dvals_R, dvals_R[0])
        
        return R_s
    
    def __fit_scale_height__(self, dat):
        
        z = np.abs(dat["z"])
        zbins, dvals_z = DREAMS_utils.return_density(r=z,
                                                     weights=dat["mass"],
                                                     rangevals=[0,z.max()],
                                                     bins=200,
                                                     log=False,
                                                     smooth=True)

        z_s = DREAMS_utils.get_scale_factor(zbins, dvals_z, dvals_z[0])
        
        return z_s
        


    def track_particles(self,
                        particleIDs: np.array,
                        PartType: int,
                        snapshots: list[int]):
        
        # Save position and velocity of the particles at different snapshots
        out = {}
        # List to put IDs of particles not found in any snapshot
        flagged_particles = []
        
        # Load all the particles bound to the MW at different snapshots
        particles_list = [self.__load_part_data__(snap=snap,
                                                  PartType=PartType) for snap in snapshots]
        
        for pid in particleIDs:    
            
            xyz_list, v_xyz_list, E_list = [], [], []
        
            for snap in snapshots:
                
                # Load particles bound to the MW halo at snapshot
                particles = particles_list[snapshots.index(snap)]
                
                #Check if particle is in the snapshot
                idx = np.isin(particles["iord"],pid)
                if np.sum(idx)==0:
                    # Particle not found in this snapshot
                    flagged_particles.append(pid)
                    continue
            
                xyz = np.array([particles[idx][f] for f in ["x", "y","z"]])*u.kpc
                v_xyz = np.array([particles[idx][f] for f in ["vx", "vy","vz"]])*(u.km/u.s)
                
                # Calculate the total energy of the particle
                E = particles[idx]["phi"] + 0.5*np.sum(v_xyz**2)
                
                xyz_list.append(xyz)
                v_xyz_list.append(v_xyz)
                E_list.append(E)
                
                
            out[pid] = {"xyz": np.hstack(xyz_list),
                        "v_xyz": np.hstack(v_xyz_list),
                        "E": np.hstack(E_list)
                        }
            
        # Remove flagged particles
        for pid in flagged_particles:
            _ = out.pop(pid, None)
            
        return out

    
    def plot_subhalos_tracks(self):

        # Load merger tree
        tree = h5py.File(f"{self.__group_path__}/box_{self.__box__}/tree_extended.hdf5")

        # Initialise plot
        fig,axs = plt.subplots(1,2, layout="constrained")
        axs[0].set_xlabel("x [ckpc/h]")
        axs[0].set_ylabel("y [ckpc/h]")
        axs[1].set_xlabel("x [ckpc/h]")
        axs[1].set_ylabel("z [ckpc/h]")
        for ax in axs:
            ax.set_xlim([-2000,2000])
            ax.set_ylim([-2000,2000])
            ax.set_aspect(1)

        # Get position of MW at different snaps
        MW_pos_dict = self.__get_subhalo_property__(SubhaloID=0, prop="SubhaloPos", tree=tree)
        MW_R200_dict = self.__get_subhalo_property__(SubhaloID=0, prop="Group_R_Crit200", tree=tree)

        max_snap = max(tree["SnapNum"][...])
        excluded_subhalos = self.__get_SubhaloID_progenitors__(SubhaloID=0, tree=tree)

    
        # Identify all subhalos that meet a certain mass threshold
        SubhaloID_list = tree["SubhaloID"][(tree["SnapNum"][...]==max_snap) &
                                            (tree["SubhaloMass"][...]>0.1)] 
        
        # Check that the subhalos have not already been plotted
        SubhaloID_list = [SubhaloID for SubhaloID in SubhaloID_list 
                            if SubhaloID not in excluded_subhalos]
        
        
        for SubhaloID in SubhaloID_list:
            # Get positions over time of all the identified subhalos
            positions_dict = self.__get_subhalo_property__(SubhaloID=SubhaloID,
                                                           prop="SubhaloPos",
                                                           tree=tree)
            
            # Transform positions w.r.t. position of the MW
            positions = []
            for snap,subhalo_pos in positions_dict.items():
                try:
                    pos = subhalo_pos-MW_pos_dict[snap]
                    positions.append(pos)
                except KeyError:
                    continue
                if snap==90:
                    axs[0].scatter(pos[0], pos[1], c="k", s=1)
                    axs[1].scatter(pos[0], pos[2], c="k", s=1)

            # Plot subhalo track
            if len(positions)==0:
                continue
            positions = np.vstack(positions)
            axs[0].plot(positions[:,0], positions[:,1], lw=1)
            axs[1].plot(positions[:,0], positions[:,2], lw=1)

            # Save IDs of the subhalo at different snapshots to avoid replotting
            IDs_sub = self.__get_SubhaloID_progenitors__(SubhaloID=SubhaloID, tree=tree)
            excluded_subhalos += IDs_sub

        # Plot extent of the virial radius over time
        for snap, r200 in MW_R200_dict.items():
            for ax in axs:
                # Create a Circle patch
                circle = patches.Circle(
                    (0,0), # center coordinates
                    r200,  # Radius
                    color="k",
                    fill=True, 
                    alpha=0.2,
                    linewidth=0.1,
                )
                ax.add_patch(circle)
                


        return fig
class DREAMSMW_high_cadence(DREAMSMW):

    def __init__(self, 
                 snap_path: str):
        
        self.__snap_path__ = snap_path
        # Get list of snapshot files from earliest to latest
        self.snapshot_files = self.__get_snapshot_files__()
        
        # Set coordinates frame of reference
        dat = self.__load_part_data__(snap=self.snapshot_files[-1],
                                      PartType=4,
                                      rotate=False)
        _, self.rotation_matrix = DREAMS_utils.rotate_galaxy(dat=dat) 
        
        # Define cosmology of the simulation
        f = h5py.File(f"{snap_path}{self.snapshot_files[-1]}")
        Om_0 = f["Header"].attrs["Omega0"]
        H0 = f["Header"].attrs["HubbleParam"]*100 
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=Om_0)
        
        # Get Fit NFW profile to galaxy
        self.r_scale, self.r_vir, self.M_vir = self.__fit_nfw__(snap=self.snapshot_files[-1])
        
        dat = self.__load_part_data__(snap=self.snapshot_files[-1], PartType=4)
        disc_star_ids = self.select_disc_stars(dat, 
                                               k_threshold=0.7, 
                                               r_max=30,
                                               z_max=10)
        disc_dat = dat[np.isin(dat["iord"], disc_star_ids)]
        self.r_scale_disc = self.__fit_scale_radius__(disc_dat)
        self.z_scale_disc = self.__fit_scale_height__(disc_dat)
        
        print(f"""High-cadence snpashots Galaxy:\r
               Scale Radius (DM): \t\t{self.r_scale:.1f} kpc\r
               Virial Radius: \t\t{self.r_vir:.1f} kpc\r
               Virial Mass: \t\t{self.M_vir:.1g} M_sun\r
               Disc Scale Radius: \t\t{self.r_scale_disc:.1f} kpc\r
               Disc Scale Height: \t\t{self.z_scale_disc:.1f} kpc
               """)
        
    def __get_snapshot_files__(self):
        
        # Get all the .hdf5 files in the directory
        outputs = os.listdir(snap_path)
        outputs = [out for out in outputs if ".hdf5" in out]
        
        # Order files by snapshot
        n_outputs = sorted([int(out.replace("snap_","").replace(".hdf5","")) for out in outputs])
        ordered_outputs = [f"snap_{n}.hdf5" for n in n_outputs]
        
        return ordered_outputs
    
    # Overwrite the function to load the data from the simulation
    def __load_part_data__(self, 
                           snap, 
                           PartType, 
                           rotate = True):
        
        # Load particle data
        f = h5py.File(f"{snap_path}{snap}")
        
        
        # Create pynbody simulation object
        N = int(f["Header"].attrs["NumPart_Total"][PartType])
        if PartType==0:
            dat = pynbody.new(gas=N)
        elif PartType==1:
            dat = pynbody.new(dark=N)
        elif PartType==4:
            dat = pynbody.new(star=N)
        else:
            raise ValueError("Only gas (0), dark matter (1), ad star (4) PartTypes are supported")
        
        
        # Define simulation units
        a = float(f["Header"].attrs["Time"])
        pynbody.config['omegaM0'] = float(f["Header"].attrs["Omega0"])
        pynbody.config['omegaL0'] = float(f["Header"].attrs["OmegaLambda"])
        pynbody.config['h'] = float(f["Header"].attrs["HubbleParam"]) #should be .6909, but file gives 69.09
        pynbody.config['omegaB0'] = float(f["Header"].attrs["OmegaBaryon"])
        pynbody.config['a'] = a
        pynbody.units.a = a
        pynbody.units.h = float(f["Header"].attrs["HubbleParam"])
        
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
            'HighResGasMass': units.Unit(1)}

        
        # Create dictionary to map variable names from Illustris outputs to pynbody
        name_map = pynbody.snapshot.namemapper.AdaptiveNameMapper('gadgethdf-name-mapping',return_all_format_names=False)
        
        # Assign units to each field in the data object
        for key in f[f"PartType{PartType}"].keys():
            mapped_name = name_map(key, reverse=True)
            dat[mapped_name] = f[f"PartType{PartType}/{key}"]
            dat[mapped_name].units = unit_dict[key]
        
        # Convert to physical units
        dat.physical_units()
        
        # Center the halo using the shrink sphere center method
        pynbody.analysis.center(dat, 
                                mode='ssc',
                                with_velocity=True, # Correct for the motion of the center of mass of the halo
                                cen_size="5 kpc")
        
        if rotate:
            # Rotate to align total angular momentum vector to z-axis
            r_sim = np.stack([dat["x"], dat["y"], dat["z"]])
            v_sim = np.stack([dat["vx"], dat["vy"], dat["vz"]])
            # Apply rotation
            r_rot = self.rotation_matrix @ r_sim
            v_rot = self.rotation_matrix @ v_sim

            for i,xi in enumerate(["x","y","z"]):
                dat[f"{xi}"] = r_rot[i]
                dat[f"v{xi}"] = v_rot[i]
        
        return dat
    
    def __fit_nfw__(self, snap: str):

        # Load particles 
        dat = self.__load_part_data__(snap=snap, PartType=1)

        # Calculate density profile of the DM halo
        rbins, dvals = DREAMS_utils.return_density(r=dat["r"],
                                                   weights=dat["mass"], 
                                                   bins=100,
                                                   rangevals=[1,300],
                                                   smooth=True)
        
        # Calculate critical density of the universe at snap
        f = h5py.File(f"{self.__snap_path__}{self.snapshot_files[-1]}")
        z = f["Header"].attrs["Redshift"]
        rho_crit = self.cosmo.critical_density(z).to(u.Msun/u.kpc**3).value
        
        # Fit NFW profile to the density profile
        popt, pcov = curve_fit(lambda rbins, c, r_vir: self.__NFW_profile__(rbins, rho_crit, c, r_vir),
                       rbins, dvals/max(dvals),
                       bounds=([1e-5, 1e-5], [100, 400]),
                       maxfev=1000
                       )
        
        r_scale = popt[1]/popt[0]
        r_vir = popt[1]
        M_vir = np.sum(dat["mass"][dat["r"]<r_vir])

        return r_scale, r_vir, M_vir
    

    def __NFW_profile__(self, r, rho_crit, c, r_vir):
        
        delta = 200/3 * c**3 / (np.log(1+c) - c/(1+c))
        
        x = r/(r_vir/c)
        
        d = rho_crit*delta / (x * (1+x)**2)

        return  d/max(d)
  
class EXPBFE_builder():

    def __init__(self, 
                 sim,
                 basis_params_dict: dict,
                 density_dict: dict,
                 snapshots: list, # can be int or str depending if it isn't or is highcadence
                 output_dir: str,
                 high_cadence: bool=False):
        
        self.sim = sim
        self.__output_dir__ = output_dir
        self.snapshots = snapshots
        
        # Define units of the simulation
        self.exp_units = SimulationUnitSystem(mass=self.sim.M_vir*u.Msun, 
                                              length=self.sim.r_vir*u.kpc, 
                                              G=1)
        
        # Define the name of the output files
        self.model_files_dict = {} # Density tables
        self.basis_files_dict = {} # Basis functions
        self.coefs_files_dict = {} # Coefficients
        
        for PartType in basis_params_dict.keys():
            if not high_cadence:
                self.model_files_dict[PartType] = f"{self.__output_dir__}basis_empirical_PartType{PartType}_box_{self.sim.__box__:04}.txt" 
                self.basis_files_dict[PartType] = f"{self.__output_dir__}basis_yaml_PartType{PartType}_box_{self.sim.__box__:04}.yml"
                self.coefs_files_dict[PartType] = f"{self.__output_dir__}coefs_PartType{PartType}_box_{self.sim.__box__:04}.h5"
                
            
            else:
                self.model_files_dict[PartType] = f"{self.__output_dir__}basis_empirical_PartType{PartType}_highcadence.txt" 
                self.basis_files_dict[PartType] = f"{self.__output_dir__}basis_yaml_PartType{PartType}_highcadence.yml"
                self.coefs_files_dict[PartType] = f"{self.__output_dir__}coefs_PartType{PartType}_highcadence.h5"
                
        
        # Build basis
        print("Building basis for the expansion...", flush=True)
        self.basis = {}
        for PartType, basis_params in basis_params_dict.items():
            basis = self.__build_basis__(PartType=PartType,
                                         basis_params=basis_params,
                                         density_params= density_dict[PartType])
            self.basis[PartType] = basis
            
        # Calculate the coefficients 
        print(f"Calculating the coefficients at snapshots: {snapshots}", flush=True)
        self.coefs = {}
        for PartType, basis in self.basis.items():
            self.coefs[PartType] = self.__get_coefs__(basis=basis,
                                                      snapshots=snapshots,
                                                      PartType=PartType)
            
    
    def __build_basis__(self, 
                        PartType: int,
                        basis_params: dict = {},
                        density_params: dict = {}):
        

        # Load particles from z=0 snapshot
        dat = self.sim.__load_part_data__(snap=self.snapshots[-1], PartType=PartType)
        
        # Define the basis parameters
        if PartType==1:
            yaml_file = self.__get_DM_basis_config__(dat, basis_params, density_params)
        elif PartType==4:
            yaml_file = self.__get_star_basis_config__(dat, basis_params, density_params)
        
        # Load the basis config in the yaml file with the basis parameters
        with open(yaml_file, "r") as f:
            yaml_config = f.read()

        # Build the basis
        basis = pyEXP.basis.Basis.factory(yaml_config)      

        return basis  

     

    def __get_DM_basis_config__(self, 
                                dat, 
                                basis_params = {},
                                density_params = {}):

        # Default values for calculating the density profile
        density_params_df = {"bins": 400,
                             "rangevals": [0,2.5]}
        
        density_params_df.update(density_params)

        rbins, dvals = DREAMS_utils.return_density(r=dat["r"],
                                                   weights=dat["mass"], 
                                                   smooth=True,
                                                   **density_params_df)
        
        # Scale values to virial quantities
        rbins /= self.sim.r_vir
        dvals /= (self.sim.M_vir / (self.sim.r_vir**3))
        

        # Create an EXP-compatible spherical basis function table 
        model_file = self.model_files_dict[1]  
        cache_file = model_file.replace(".txt",".cache.run0")

        # Check if model or table have already been computed
        if os.path.exists(model_file):
            os.remove(model_file)
        if os.path.exists(cache_file):
            os.remove(cache_file)

        rbins, dvals, mass, potential = DREAMS_utils.makemodel_empirical(rvals=rbins,
                                                                         dvals=dvals,
                                                                         pfile=model_file) 
        config = {"id" : "sphereSL",
                  "parameters": {"numr": 4000,
                                 "rmin": float(1/self.sim.r_vir),
                                 "rmax": 1,
                                 "Lmax": 5,
                                 "nmax": 10,
                                 "rmapping": float(self.sim.r_scale/self.sim.r_vir),
                                 "modelname": model_file,
                                 "cachename": cache_file,
                                 "pcavar": True,
                                 "subsamp": 1
                                 },
                  "runtag": "run0"
                 }
        
        # Update config parameters with the one specified at the class initialization
        config["parameters"].update(basis_params)
        print(config)
        # Save yaml file for constructing gala potential
        yaml_file = self.basis_files_dict[1]

        with open(yaml_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return yaml_file
    
    def __get_star_basis_config__(self, 
                                  dat,
                                  basis_params = {},
                                  density_params = {}):

        cache_file = self.model_files_dict[4].replace(".txt",".cache.run0")
        
        config = {"id" : "cylinder",
                  "parameters": {"acyl": float(self.sim.r_scale_disc/self.sim.r_vir),       # The scale length of the exponential disk
                                 "hcyl": float(self.sim.z_scale_disc/self.sim.r_vir),      # The scale height of the exponential disk
                                 "lmaxfid": 32,      # The maximum spherical harmonic order for the input basis
                                 "nmaxfid": 32,      # The radial order for the input spherical basis
                                 "mmax": 6,          # The maximum azimuthal order for the cylindrical basis
                                 "nmax": 12,         # The maximum radial order of the cylindrical basis
                                 "ncylnx": 256,      # The number of grid points in mapped cylindrical radius
                                 "ncylny": 128,      # The number of grid points in mapped verical scale
                                 "ncylodd": 3,       # The number of anti-symmetric radial basis functions per azimuthal order m
                                 "rnum": 1000,       # The number of radial integration knots in the inner product
                                 "pnum": 0,          # The number of azimuthal integration knots (pnum: 0, assume axisymmetric target density)
                                 "tnum": 80,         # The number of colatitute integration knots
                                 "ashift": 0.5,      # Target shift length in scale lengths to create more variance
                                 "vflag": 16,        # Verbosity flag: print diagnostics to stdout for vflag>0
                                 "logr": False,      # Log scaling in cylindrical radius
                                 "cachename": cache_file,  # The cache file name
                                }   
                 }
        
        # Optional arguments:
        # "dtype": "python",  # Use user-supplied python module
        # "pyname": "pyDens", # The module name
        
        # Update config parameters with the one specified at the class initialization
        config["parameters"].update(basis_params)
        print(config)
        # Save yaml file for constructing gala potential
        yaml_file = self.basis_files_dict[4]

        with open(yaml_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return yaml_file
        

    

    def __get_coefs__(self,
                      basis,
                      snapshots: list[int],
                      PartType: int):    
        
        # Important for the covariance file
        os.chdir(self.__output_dir__)
        coefs_container = None

        for snap in snapshots:  

            # Load particles from z=0 snapshot
            dat = self.sim.__load_part_data__(snap=snap, PartType=PartType)   

            # Scale to virial units
            mass = np.array(dat["mass"]) / self.sim.M_vir
            pos = np.vstack([dat["x"], dat["y"], dat["z"]]).T / self.sim.r_vir


            # Read age of the universe at snapshot
            try:
                f = h5py.File(f"{self.sim.__snap_path__}box_{self.sim.__box__}/snap_{snap:03}.hdf5")
            except AttributeError:
                f = h5py.File(f"{self.sim.__snap_path__}{snap}")
                
            z = f["Header"].attrs["Redshift"]
            t = self.sim.cosmo.age(z).value                                                           
            
            # Calculate the coefficients of the BFE 
            coefs = basis.createFromArray(mass, 
                                          pos, 
                                          time=t)
            # Compute the covariance matrix for the coefficients
            if PartType==1:
                basis.writeCoefCovariance("sphereSL", "run0", t)
            elif PartType==4:
                basis.writeCoefCovariance("cylinder", "run0", t)
            
            if coefs_container is None:
                coefs_container = pyEXP.coefs.Coefs.makecoefs(coefs)
                coefs_container.add(coefs)
            else:
                coefs_container.add(coefs)

        # Save the coefficients
        coefs_file = self.coefs_files_dict[PartType]
            
        if os.path.exists(coefs_file):
            os.remove(coefs_file)
            coefs_container.WriteH5Coefs(coefs_file) 
        else:
            coefs_container.WriteH5Coefs(coefs_file) 
        
        return coefs_container
    
    def __get_CoefCovariance__(self, 
                               PartType: int):
        
        # Read-in the covariance matrix of the coefficients
        if PartType==1:
            covar = pyEXP.basis.CovarianceReader(f'{self.__output_dir__}/coefcovar.sphereSL.run0.h5')
        elif PartType==4:
            covar = pyEXP.basis.CovarianceReader(f'{self.__output_dir__}/coefcovar.cylinder.run0.h5')
        
        return covar
    

    def __shell_average__(self,
                          basis, 
                          coefs,
                          r_min: float, 
                          field: str = None, 
                          n_points=1000):
        
        # Load basis coefficients
        basis.set_coefs(coefs)

        theta = np.arccos(np.random.uniform(-1, 1, size=n_points))
        phi = np.random.uniform(0, 2*np.pi, size=n_points)
        r = r_min

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        sph_avg = np.mean([basis.getFields(xi, yi, zi) for xi, yi, zi in zip(x, y, z)], axis=0)

        output_dict = dict(zip(basis.getFieldLabels(), sph_avg))

        if field is None:
            return output_dict
        else:
            return output_dict[field]
        

    def plot_density_profile(self):
        
        dat = self.sim.__load_part_data__(snap=90, PartType=1)
        rbins, dvals = DREAMS_utils.return_density(r=dat["r"],
                                                   weights=dat["mass"], 
                                                   bins=100,
                                                   rangevals=[1,300]) 
        
        # Get density contribution from different m values
        exp_dens, exp_densm0, exp_densml0 = [],[],[]
        rbins_exp = np.zeros(len(rbins)+1)
        rbins_exp[1:] = rbins
        for i in range(len(rbins_exp)-1):
            out_dict = self.__shell_average__(field="dens", 
                                              basis=self.basis[1],
                                              coefs=self.coefs[1],
                                              r_min=rbins_exp[i])
            exp_dens.append(out_dict["dens"])
            exp_densm0.append(out_dict["dens m=0"])
            exp_densml0.append(out_dict["dens m>0"])
        exp_dens = np.array(exp_dens)
        
        fig,ax = plt.subplots(figsize=(4,4))
        ax.plot(rbins, dvals, label="Data")
        ax.plot(rbins, exp_dens, label="EXP")
        ax.plot(rbins, exp_densm0, label="EXP m=0")
        ax.plot(rbins, exp_densml0, label="EXP m>0")
        ax.set_yscale("log")
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel("Density [$\\rm{M}_{\\odot}/\\rm{kpc}^3$]")
        ax.set_xscale("log")
        ax.legend()
        
        return fig,ax

        

    def plot_2D_integrated_field(self, 
                                 basis,
                                 coefs,
                                 field: str,
                                 x_bins: np.array,
                                 y_bins: np.array,
                                 z_bins: np.array,
                                 ax: plt.axes,
                                 integrate_over: str = "z",
                                 **kwargs
                                 ):
        
        basis.set_coefs(coefs)

        # Understand which field to plot
        field_labels = basis.getFieldLabels()
        idx = field_labels.index(field)


        # Understand which dimension to integrate on
        if integrate_over=="z":
            # Create array to hold the 2D field projection
            exp_2d = np.zeros((len(x_bins),len(y_bins)))
            dz = z_bins[1] - z_bins[0]
            extent = [x_bins[0], x_bins[1], y_bins[0], y_bins[1]]
            for i,xi in enumerate(x_bins):
                for j, yj in enumerate(y_bins):
                    field_sum = 0
                    for zi in z_bins:
                        outs = basis.getFields(xi, yj, zi)
                        field_sum += outs[idx]*dz
                    exp_2d[i,j] = field_sum

        elif integrate_over=="y":
            exp_2d = np.zeros((len(x_bins),len(z_bins)))
            dy = y_bins[1] - y_bins[0]
            extent = [x_bins[0], x_bins[1], z_bins[0], z_bins[1]]
            for i,xi in enumerate(x_bins):
                for j, zj in enumerate(z_bins):
                    field_sum = 0
                    for yi in y_bins:
                        outs = basis.getFields(xi, yi, zj)
                        field_sum += outs[idx]*dy
                    exp_2d[i,j] = field_sum

        elif integrate_over=="x":
            exp_2d = np.zeros((len(y_bins),len(z_bins)))
            dx = x_bins[1] - x_bins[0]
            extent = [y_bins[0], y_bins[1], z_bins[0], z_bins[1]]
            for i,yi in enumerate(y_bins):
                for j, zj in enumerate(z_bins):
                    field_sum = 0
                    for xi in x_bins:
                        outs = basis.getFields(xi, yi, zj)
                        field_sum += outs[idx]*dx
                    exp_2d[i,j] = field_sum
        

        # Plot the 2D projection
        plot = ax.imshow(exp_2d, origin="lower",
                         extent=extent, **kwargs)

        return plot



    def build_gala_potential(self, **kwargs):

        
        pot = gp.CCompositePotential()
        for PartType in self.basis.keys():
            # Read basis and coefficients of EXP approximation
            pot[PartType] = gp.EXPPotential(units=self.exp_units,
                                            config_file=self.basis_files_dict[PartType],
                                            coef_file=self.coefs_files_dict[PartType],
                                            snapshot_time_unit=u.Gyr, **kwargs)

        return pot, self.exp_units
    
    def surface_projection(self,
                           basis,
                           coefs,
                           field: str, # dens, dens m=0, dens m>0, potl, potl m-0, ...
                           time: float,
                           extent: list, # e.g. [[xmin, ymin, 0.],[xmax, ymax, 0.]]
                           grid: list, # [bins_x, bins_y, 0.]
                           ):
        
        # Initialise surface field generator
        times = coefs.Times()
    
        generator = pyEXP.field.FieldGenerator(times, 
                                               [el.to(self.exp_units["length"]).value for el in extent[0] if el!=0], 
                                               [el.to(self.exp_units["length"]).value for el in extent[1] if el!=0], 
                                               grid)
        
        surfaces = generator.slices(basis, coefs)

        surface = surfaces[time][field]

        non_zero_entries = [i for i in range(3) if grid[i]!=0]

        x = np.linspace(extent[0][non_zero_entries[0]],
                        extent[1][non_zero_entries[0]],
                        grid[non_zero_entries[0]])
        
        y = np.linspace(extent[0][non_zero_entries[1]],
                        extent[1][non_zero_entries[1]],
                        grid[non_zero_entries[1]])
        
        xv, yv = np.meshgrid(x, y)
        
        fig, ax = plt.subplots()
        ax.set_xlim([min(x.value),max(x.value)])
        ax.set_ylim([min(y.value), max(y.value)])
        cbar_label = field

        if field in ["dens", "dens m=0", "dens m>0"]:

            # Convert to M_sun / kpc^2
            surface = surface*(self.exp_units["mass"]/self.exp_units["length"]**2).to(u.Msun / u.pc**2)
            surface = np.log10(surface)
                
            cbar_label = "$\\log_{10}(\\Sigma) \\; [\\rm{M}_{\\odot} \\, \\rm{pc}^{-2}]$"

                
        if field in ["potl", "potl m=0", "potl m>0"]:
            # Convert to (km/s)^2
            surface = surface*(self.exp_units["length"]**2 / self.exp_units["time"]**2).to(u.km**2 / u.s**2)
            cbar_label = "$\\Phi \\; [\\rm{km}^2 \\, \\rm{s}^{-2}]$"

        
        cont1 = ax.contour(xv, yv, surface, colors='k')
        cont1.clabel(fontsize=9, inline=True)
        cont2 = ax.contourf(xv, yv, surface)
        cbar = fig.colorbar(cont2)
        cbar.set_label(cbar_label)
        
        # Plot circles showing 10 and 100 kpc
        for r in [10, 100]:
            circle = patches.Circle(
                        (0,0), # center coordinates
                        r,  # Radius
                        color="red",
                        fill=False, 
                        linewidth=2,
                        linestyle='-'
                    )
            ax.add_patch(circle)
            
        

        return fig

    def volume_render(self,
                      basis,
                      coefs,
                      time: float, 
                      field: str,
                      grid_lim: int = 100,
                      n_points: int = 100
                      ):

        # Calculate 3D distribution of fields
        times = coefs.Times()
        pmin  = [-grid_lim, -grid_lim, -grid_lim]
        pmax  = [grid_lim, grid_lim, grid_lim]
        grid  = [n_points, n_points, n_points]

        generator = pyEXP.field.FieldGenerator(times, pmin, pmax, grid)
        volumes = generator.volumes(basis, coefs)

        volume = volumes[time][field]

        # Initialise plot
        plot = k3d.plot()

        value_range = [np.percentile(volume, 5), np.percentile(volume, 95)]
        size = [-grid_lim, grid_lim, -grid_lim, grid_lim, -grid_lim, grid_lim]

        volume = k3d.volume(volume.astype(np.float32), 
                          alpha_coef=250,
                          color_range=value_range,  
                          color_map=(np.array(k3d.colormaps.paraview_color_maps.Blues).reshape(-1,4) 
                          * np.array([1,1.0,1.0,1.0])).astype(np.float32), 
                          compression_level=7)
        
        volume.transform.bounds = [-size[0], size[0], -size[1], size[1], -size[2], size[2]]

        plot += volume

        return plot
  
    
########################################################################

def plot_acceleration_field_xy(pot,
                               r_vir,
                               t=13*u.Gyr,
                               grid_lim: float = 1.,
                               n_points: int = 100,
                               z_level: float = 0.0):
    
    x = np.linspace(-grid_lim, grid_lim, n_points)
    y = np.linspace(-grid_lim, grid_lim, n_points)

    Fx = np.zeros((n_points,n_points))
    Fy = np.zeros((n_points,n_points))

    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            pos = [xi,yj,z_level]*u.kpc
            acc = pot.acceleration(pos, t=t).to(u.km/u.s/u.Gyr).value
            Fx[i,j] = acc[0][0]
            Fy[i,j] = acc[1][0]

    fig,axs = plt.subplots(1,2, sharex=True, sharey=True, layout="constrained")

    axs[0].pcolormesh(x,y, Fx, 
                    norm=mpl.colors.Normalize(vmin=-1000, vmax=1000),
                    cmap="seismic")
    plot = axs[1].pcolormesh(x,y, Fy, 
                    norm=mpl.colors.Normalize(vmin=-1000, vmax=1000),
                    cmap="seismic")

    cbar = plt.colorbar(plot, ax=axs[:], orientation="horizontal", shrink=0.8)
    cbar.set_label("Acceleration $[\\rm{km} \, \\rm{s}^{-1} \, \\rm{Gyr}^{-1}]$")
    axs[0].set_xlabel("x [kpc]")
    axs[1].set_xlabel("x [kpc]")
    axs[0].set_ylabel("y [kpc]")
    axs[0].set_aspect(1)
    axs[1].set_aspect(1)
    fig.suptitle(f"z = {z_level} [kpc]")

    axs[0].set_title("$a_{X}$")
    axs[1].set_title("$a_{Y}$")
    
    
    for ax in axs:
        # Plot circles showing 10 and 100 kpc
        r_10kpc = 10 / r_vir
        r_100kpc = 100 / r_vir
        for r in [r_10kpc, r_100kpc]:
            circle = patches.Circle(
                        (0,0), # center coordinates
                        r,  # Radius
                        color="red",
                        fill=False, 
                        linewidth=2,
                        linestyle='-'
                    )
            ax.add_patch(circle)
    
    
    return fig,axs


def plot_acceleration_field_xz(pot,
                               r_vir,
                               t=13*u.Gyr,
                               grid_lim: float = 1.,
                               n_points: int = 100,
                               y_level: float = 0.0):
    

    x = np.linspace(-grid_lim, grid_lim, n_points)
    z = np.linspace(-grid_lim, grid_lim, n_points)

    Fx = np.zeros((n_points,n_points))
    Fz = np.zeros((n_points,n_points))

    for i, xi in enumerate(x):
        for j, zj in enumerate(z):
            pos = [xi,y_level,zj]*u.kpc
            acc = pot.acceleration(pos, t=t).to(u.km/u.s/u.Gyr).value
            Fx[i,j] = acc[0][0]
            Fz[i,j] = acc[1][0]

    fig,axs = plt.subplots(1,2, sharex=True, sharey=True, layout="constrained")

    axs[0].pcolormesh(x,y_level, Fx, 
                    norm=mpl.colors.Normalize(vmin=-1000, vmax=1000),
                    cmap="seismic")
    plot = axs[1].pcolormesh(x,y_level, Fz, 
                    norm=mpl.colors.Normalize(vmin=-1000, vmax=1000),
                    cmap="seismic")

    cbar = plt.colorbar(plot, ax=axs[:], orientation="horizontal", shrink=0.8)
    cbar.set_label("Acceleration $[\\rm{km} \, \\rm{s}^{-1} \, \\rm{Gyr}^{-1}]$")
    axs[0].set_xlabel("x [kpc]")
    axs[1].set_xlabel("x [kpc]")
    axs[0].set_ylabel("z [kpc]")
    axs[0].set_aspect(1)
    axs[1].set_aspect(1)
    fig.suptitle(f"y = {y_level} [R_vir]")

    axs[0].set_title("$a_{X}$")
    axs[1].set_title("$a_{Z}$")
    
    for ax in axs:
        # Plot circles showing 10 and 100 kpc
        r_10kpc = 10 / r_vir
        r_100kpc = 100 / r_vir
        for r in [r_10kpc, r_100kpc]:
            circle = patches.Circle(
                        (0,0), # center coordinates
                        r,  # Radius
                        color="red",
                        fill=False, 
                        linewidth=2,
                        linestyle='-'
                    )
            ax.add_patch(circle)
    
    return fig,axs
                        
                            
    




    

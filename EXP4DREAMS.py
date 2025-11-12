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
        
        self.r_scale, self.r_vir, self.M_vir = self.__fit_nfw__(snap=90)


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
        rbins, dvals = DREAMS_utils.return_density(logr=np.log10(dat["r"]),
                                                   weights=dat["mass"], 
                                                   bins=100,
                                                   rangevals=[0,2.5])
        # Fit NFW profile to the density profile
        popt, pcov = curve_fit(lambda r, c, R_vir : DREAMS_utils.NFW_profile(r, c, R_vir, rho_crit),
                       rbins, dvals,
                       bounds=([1e-5, 1e-5], [1000, 1000]),
                       )
        
        r_scale = popt[1]/popt[0]
        r_vir = popt[1]
        M_vir = np.sum(dat["mass"][dat["r"]<r_vir])

        return r_scale, r_vir, M_vir


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


class EXPBFE_builder():

    def __init__(self, 
                 sim: DREAMSMW,
                 basis_params_dict: dict,
                 density_dict: dict,
                 snapshots: list[int],
                 output_dir: str):
        
        self.sim = sim
        self.__output_dir__ = output_dir

        # Define virial units
        self.r_scale, self.r_vir, self.M_vir = sim.__fit_nfw__(snap=90)
        
        # Define units of the simulation
        self.exp_units = SimulationUnitSystem(mass=self.M_vir*u.Msun, 
                                              length=self.r_vir*u.kpc, 
                                              G=1)

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
        dat = self.sim.__load_part_data__(snap=90, PartType=PartType)
        
        # Define the basis parameters
        if PartType==1:
            config = self.__get_DM_basis_config__(dat, basis_params, density_params)
        
        # Load the basis config in the yaml file with the basis parameters
        yaml_file = f"{self.__output_dir__}basis_yaml_PartType1_box_{self.sim.__box__:04}.yml"
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

        rbins, dvals = DREAMS_utils.return_density(logr=np.log10(dat["r"]),
                                                   weights=dat["mass"], 
                                                   **density_params_df)
        
        # Scale values to virial quantities
        rbins /= self.r_vir
        dvals /= (self.M_vir / (self.r_vir**3))
        
        # Smooth density values
        dvals = gaussian_filter1d(dvals, 4.)
        

        # Create an EXP-compatible spherical basis function table 
        model_file = f"{self.__output_dir__}basis_empirical_PartType1_box_{self.sim.__box__:04}.txt" 
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
                                 "rmin": float(1/self.r_vir),
                                 "rmax": 1,
                                 "Lmax": 2,
                                 "nmax": 10,
                                 "rmapping": 0.067,
                                 "modelname": model_file,
                                 "cachename": model_file.replace(".txt",".cache.run0")
                                 }
                 }
        
        # Update config parameters with the one specified at the class initialization
        config["parameters"].update(basis_params)
        print(config)
        # Save yaml file for constructing gala potential
        yaml_file = f"{self.__output_dir__}basis_yaml_PartType1_box_{self.sim.__box__:04}.yml"

        with open(yaml_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return config
    

    def __get_coefs__(self,
                      basis,
                      snapshots: list[int],
                      PartType: int):    

        coefs_container = None

        for snap in snapshots:  

            # Load particles from z=0 snapshot
            dat = self.sim.__load_part_data__(snap=snap, PartType=PartType)   

            # Scale to virial units
            mass = np.array(dat["mass"]) / self.M_vir
            pos = np.vstack([dat["x"], dat["y"], dat["z"]]).T / self.r_vir


            # Read age of the universe at snapshot
            f = h5py.File(f"{self.sim.__snap_path__}box_{self.sim.__box__}/snap_{snap:03}.hdf5")
            z = f["Header"].attrs["Redshift"]
            t = self.sim.cosmo.age(z).value                                                           
            
            # Calculate the coefficients of the BFE 
            coefs = basis.createFromArray(mass, 
                                          pos, 
                                          time=t)
            
            if coefs_container is None:
                coefs_container = pyEXP.coefs.Coefs.makecoefs(coefs)
                coefs_container.add(coefs)
            else:
                coefs_container.add(coefs)

        # Save the coefficients
        coefs_file = f"{self.__output_dir__}coefs_PartType{PartType}_box_{self.sim.__box__:04}.h5"
        if os.path.exists(coefs_file):
            os.remove(coefs_file)
            coefs_container.WriteH5Coefs(coefs_file) 
        else:
            coefs_container.WriteH5Coefs(coefs_file) 
        
        return coefs_container
    

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
        rbins, dvals = DREAMS_utils.return_density(logr=np.log10(dat["r"]),
                                                   weights=dat["mass"], 
                                                   bins=100,
                                                   rangevals=[0,2.5]) 
        
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
            coefs_file = f"{self.__output_dir__}coefs_PartType{PartType}_box_{self.sim.__box__:04}.h5"
            basis_yaml = f"{self.__output_dir__}basis_yaml_PartType{PartType}_box_{self.sim.__box__:04}.yml"

            pot[PartType] = gp.EXPPotential(units=self.exp_units,
                                            config_file=basis_yaml,
                                            coef_file=coefs_file,
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
                        
                            
    




    

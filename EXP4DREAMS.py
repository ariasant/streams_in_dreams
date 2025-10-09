import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import math
import numpy as np
import os
import pynbody
from nightmares.reader_funcs import load_zoom_particle_data_pynbody
import DREAMS_utils
import pyEXP
import yaml
import h5py
import k3d
from gala.units import SimulationUnitSystem
import gala.potential as gp
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


    def __load_part_data__(self,
                           snap: int, 
                           PartType: int,
                           rotate: bool = True,
                           ):
    

        # Get raw simulation data
        dat, grp_dat = load_zoom_particle_data_pynbody(self.__snap_path__, 
                                                       self.__group_path__, 
                                                       self.__box__, 
                                                       snap, 
                                                       PartType,
                                                       subhaloes=False)
    
        # Centre the galaxy, convert to physical units
        pynbody.analysis.center(dat, mode='ssc')
        dat.physical_units()

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
            dat, grp_dat = load_zoom_particle_data_pynbody(self.__snap_path__, 
                                                           self.__group_path__, 
                                                           self.__box__, 
                                                           snap, 
                                                           4, # stars
                                                           subhaloes=False)
            
            # Save position of centre in box coordinates
            #centre_pos.append(grp_dat["SubhaloPos"][0])
            centre_pos.append(pynbody.analysis.halo.shrink_sphere_center(dat))

            # Convert to physical units
            dat.physical_units()

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
                 snapshots: list[int],
                 output_dir: str):
        
        self.sim = sim
        self.__output_dir__ = output_dir

        # Build basis
        print("Building basis for the expansion...", flush=True)
        self.basis = {}
        for PartType, basis_params in basis_params_dict.items():
            basis = self.__build_basis__(PartType=PartType,
                                         basis_params=basis_params)
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
                        basis_params: dict = {}):
        

        # Load particles from z=0 snapshot
        dat = self.sim.__load_part_data__(snap=90, PartType=PartType)
        
        # Define the basis parameters
        if PartType==1:
            config = self.__get_DM_basis_config__(dat, basis_params)
        
        # Load the basis config in the yaml file with the basis parameters
        yaml_file = f"{self.__output_dir__}basis_yaml_PartType1_box_{self.sim.__box__:04}.yml"
        with open(yaml_file, "r") as f:
            yaml_config = f.read()

        # Build the basis
        basis = pyEXP.basis.Basis.factory(yaml_config)      

        return basis  

     

    def __get_DM_basis_config__(self, 
                                dat, 
                                basis_params = {}):

        # Calculate virial and scale radii for the halo
        # scale_r assumes: rho(r) = rho_MAX * e^(-r/scale_r) 
        # (rho is density)

        #r_vir = pynbody.analysis.halo.virial_radius(dat)

        # Exclude stars outside r90
        r90 = self.sim.__get_r90__(snap=90)
        dat = dat[dat["r"]<r90]

        log_r = np.log10(np.sqrt(dat["x"]**2 + dat["y"]**2 + dat["z"]**2))
        rbins, dvals = DREAMS_utils.return_density(logr=log_r,
                                                   weights=dat["mass"], 
                                                   bins=400,
                                                   rangevals=[0,np.log10(r90)])
        
        r_scale = rbins[np.argmin((dvals-dvals[0]/math.e)**2)]


        # Create an EXP-compatible spherical basis function table 
        model_file = f"{self.__output_dir__}basis_empirical_PartType1_box_{self.sim.__box__:04}.txt" 
        rbins, dvals, mass, potential = DREAMS_utils.makemodel_empirical(rvals=rbins,
                                                                         dvals=dvals,
                                                                         pfile=model_file) 
        config = {"id" : "sphereSL",
                  "parameters": {"numr": 4000,
                                 "rmin": float(np.round(rbins[0], decimals=2)),
                                 "rmax": float(np.round(rbins[-1], decimals=2)),
                                 "Lmax": 2,
                                 "nmax": 10,
                                 "rmapping": float(np.round(r_scale, decimals=2)),
                                 "self_consistent" : True,
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

            # Exclude stars outside r90
            r90 = self.sim.__get_r90__(snap=90)
            dat = dat[dat["r"]<r90]


            # Read age of the universe at snapshot
            f = h5py.File(f"{self.sim.__snap_path__}box_{self.sim.__box__}/snap_{snap:03}.hdf5")
            z = f["Header"].attrs["Redshift"]
            t = self.sim.cosmo.age(z).value                                                           
            
            # Calculate the coefficients of the BFE 
            coefs = basis.createFromArray(dat["mass"], 
                                          [dat["x"],dat["y"],dat["z"]], 
                                          time=t)
            
            if coefs_container is None:
                coefs_container = pyEXP.coefs.Coefs.makecoefs(coefs)
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



    def build_gala_potential(self, **kwargs):

        # Define units of the simulation
        exp_units = SimulationUnitSystem(mass=1*u.Msun, 
                                         length=1*u.kpc, 
                                         G=1)
        
        pot = gp.CCompositePotential()
        for PartType in self.basis.keys():
            # Read basis and coefficients of EXP approximation
            coefs_file = f"{self.__output_dir__}coefs_PartType{PartType}_box_{self.sim.__box__:04}.h5"
            basis_yaml = f"{self.__output_dir__}basis_yaml_PartType{PartType}_box_{self.sim.__box__:04}.yml"

            pot[PartType] = gp.EXPPotential(units=exp_units,
                                            config_file=basis_yaml,
                                            coef_file=coefs_file,
                                            snapshot_time_unit=u.Gyr, **kwargs)

        return pot, exp_units
    
    def surface_projection(self,
                           basis,
                           coefs,
                           field: str, # dens, dens m=0, dens m>0, potl, potl m-0, ...
                           time: float,
                           extent: list, # e.g. [[xmin, ymin, 0.],[xmax, ymax, 0.]]
                           grid: list # [bins_x, bins_y, 0.]
                           ):
        
        # Initialise surface field generator
        times = coefs.Times()
    
        generator = pyEXP.field.FieldGenerator(times, 
                                               extent[0], 
                                               extent[1], 
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
        cont1 = ax.contour(xv, yv, surface, colors='k')
        cont1.clabel(fontsize=9, inline=True)
        cont2 = ax.contourf(xv, yv, surface)
        cbar = fig.colorbar(cont2)
        cbar.set_label(field)

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
                          color_map=(np.array(k3d.colormaps.paraview_color_maps.Cool_to_Warm_Extended).reshape(-1,4) 
                          * np.array([1,1.0,1.0,1.0])).astype(np.float32), 
                          compression_level=7)
        
        volume.transform.bounds = [-size[0], size[0], -size[1], size[1], -size[2], size[2]]

        plot += volume

        return plot





        

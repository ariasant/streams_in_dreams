import sys
sys.path.append("/mnt/home/asante/ceph/streams_in_dreams")

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import math
import numpy as np
import os
import pynbody
import DREAMS_utils
import yaml
import h5py

from gala.units import SimulationUnitSystem
import gala.integrate as gi
import gala.potential as gp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyEXP

from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
import time




class DREAMSMW():

    def __init__(self, 
                 snap_path: str,
                 group_path: str):
        
        self.__snap_path__ = snap_path
        self.__group_path__ = group_path
        
        # Find snapshot number corresponding to z=0
        snapshots = get_snapshot_files(snap_path=self.__snap_path__)
        try:
            self.snap_z0 = int(snapshots[-1].split("_")[1])
        except ValueError:
            self.snap_z0 = int(snapshots[-1].split("_")[1].replace(".hdf5",""))

        ## Set coordinates frame of reference
        self.rotation_matrix = DREAMS_utils.get_rotation_matrix(snap_path=self.__snap_path__,
                                                                group_path=self.__group_path__,
                                                                snap_z0=self.snap_z0)
        
        ## Define the cosmology of the simulation
        self.cosmo = DREAMS_utils.get_cosmology(snap_path=self.__snap_path__)
        
        # Calculate characteristic scales of the galaxy
        self.r_scale, self.r_vir, self.M_vir = self.__fit_nfw__(snap=self.snap_z0)
        
        dat = self.__load_part_data__(snap=self.snap_z0, PartType=4)
        disc_star_idx = self.select_disc_stars(dat, 
                                               k_threshold=0.7, 
                                               r_max=30,
                                               z_max=10)
        disc_dat = dat[disc_star_idx]
        self.r_scale_disc = self.__fit_scale_radius__(disc_dat)
        self.z_scale_disc = self.__fit_scale_height__(disc_dat)
        
        print(f"""
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
                           r_cut: bool = True,
                           snap_path = None
                           ):
        
        if snap_path is None:
            snap_path = self.__snap_path__    

        # Get raw simulation data
        dat = DREAMS_utils.load_zoom_particle_data_pynbody(snap_path, 
                                                           self.__group_path__, 
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
                
        if r_cut:
            # Exclude all particles outside the empirically determined virial radius at z=0
            dat = dat[dat["r"]<=self.r_vir]

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
        disc_stars_idx = np.logical_and.reduce([(K_rot/K_tot>k_threshold),
                                                (dat["r"]<r_max),
                                                (dat["z"]**2<z_max**2)])
        
        return disc_stars_idx
    
    

    def plot_center_L_evolution(self, 
                                snapshots: list[int]):
        
        centre_pos = [] # box coordinates
        L_tilts = [] # cos(alpha)
        times = [] # Gyr

        for snap in snapshots:

            # Read age of the universe at snapshot
            f = h5py.File(f"{self.__snap_path__}snap_{snap:03}.hdf5")
            times.append(self.cosmo.age(f["Header"].attrs["Redshift"]).value)

            # Get raw simulation data
            dat, grp_dat = DREAMS_utils.load_zoom_particle_data_pynbody(self.__snap_path__, 
                                                                        self.__group_path__, 
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
        f = h5py.File(f"{self.__snap_path__}snap_{self.snap_z0:03}.hdf5")

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
        dat = self.__load_part_data__(snap=snap, PartType=1, r_cut=False)

        # Calculate critical density of the universe at snap
        if os.path.exists(f"{self.__snap_path__}snap_{snap:03}.hdf5"):
            f = h5py.File(f"{self.__snap_path__}snap_{snap:03}.hdf5")
        else:
            f = h5py.File(os.path.join(self.__snap_path__,f"snapdir_{snap}/snap_{snap:03}.0.hdf5"))
        z = f["Header"].attrs["Redshift"]
        f.close()
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

    
    def plot_subhalos_tracks(self):

        # Load merger tree
        tree = h5py.File(f"{self.__group_path__}tree_extended.hdf5")

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
                if snap==self.snap_z0:
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
    
    

    def track_particles(self, particleIDs, PartType, z_range=None, snap_path=None):
        
        # --- Helper: Vectorized Distance Calculation ---
        def get_best_match_indices(pred_pos, pred_vel, pred_mass, 
                                cand_pos, cand_vel, cand_mass, 
                                k=5):
            """
            Finds the best match based on Position, Velocity, and Mass.
            1. Finds k-nearest spatial neighbors.
            2. Filters/Scores them based on Velocity difference and Mass similarity.
            """
            # Build Tree once for the candidate snapshot
            tree = cKDTree(cand_pos)
            
            # Query k nearest neighbors for all particles at once
            # dists: (N_tracked, k), indices: (N_tracked, k)
            dists, indices = tree.query(pred_pos, k=k)
            
            best_indices = []
            
            # Iterate through each tracked particle to find the best among the k neighbors
            for i in range(len(pred_pos)):
                # Get the indices of the k candidates for this particle
                neighbor_idxs = indices[i]
                
                # Extract properties of these neighbors
                n_vel = cand_vel[neighbor_idxs]   # shape (k, 3)
                n_mass = cand_mass[neighbor_idxs] # shape (k,)
                
                # --- SCORING METRIC ---
                
                # 1. Velocity Score: Euclidean distance in velocity
                # We predict velocity is constant. Calculate diff.
                v_diff = np.linalg.norm(n_vel - pred_vel[i], axis=1)
                
                # 2. Mass Score: Fractional difference
                # Avoid divide by zero
                m_target = pred_mass[i] 
                m_diff = np.abs(n_mass - m_target) / m_target
                
                # 3. Combine Scores (Heuristic weights)
                # Weights depend on simulation units, but generally:
                # We prioritize Mass (hard constraint) and Velocity (phase space)
                # Since 'dists' (position) is already minimized by k-NN, we use it as a tiebreaker
                # if velocity/mass are similar.
                
                # Simple weighted cost: 
                # cost = w_v * v_diff + w_m * m_diff * scaling
                # Heuristic: reject if mass differs by > 5%
                valid_mass_mask = m_diff < 1.0 
                
                if np.any(valid_mass_mask):
                    # If we have valid mass candidates, pick the one with closest velocity
                    masked_v_diff = v_diff.copy()
                    masked_v_diff[~valid_mass_mask] = np.inf
                    best_k_index = np.argmin(masked_v_diff)
                else:
                    # Fallback: Just pick the closest position (index 0)
                    best_k_index = 0
                    
                best_indices.append(neighbor_idxs[best_k_index])
                
            return np.array(best_indices)

        # --------------------------------------------------

        out = {pid: {"xyz": [], "v_xyz": [], "mass": []} for pid in particleIDs}
        
        # Load snapshots
        snapshots_indices = convert_zrange_to_snapshots(z_range, self.__snap_path__)
        snapshots_indices = list(range(snapshots_indices[0], snapshots_indices[-1]+1))
        
        # Pre-load snapshots to clean up missing ones
        # (Memory optimized: In large sims, don't load ALL at once. 
        # But sticking to your logic, we clean the list first)
        valid_snapshots = []
        valid_data = []
        
        for snap in snapshots_indices:
            # (Your existing try/except loading logic here)
            try:
                p_data = self.__load_part_data__(snap=snap, snap_path=snap_path, PartType=PartType)
                valid_snapshots.append(snap)
                valid_data.append(p_data)
            except (FileNotFoundError, KeyError):
                if "res_varied_tng" in self.__snap_path__:
                    try:
                        # Fallback path logic
                        p_data = self.__load_part_data__(snap=snap, snap_path="/mnt/home/jrose/ceph/res_varied_tng/adaptive/RUNs/output/quick_snaps/", PartType=PartType)
                        valid_snapshots.append(snap)
                        valid_data.append(p_data)
                    except:
                        continue
                else:
                    continue

        if not valid_data:
            return {}

        redshifts = [d["Redshift"][0] for d in valid_data]
        times = [self.cosmo.age(z).to(u.Gyr) for z in redshifts]

        # --- Initialization ---
        # We need to maintain the "Current State" of the particles we are tracking
        # to perform predictions for the next step.
        
        # Check first snapshot for IDs
        first_snap_data = valid_data[0]
        
        # Indices of requested PIDs in the first snapshot
        # This assumes all PIDs exist in first snapshot as per your code
        mask = np.isin(first_snap_data["iord"], particleIDs)
        
        # We need a mapping from PID -> Index in current arrays to keep order sorted
        # Let's align everything to the order of 'particleIDs' input
        current_xyz = []
        current_vxyz = []
        current_mass = []
        
        valid_pids = []
        
        for pid in particleIDs:
            idx = np.where(first_snap_data["iord"] == pid)[0]
            if len(idx) > 0:
                idx = idx[0]
                current_xyz.append([first_snap_data[f][idx] for f in ["x","y","z"]]) # magnitude
                current_vxyz.append([first_snap_data[f][idx] for f in ["vx","vy","vz"]]) # magnitude
                current_mass.append(first_snap_data["mass"][idx])
                valid_pids.append(pid)
                
                # Save first step to output
                out[pid]["xyz"].append(np.array([first_snap_data[f][idx] for f in ["x","y","z"]]) * u.kpc)
                out[pid]["v_xyz"].append(np.array([first_snap_data[f][idx] for f in ["vx","vy","vz"]]) * (u.km/u.s))
            
            # Append time info
        
        # Convert to numpy arrays for vectorization (N_particles, 3)
        current_xyz = np.array(current_xyz)
        current_vxyz = np.array(current_vxyz)
        current_mass = np.array(current_mass)
        
        particleIDs = valid_pids # Only track found particles

        # --- Tracking Loop ---
        # We iterate snapshots, not particles. This is the key optimization.
        
        for i in range(1, len(valid_snapshots)):
            prev_data = valid_data[i-1]
            curr_data = valid_data[i]
            
            # Prepare Current Snapshot Data Arrays (Candidates)
            cand_pos = np.vstack([curr_data["x"], curr_data["y"], curr_data["z"]]).T
            cand_vel = np.vstack([curr_data["vx"], curr_data["vy"], curr_data["vz"]]).T
            cand_mass = curr_data["mass"]
            
            matched_indices = None

            # METHOD A: IDs exist in this snapshot (Exact Match)
            if "iord" in curr_data.keys():
                # Create a lookup dictionary for speed: ID -> Array Index
                id_to_index = {pid: idx for idx, pid in enumerate(curr_data["iord"])}
                
                iteration_indices = []
                found_mask = []
                
                for pid in particleIDs:
                    if pid in id_to_index:
                        iteration_indices.append(id_to_index[pid])
                        found_mask.append(True)
                    else:
                        # Particle lost even though IDs exist (rare/error)
                        iteration_indices.append(0) # Placeholder
                        found_mask.append(False)
                
                matched_indices = np.array(iteration_indices)
                # Note: You might want to handle lost particles here, 
                # but for now we assume they are found if IDs exist.

            # METHOD B: IDs do not exist (Predictive Match)
            else:
                # 1. Calculate dt
                z_prev = redshifts[i-1]
                z_curr = redshifts[i]
                prev_t = self.cosmo.age(z_prev)
                curr_t = self.cosmo.age(z_curr)
                dt = (curr_t - prev_t).to(u.s).value # Get magnitude in seconds
                
                # Convert velocity units if necessary. 
                # Assuming Position is kpc and Velocity is km/s:
                # 1 km/s = 1.022e-9 kpc/s roughly, or convert explicitly using astropy
                v_conv_factor = (1 * (u.km/u.s)).to(u.kpc/u.s).value
                
                # 2. Predict Positions (Linear Extrapolation)
                # x_pred = x_prev + v_prev * dt
                pred_pos = current_xyz + (current_vxyz * v_conv_factor * dt)
                
                # 3. Find Matches (Vectorized KDTree + Physics Check)
                matched_indices = get_best_match_indices(
                    pred_pos, current_vxyz, current_mass,
                    cand_pos, cand_vel, cand_mass,
                    k=20 # Check top 10 nearest spatial neighbors
                )

            # --- Update State & Save ---
            
            # Extract new data using the found indices
            new_pos = cand_pos[matched_indices]
            new_vel = cand_vel[matched_indices]
            new_mass = cand_mass[matched_indices]
            
            # Update current state for the *next* iteration prediction
            current_xyz = new_pos
            current_vxyz = new_vel
            current_mass = new_mass
            
            # Append to output dictionary
            for k, pid in enumerate(particleIDs):
                out[pid]["xyz"].append(new_pos[k] * u.kpc)
                out[pid]["v_xyz"].append(new_vel[k] * (u.km/u.s))

        # Formatting output arrays (vstack) as per original requirement
        for pid in particleIDs:
            out[pid]["xyz"] = np.vstack(out[pid]["xyz"]).T # shape (3, N_snaps) usually preferred, check your desired output
            out[pid]["v_xyz"] = np.vstack(out[pid]["v_xyz"]).T
            out[pid]["times"] = times

        return out
      
    
    
    
    

  
class EXPBFE_builder():

    def __init__(self, 
                 sim,
                 basis_params_dict: dict,
                 density_dict: dict,
                 z_range: list[float], # can be int or str depending if it isn't or is highcadence
                 output_dir: str):
        
        self.sim = sim
        self.__output_dir__ = output_dir
        self.snapshots = convert_zrange_to_snapshots(z_range, self.sim.__snap_path__)
        self.snapshots = [s for s in range(self.snapshots[0], self.snapshots[-1]+1)]
        # Print first and last snapshot to check
        print(f"Snapshots used for the expansion: {self.snapshots[0]} to {self.snapshots[-1]}")
        
        # Define units of the simulation
        self.exp_units = SimulationUnitSystem(mass=self.sim.M_vir*u.Msun, 
                                              length=self.sim.r_vir*u.kpc, 
                                              G=1)
        
        # Define the name of the output files
        self.model_files_dict = {} # Density tables
        self.basis_files_dict = {} # Basis functions
        self.coefs_files_dict = {} # Coefficients
        
        for PartType in basis_params_dict.keys():
            
            self.model_files_dict[PartType] = f"{self.__output_dir__}basis_empirical_PartType{PartType}.txt" 
            self.basis_files_dict[PartType] = f"{self.__output_dir__}basis_yaml_PartType{PartType}.yml"
            self.coefs_files_dict[PartType] = f"{self.__output_dir__}coefs_PartType{PartType}.h5"
                
        
        # Build basis
        print("Building basis for the expansion...", flush=True)
        self.basis = {}
        for PartType, basis_params in basis_params_dict.items():
            basis = self.__build_basis__(PartType=PartType,
                                         basis_params=basis_params,
                                         density_params= density_dict[PartType])
            self.basis[PartType] = basis
            
        # Calculate the coefficients 
        print(f"Calculating the coefficients at snapshots: {self.snapshots}", flush=True)
        self.coefs = {}
        # Save time steps at which the covariance matrix is calculated
        self.covariance_times = []
        for PartType, basis in self.basis.items():
            start = time.time()
            self.coefs[PartType] = self.__get_coefs__(basis=basis,
                                                      snapshots=self.snapshots,
                                                      PartType=PartType)
            print(f"Time taken for PartType {PartType}: {(time.time()-start)/60:.2f} minutes", flush=True)
        
        print("Covariance matrices are calculated at the following time steps (Gyr):", self.covariance_times)
        
    
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
            yaml_file = self.__get_star_basis_config__(basis_params, density_params)
        
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
                             "rangevals": [0.1,600]}
        
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
                                 "pcavar": True, # enable to calculate the coefficients covariance matrix 
                                 "subsamp": 1000,
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
                                 "pcavar": True,     # enable to calculate the coefficients covariance matrix 
                                 "subsamp": 1000,
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
            try:
                dat = self.sim.__load_part_data__(snap=snap, PartType=PartType)   
            except FileNotFoundError:
                if "res_varied_tng" not in self.sim.__snap_path__:
                    print(f"Snapshot {snap} not found")
                    continue
                else:
                    dat = self.sim.__load_part_data__(snap=snap, 
                                                      snap_path="/mnt/home/jrose/ceph/res_varied_tng/adaptive/RUNs/output/quick_snaps/", 
                                                      PartType=PartType)


            # Scale to virial units
            mass = np.array(dat["mass"]) / self.sim.M_vir
            pos = np.vstack([dat["x"], dat["y"], dat["z"]]).T / self.sim.r_vir


            # Read age of the universe at snapshot
            z = dat["Redshift"][0]
            t = self.sim.cosmo.age(z).value                                                           
            
            # Calculate the coefficients of the BFE 
            coefs = basis.createFromArray(mass, 
                                          pos, 
                                          time=t)
            # Compute the covariance matrix for the coefficients only every 300 Myr
            if np.isclose(t%0.5, 0, atol=1e-3) or snapshots.index(snap)==len(snapshots)-1:
                self.covariance_times.append(t)
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
    
        
    def __update_coefs__(self, 
                         new_coefs_file: str, 
                         PartType: int):
        
        self.coefs_files_dict[PartType] = new_coefs_file
        # Read new coefficient matrix at snapshots
        new_coefs = pyEXP.coefs.Coefs.factory(new_coefs_file)
        self.coefs[PartType] = new_coefs
        print(f"PartType{PartType} coefficients updated.")



    def build_gala_potential(self, **kwargs):

        
        pot = gp.CCompositePotential()
        for PartType in self.basis.keys():
            # Read basis and coefficients of EXP approximation
            pot[PartType] = gp.EXPPotential(units=self.exp_units,
                                            config_file=self.basis_files_dict[PartType],
                                            coef_file=self.coefs_files_dict[PartType],
                                            snapshot_time_unit=u.Gyr, **kwargs)

        return pot, self.exp_units
    
    
    def get_SNR_matrix(self, 
                       basis_PartType: int,
                       time: float, 
                       decorrelate=False):
        """
        Returns the signal-to-noise (SNR, i.e. amplitude to variance) ratio for the coefficients 
        of the basis functions for all the radial orders associated to the given m and l 
        orders at a given time. 

        Args:
            time (float): time at which the coefficients are evaluated

        Returns:
            ndarray: array of the SNR for each radial mode at the specified angular mode (l(l+1),n)
        """
        
        # Get the values and covariance of the coefficients for the basis at time t
        # output is a list where each element refers to a given spherical orders
        
        covar = self.__get_CoefCovariance__(PartType=basis_PartType)
        coefs_var_subsamples = covar.getCoefCovariance(time)
        
        n_subsamples = len(coefs_var_subsamples)
        
        if basis_PartType==1:
        
            # Read-in the order of the expansion
            with open(self.basis_files_dict[basis_PartType], "r") as yaml_file:
                basis_params = yaml.safe_load(yaml_file)
                lmax = basis_params["parameters"]["Lmax"]
                nmax = basis_params["parameters"]["nmax"]

            SNR_mesh = []
            for l in range(lmax+1):
                for m in range(l+1):
                    
                    # Define the index of the coefficients and covariance basis in the outputs
                    spherical_index = (l * (l + 1)) // 2 + m
                    
                    # Read coefficients from each subsample
                    sumCof = np.zeros((nmax), dtype=np.complex128)
                    sumVar = np.zeros((nmax, nmax), dtype=np.complex128)
                    for subsample in coefs_var_subsamples:
                        sumCof += subsample[spherical_index][0]
                        sumVar += subsample[spherical_index][1]
                    # Subtract cross-samples covariance
                    sumVar -= np.outer(sumCof, sumCof.conj())
                    
                    if decorrelate:
                        # Make eigenvalue analysis on covariance matrix
                        val, vec = np.linalg.eigh(sumVar)
                        # Project coefficients into decorrelated basis
                        b = np.dot(vec.T, sumCof)
                        
                        SNR = np.abs(b)**2 * n_subsamples / np.abs(val)
                    
                    else:
                    
                        SNR = np.abs(sumCof)**2 *n_subsamples / np.abs(np.diag(sumVar))

                    SNR_mesh.append(SNR)
                    
        elif basis_PartType==4:
            
            with open(self.basis_files_dict[basis_PartType], "r") as yaml_file:
                basis_params = yaml.safe_load(yaml_file)
                mmax = basis_params["parameters"]["mmax"]
                nmax = basis_params["parameters"]["nmax"]
                
            SNR_mesh = []
            for m in range(mmax+1):
                
                # Read coefficients from each subsample
                sumCof = np.zeros((nmax), dtype=np.complex128)
                sumVar = np.zeros((nmax, nmax), dtype=np.complex128)
                for subsample in coefs_var_subsamples:
                    sumCof += subsample[m][0]
                    sumVar += subsample[m][1]
                # Subtract cross-samples covariance
                sumVar -= np.outer(sumCof, sumCof.conj())
                
                if decorrelate:
                    # Make eigenvalue analysis on covariance matrix
                    val, vec = np.linalg.eigh(sumVar)
                    # Project coefficients into decorrelated basis
                    b = np.dot(vec.T, sumCof)
                    
                    SNR = np.abs(b)**2 * n_subsamples / np.abs(val)
                
                else:
                
                    SNR = np.abs(sumCof)**2 * n_subsamples / np.abs(np.diag(sumVar))

                SNR_mesh.append(SNR)
            
        
        
        return np.vstack(SNR_mesh)


    def suppress_coefficients(self,
                              basis_PartType,
                              mask,
                              update=False):
        
        # Read original coefficients
        coefs_file = self.coefs_files_dict[basis_PartType]
        coefs = pyEXP.coefs.Coefs.factory(coefs_file)
        coefs_values = coefs.getAllCoefs()
        
        # Mask coefficient at each snapshot
        for i,time in enumerate(coefs.Times()):
            new_coefs = coefs_values[:,:,i]
            new_coefs[mask] = np.complex128(0)
            coefs.setMatrix(time, coefs_values[:,:,i])
            
        # Write new coefficient file
        new_coefs_file = coefs_file.replace(".h5","_masked.h5")
        
        if os.path.exists(new_coefs_file):
            os.remove(new_coefs_file)
        coefs.WriteH5Coefs(new_coefs_file) 
        
        if update:
            self.__update_coefs__(new_coefs_file=new_coefs_file,
                                  PartType=basis_PartType)
                    
        return new_coefs_file

  
    
########################################################################

                                                   
def convert_zrange_to_snapshots(z_range, snap_path):
    """
    Finds all snapshots within a given redshift range [z_min, z_max].
    Optimized to read file headers only once.
    """
    print("Indexing snapshot directory...", flush=True)
    
    # 1. Get all .hdf5 files and sort them by snapshot number immediately
    # This prevents 'os.listdir' from returning arbitrary order
    files = [f for f in os.listdir(snap_path) if f.startswith('snap')]
    
    # Sort by the integer number in the filename (e.g., snap_002.hdf5 -> 2)
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    if not files:
        raise FileNotFoundError(f"No 'snap_*.hdf5' files found in {snap_path}")

    # 2. Single I/O Pass: Build the redshift map
    snap_nums = []
    redshifts = []

    for f_name in files:
        # Construct full path
        full_path = os.path.join(snap_path, f_name)
        if not full_path.endswith(".hdf5"):
            snap_n = f_name.split("_")[1]
            full_path = os.path.join(snap_path, f"{f_name}/snap_{snap_n}.0.hdf5")
        
        try:
            with h5py.File(full_path, "r") as f:
                # Read redshift from header
                z = f['Header'].attrs['Redshift']
                
                # Extract snapshot number from filename
                n = int(f_name.split('_')[1].split('.')[0])
                
                snap_nums.append(n)
                redshifts.append(z)
        except (OSError, KeyError):
            # Skip corrupted files or files locked by other processes
            print(f"Warning: Could not read {f_name}")
            continue

    snap_nums = np.array(snap_nums)
    redshifts = np.array(redshifts)

    # 3. Vectorized Search
    # Find indices of the snapshots closest to the requested z_range
    # We use np.abs().argmin() to find the nearest match in the array
    idx_z0 = np.argmin(np.abs(redshifts - z_range[0]))
    idx_z1 = np.argmin(np.abs(redshifts - z_range[1]))

    # Check precision (optional warning)
    if np.abs(redshifts[idx_z0] - z_range[0]) > 0.1:
        print(f"Warning: Closest snapshot for z={z_range[0]} is z={redshifts[idx_z0]:.2f}")

    # 4. Slice the range
    # Ensure we slice from low index to high index, regardless of z-direction
    start, end = sorted([idx_z0, idx_z1])
    
    # Inclusive slicing
    selected_snaps = snap_nums[start : end + 1]
    
    print(f"Found {len(selected_snaps)} snapshots in range {z_range} "
          f"(Snap {selected_snaps[0]} to {selected_snaps[-1]})")
    
    return selected_snaps.tolist()



def get_snapshot_files(snap_path):
        
    # Get all the .hdf5 files in the directory
    outputs = [f for f in os.listdir(snap_path) 
               if f.startswith('snap')]
    
    # Order files by snapshot
    try:
        idx = np.argsort([int(out.split("_")[1]) for out in outputs])
    except ValueError:
        idx = np.argsort([int(out.split("_")[1].replace(".hdf5","")) for out in outputs])
    ordered_outputs = [outputs[i] for i in idx]
    
    return ordered_outputs



def reconstruct_track(ics, 
                      pot,
                      t1,
                      t2,
                      dt: float = 10*u.Myr):
    
    track_dict = {}

    start=time.time()

    orbit = pot.integrate_orbit(ics, 
                                t1=t1, 
                                t2=t2, 
                                dt=dt, 
                                Integrator=gi.DOPRI853Integrator, 
                                Integrator_kwargs={"atol":1e-6})
    print(f"Done in {time.time()-start:.1f} seconds.")
    
    # Exclude 1st and last snapshots because of BFE cannot be evaluated there
    idx = np.arange(len(orbit.t))[1:-1]
    # Position of the particle in the orbit
    xyz = orbit.xyz.to(u.kpc)[:,idx] 
    # Velocity of the particle in the orbit
    v_xyz = orbit.v_xyz.to(u.km/u.s)[:,idx]
    # Time steps at which the orbit is evaluated
    times = orbit.t.to(u.Gyr)[idx]
    # Compute the energy at each time step
    V = pot.energy(xyz, t=times).to(u.km**2 / u.s**2) 
    K = 0.5*np.sum(v_xyz**2, axis=0)
    E = K + V
    
    # Compute apocenter, pericenter, and eccentricity of the orbit
    apo = orbit.apocenter().to(u.kpc)
    peri = orbit.pericenter().to(u.kpc)
    ecc = orbit.eccentricity()
    
    # Save outputs
    track_dict = {"xyz": xyz,
                  "v_xyz": v_xyz,
                  "E": E,
                  "times": times,
                  "apo": apo,
                  "peri": peri, 
                  "e": ecc}
        
    return track_dict



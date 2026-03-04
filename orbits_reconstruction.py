import astropy.units as u
import gala.dynamics as gd
import matplotlib.pyplot as plt
import numpy as np 
import os
import pickle
from scipy.stats import wasserstein_distance_nd
import sys
import time

sys.path.append("/mnt/home/asante/streams_in_dreams/")
import EXP4DREAMS
from EXP_visual_fns import surface_projection, plot_SNR_mesh, plot_orbit_reconstruction, compare_distributions


def visualize_expansion(exp_builder, 
                        filename, 
                        PartType=None, 
                        label=None,
                        lim=20):
    # 2D projections of the density and potential fields
    fig,axs = plt.subplots(2,3,sharex=True, sharey=True, 
                        gridspec_kw={"wspace":0, "hspace":0})

    for i,field in enumerate(["dens", "potl"]):
        surface_projection(basis=exp_builder.basis[PartType],
                        coefs=exp_builder.coefs[PartType],
                        exp_units=exp_builder.exp_units,
                        field=field, 
                        time=exp_builder.coefs[PartType].Times()[-1],
                        extent=[[-lim,-lim,0]*u.kpc,[lim,lim,0]*u.kpc],
                        grid=[100,100,0],
                        ax=axs[i,0])

        surface_projection(basis=exp_builder.basis[PartType],
                        coefs=exp_builder.coefs[PartType],
                        exp_units=exp_builder.exp_units,
                        field=field, 
                        time=exp_builder.coefs[PartType].Times()[-1],
                        extent=[[-lim,0,-lim]*u.kpc,[lim,0,lim]*u.kpc],
                        grid=[100,0,100],
                        ax=axs[i,1])

        surface_projection(basis=exp_builder.basis[PartType],
                        coefs=exp_builder.coefs[PartType],
                        exp_units=exp_builder.exp_units,
                        field=field, 
                        time=exp_builder.coefs[PartType].Times()[-1],
                        extent=[[0,-lim,-lim]*u.kpc,[0,lim,lim]*u.kpc],
                        grid=[0,100,100],
                        ax=axs[i,2])  

    axs[0,0].set_ylabel("Density")
    axs[1,0].set_ylabel("Potential")

    axs[0,0].set_title("(x,y)")
    axs[0,1].set_title("(x,z)")
    axs[0,2].set_title("(y,z)")
    
    fig.savefig(filename, dpi=400)
    
    return

def calculate_orbital_parameters(out: dict):
    """
    Calculates orbital parameters consistent with Gala definitions.
    
    Parameters:
    -----------
    out : dict
        Output from track_particles. Format: {pid: {'xyz': array, 'v_xyz': array}}
        
    Returns:
    --------
    results : np.ndarray
        A structured numpy array with fields:
        ['id', 'peri', 'apo', 'ecc', 'Lx', 'Ly', 'Lz', 'L_mag']
        Shape: (N_particles,)
    """
    
    # Define the dtype for the structured array
    # 'id': int or float depending on your ID type (usually huge ints)
    # 'peri', 'apo', 'ecc', 'L_mag': floats
    # 'Lx', 'Ly', 'Lz': floats (components of angular momentum)
    
    # Pre-allocate the result array
    results = []

    for i, (pid, data) in enumerate(out.items()):
        # Extract arrays (Chronological: t_start -> t_final)
        # Handle unit stripping if astropy quantities are present
        pos = data["xyz"].value if hasattr(data["xyz"], "value") else data["xyz"]
        vel = data["v_xyz"].value if hasattr(data["v_xyz"], "value") else data["v_xyz"]
        
        # 1. Radial History r(t)
        # Calculate norm across time axis (axis 1) if shape is (N_steps, 3)
        r = np.linalg.norm(pos, axis=1)
        
        # 2. Basic Shape Parameters
        peri = np.percentile(r, 10)
        apo = np.percentile(r, 90)
        
        denom = apo + peri
        ecc = (apo - peri) / denom if denom > 0 else 0.0
        
        # 3. Angular Momentum (Final Snapshot)
        # L = r x v (at index -1)
        L_vec = np.cross(pos[:,-1], vel[:,-1])
        L_mag = np.linalg.norm(L_vec)
        
        # 4. Fill the array
        results.append(np.array([peri,
                                 apo,
                                 ecc,
                                 L_mag]))
        
    results = np.vstack(results)

    return results


def main(snap_path,
         group_path,
         output_path,
         n_particles,
         z_range,
         basis_dict,
         density_dict):
    
    rng = np.random.default_rng(seed=42)
    
    # Load the Simulation data
    MW_sim = EXP4DREAMS.DREAMSMW(snap_path=snap_path,
                                 group_path=group_path)
    
    # Track particles across the simulation snapshots
    relevant_snaps = EXP4DREAMS.convert_zrange_to_snapshots(z_range=z_range, snap_path=snap_path)
    particles = MW_sim.__load_part_data__(snap=relevant_snaps[0], 
                                          PartType=4)
    ids = rng.choice(particles["iord"][particles["r"]<100],
                     size=n_particles, replace=False)
    sim_tracks = MW_sim.track_particles(ids, 
                                        PartType=4, 
                                        z_range=z_range)
    # Save the original tracks
    pickle.dump(sim_tracks, open(os.path.join(output_path,"original_tracks.pkl"), "wb"))
    
    # Perform EXP expansion
    EXP_gen = EXP4DREAMS.EXPBFE_builder(sim=MW_sim,
                                        basis_params_dict=basis_dict,
                                        density_dict=density_dict,
                                        z_range=z_range,
                                        output_dir=output_path)
    
    # Visualize the density and potential fields (halo / PartType=1)
    visualize_expansion(exp_builder=EXP_gen,
                        filename=os.path.join(output_path,"original_fields_halo.pdf"),
                        PartType=1,
                        lim=80,
                        label="Halo (PartType=1)")

    # Visualize the density and potential fields (stars / PartType=4)
    visualize_expansion(exp_builder=EXP_gen,
                        filename=os.path.join(output_path,"original_fields_stars.pdf"),
                        PartType=4,
                        lim=20,
                        label="Stars (PartType=4)")
    
    
    # Calculate SNR of basis coefficients
    print("Calculating SNR matrix...")
    start = time.time()
    SNR_matrix_avg = []
    for t in EXP_gen.covariance_times:
        SNR_matrix = EXP_gen.get_SNR_matrix(basis_PartType=1, time=t)
        SNR_matrix_avg.append(SNR_matrix)
    SNR_matrix_avg = np.mean(SNR_matrix_avg, axis=0)
    fig,ax = plt.subplots()
    plot_SNR_mesh(SNR_matrix_avg,ax)
    fig.savefig(os.path.join(output_path,"SNR_matrix_original.pdf"),dpi=400)
    print(f"Time taken: {time.time() - start:.1f} seconds")
    
    # Repeat for stars
    start = time.time()
    SNR_matrix_avg_stars = []
    for t in EXP_gen.covariance_times:
        SNR_matrix = EXP_gen.get_SNR_matrix(basis_PartType=4, time=t)
        SNR_matrix_avg_stars.append(SNR_matrix)
    SNR_matrix_avg_stars = np.mean(SNR_matrix_avg_stars, axis=0)
    fig,ax = plt.subplots()
    plot_SNR_mesh(SNR_matrix_avg_stars,ax)
    fig.savefig(os.path.join(output_path,"SNR_matrix_stars.pdf"),dpi=400)
    print(f"Time taken: {time.time() - start:.1f} seconds")
    
    
    # Track particles in the reconstructed potential
    print("Reconstructing orbits in the BFE potential...")
    pot, _ = EXP_gen.build_gala_potential()
    t1 = (EXP_gen.coefs[1].Times()[0]+1e-3)*u.Gyr
    t2 = (EXP_gen.coefs[1].Times()[-1]-1e-3)*u.Gyr
    bfe_tracks = {}
    skipped_ids = []
    for pid in sim_tracks.keys():
        try:
            ics = gd.PhaseSpacePosition(pos=sim_tracks[pid]["xyz"][:,0], 
                                        vel=sim_tracks[pid]["v_xyz"][:,0])
        except TypeError:
            print(f"Missing data for particle {pid}, skipping...")
            skipped_ids.append(pid)
            continue
        bfe_tracks[pid] = EXP4DREAMS.reconstruct_track(ics,
                                                       pot=pot,
                                                       t1=t1,
                                                       t2=t2,
                                                       dt=10*u.Myr)
        
    print(f"Reconstructed tracks for {len(bfe_tracks)} particles. Missing data for {len(skipped_ids)} particles.")
        
    # Save the reconstructed tracks
    np.savez(os.path.join(output_path,"bfe_tracks.npz"),
             bfe_tracks=bfe_tracks)
    
    # Estimate difference between original and reconstructed tracks
    sim_orbits = calculate_orbital_parameters({k:v for k,v in sim_tracks.items() if k not in skipped_ids})
    bfe_orbits = calculate_orbital_parameters(bfe_tracks)
    
    # Plot some of the orbits
    fig,axs = plt.subplots(5,3)
    selected_ids = rng.choice(ids, size=5, replace=False)
    for i,pid in enumerate(selected_ids):
        plot_orbit_reconstruction(pid=pid,
                                  sim_tracks=sim_tracks,
                                  bfe_tracks=bfe_tracks,
                                  axs=axs[i])
    fig.savefig(os.path.join(output_path, "sim_vs_bfe_single_orbits.pdf"), dpi=400)
    
    # Calculate difference between the original and reconstructed orbit distribution
    diff = wasserstein_distance_nd(sim_orbits, bfe_orbits)
    fig = compare_distributions(data1=sim_orbits,
                                data2=bfe_orbits,
                                labels=["peri", "apo", "ecc", "L_mag"],
                                title=f"Diff {diff:.1f}")
    fig.savefig(os.path.join(output_path, "sim_vs_bfe_dist_orbits.pdf"),dpi=400)


    # Mask the expansion components with SNR < 10
    mask = SNR_matrix_avg < 5
    masked_SNR_matrix = SNR_matrix_avg.copy()
    masked_SNR_matrix[mask] = 0.
    fig,ax = plt.subplots()
    plot_SNR_mesh(masked_SNR_matrix, ax)
    fig.savefig(os.path.join(output_path,"SNR_matrix_masked.pdf"),dpi=400)
    
    _ = EXP_gen.suppress_coefficients(basis_PartType=1,
                                      mask=mask,
                                      update=True)  
    
    # Repeat for stars
    mask_stars = SNR_matrix_avg_stars < 5
    masked_SNR_matrix_stars = SNR_matrix_avg_stars.copy()
    masked_SNR_matrix_stars[mask_stars] = 0.
    fig,ax = plt.subplots()
    plot_SNR_mesh(masked_SNR_matrix_stars, ax)
    fig.savefig(os.path.join(output_path,"SNR_matrix_stars_masked.pdf"),dpi=400)
    
    _ = EXP_gen.suppress_coefficients(basis_PartType=4,
                                      mask=mask_stars,
                                      update=True)
    
    # Visualise the new fields (halo / PartType=1)
    visualize_expansion(exp_builder=EXP_gen,
                        filename=os.path.join(output_path,"masked_fields_halo.pdf"),
                        PartType=1,
                        lim=80,
                        label="Masked Halo (PartType=1)")
    
    # Visualise the new fields (stars / PartType=4)
    visualize_expansion(exp_builder=EXP_gen,
                        filename=os.path.join(output_path,"masked_fields_stars.pdf"),
                        PartType=4,
                        lim=20,
                        label="Masked Stars (PartType=4)")
    
    # Repeat the orbit reconstruction with the masked expansion
    print("Reconstructing orbits in the masked BFE potential...")
    pot, _ = EXP_gen.build_gala_potential()
    bfe_tracks = {}
    skipped_ids = []
    for pid in sim_tracks.keys():
        try:
            ics = gd.PhaseSpacePosition(pos=sim_tracks[pid]["xyz"][:,0], 
                                        vel=sim_tracks[pid]["v_xyz"][:,0])
        except TypeError:
            skipped_ids.append(pid)
            continue
        
        bfe_tracks[pid] = EXP4DREAMS.reconstruct_track(ics,
                                                       pot=pot,
                                                       t1=t1,
                                                       t2=t2,
                                                       dt=10*u.Myr)
    print(f"Reconstructed tracks for {len(bfe_tracks)} particles after masking. Missing data for {len(skipped_ids)} particles.")
        
    # Save the new reconstructed tracks
    np.savez(os.path.join(output_path,"bfe_tracks_new.npz"),
             bfe_tracks=bfe_tracks)
    
    # Estimate difference between original and reconstructed tracks
    sim_orbits = calculate_orbital_parameters({k:v for k,v in sim_tracks.items() if k not in skipped_ids})
    bfe_orbits = calculate_orbital_parameters(bfe_tracks)
    
    # Plot some of the orbits
    fig,axs = plt.subplots(5,3)
    for i,pid in enumerate(selected_ids):
        plot_orbit_reconstruction(pid=pid,
                                  sim_tracks=sim_tracks,
                                  bfe_tracks=bfe_tracks,
                                  axs=axs[i])
    fig.savefig(os.path.join(output_path, "sim_vs_bfe_single_orbits_new.pdf"), dpi=400)
    
    # Calculate difference between the original and reconstructed orbit distribution
    diff = wasserstein_distance_nd(sim_orbits, bfe_orbits)
    fig = compare_distributions(data1=sim_orbits,
                                data2=bfe_orbits,
                                labels=["peri", "apo", "ecc", "L_mag"],
                                title=f"Diff {diff:.1f}")
    fig.savefig(os.path.join(output_path, "sim_vs_bfe_dist_orbits_new.pdf"),dpi=400)
    
    return


        
if __name__ == "__main__":
    
    snap_path = "/mnt/home/jrose/ceph/res_varied_tng/adaptive/RUNs/output/" 
    group_path = "/mnt/home/jrose/ceph/res_varied_tng/adaptive/RUNs/output/"
    #snap_path = "/mnt/home/dreams/ceph/Sims/CDM/MW_zooms/SB5/box_695/"
    #group_path = "/mnt/home/dreams/ceph/FOF_Subfind/CDM/MW_zooms/SB5/box_695/"  
    output_path = "/mnt/home/asante/ceph/orbit_reconstruction/with_disk/high_cadence/"
    n_particles = 500
    z_range = [1,0]
    
    basis_dict = {# PartType : basis params 
                1: {"Lmax": 6, "nmax": 20, "numr": 2000, "rmin": 0.001, "rmax":1.5},
                4: {"mmax": 6, "nmax": 12 }
                }
    density_dict = {1: {"bins": 500, 
                        "rangevals": [0.1, 100] 
                        },
                    4: {}
                    }
    
    main(snap_path=snap_path,
         group_path=group_path,
         output_path=output_path,
         n_particles=n_particles,
         z_range=z_range,
         basis_dict=basis_dict,
         density_dict=density_dict)
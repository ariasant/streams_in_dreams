import sys
sys.path.append("/mnt/home/asante/ceph/streams_in_dreams")
import EXP4DREAMS
import astropy.units as u

import matplotlib.pyplot as plt
import numpy as np
import gala.potential as gp
import gala.dynamics as gd
from gala.units import galactic
import gala.integrate as gi
import time
import pickle

#=======================================================
# User inputs

# Simulation
box = 34
snap_path = "/mnt/home/dreams/ceph/Sims/CDM/MW_zooms/SB5/"
group_path = "/mnt/home/dreams/ceph/FOF_Subfind/CDM/MW_zooms/SB5/"



# BFE
basis_dict = {# PartType : basis params 
              1: {"Lmax": 40, "nmax": 20, "numr": 2000}}
density_dict = {1: {"bins": 400, 
                    "rangevals": [0, 2.5] # log10 scale
                    }
                }
snapshots =  [61, 67, 73, 81, 90]
dt = 1*u.Myr # for orbit integration

# Output directory
output_dir = "/mnt/home/asante/ceph/parc/"


#=======================================================
# Create simulation object
MW_sim = EXP4DREAMS.DREAMSMW(box=box,
                             snap_path=snap_path,
                             group_path=group_path)

# Select DM particles at random from different regions of the galaxy
n = 10
DM_particles = MW_sim.__load_part_data__(snap=snapshots[0], PartType=1)
DM_ids = []
for r_min, r_max in [(0,5), (5,20), (20,50), (50,1000)]:
    # Select particles within that shell at random
    idx = (DM_particles["r"]>r_min) & (DM_particles["r"]<r_max)
    DM_ids_shell = DM_particles["iord"][idx] 
    
    DM_ids.append(np.random.choice(DM_ids_shell, size=n, replace=False))

DM_ids = np.hstack(DM_ids)
sim_tracks = MW_sim.track_particles(DM_ids, 1, snapshots=snapshots)

# Create EXPpotential
EXP_gen = EXP4DREAMS.EXPBFE_builder(sim=MW_sim,
                                    basis_params_dict=basis_dict,
                                    density_dict=density_dict,
                                    snapshots = snapshots,
                                    output_dir=output_dir)


pot, exp_units = EXP_gen.build_gala_potential()

# Use EXP approximation to replicate the particle orbits

bfe_tracks = {}

for pid in sim_tracks.keys():
    print(f"Integrating particle ID {pid}...")
    start=time.time()
    ics = gd.PhaseSpacePosition(pos=sim_tracks[pid]["xyz"][:,0], 
                                vel=sim_tracks[pid]["v_xyz"][:,0])
    
    orbit = pot.integrate_orbit(ics, 
                                t1=EXP_gen.coefs[1].Times()[0]*u.Gyr, 
                                t2=EXP_gen.coefs[1].Times()[-1]*u.Gyr, 
                                dt=dt, 
                                Integrator=gi.DOPRI853Integrator,
                                Integrator_kwargs={"atol":1e-4})
    print(f"Done in {time.time()-start} seconds.")
    
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
    bfe_tracks[pid] = {"xyz": xyz,
                       "v_xyz": v_xyz,
                       "E": E,
                       "times": times,
                       "apo": apo,
                       "peri": peri, 
                       "e": ecc
                       }
    
    

pickle.dump(bfe_tracks, open(f"{output_dir}bfe_tracks_l40n20.pkl", "wb"))
pickle.dump(sim_tracks, open(f"{output_dir}sim_tracks_l40n20.pkl", "wb"))



fig,axs = plt.subplots(1,2,figsize=(5,10))


##############################################
import pickle
from scipy.stats import binned_statistic
import numpy as np
import matplotlib.pyplot as plt

exp_N_list = [4, 5, 6]
# Use the 'viridis' colormap to generate a gradual color scale
cmap = plt.cm.get_cmap('plasma')
# Get 4 distinct, spaced colors from the colormap (0.1 to 0.9 range)
colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(exp_N_list))]

fig, ax = plt.subplots(figsize=(10, 6))

ax.set_xlabel(r"$E_0$", fontsize=12)
ax.set_ylabel(r"$\Delta E = E - E_0$", fontsize=12)
ax.set_ylim(-0.5, 0.5)
ax.grid(True, linestyle='--', alpha=0.5)

for i, exp_N in enumerate(exp_N_list):
    
    color = colors[i]

    # Load the data from the simulation
    """nbody_sim = pickle.load(open(f"/mnt/home/asante/ceph/parc/nbody_results_theta_0.1_N{exp_N}.pkl","rb"))
    analytical = pickle.load(open(f"/mnt/home/asante/ceph/parc/analytical_results_theta_0.1_N{exp_N}.pkl","rb"))"""
    nbody_sim = pickle.load(open(f"/mnt/home/asante/ceph/parc/nbody_results_N{exp_N}.pkl","rb"))
    analytical = pickle.load(open(f"/mnt/home/asante/ceph/parc/analytical_results_N{exp_N}.pkl","rb"))

    E0 = nbody_sim["pot0"] + nbody_sim["KE0"]
    E = nbody_sim["pot"] + nbody_sim["KE"]
    dE = E-E0

    # Bin particles by total initial energy
    bins = np.linspace(-1,0,20)

    p5, bin_edges, _ = binned_statistic(E0, E-E0, 
                                        bins=bins, 
                                        statistic=lambda x: np.percentile(x,5))
    p50, bin_edges, _ = binned_statistic(E0, E-E0, 
                                        bins=bins, 
                                        statistic=lambda x: np.percentile(x,50))
    p95, bin_edges, _ = binned_statistic(E0, E-E0, 
                                        bins=bins, 
                                        statistic=lambda x: np.percentile(x,95))
    # energy change
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Width of the bins (for bar width)
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Height of the bar (p95 - p5)
    bar_heights = p95 - p5


    # Plot the 5th-to-95th Percentile Range Bars
    ax.bar(bin_centers, 
        height=bar_heights, 
        width=bin_widths, 
        bottom=p5,
        color=color, 
        alpha=0.5,
        edgecolor='black',
        linewidth=1,
        label=f'$N: 10^{exp_N}$')

    # Plot the Median (50th percentile) as a distinct line
    ax.plot(bin_centers, 
            p50, 
            'o-', 
            color=color, 
            linewidth=2, 
            markersize=5)

ax.legend()
    

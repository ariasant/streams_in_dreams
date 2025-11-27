import matplotlib.pyplot as plt
import numpy as np
from pytreegrav import Accel, Potential
import DREAMS_utils


def generate_plummer(num_particles, total_mass, scale_radius, option, G=1.0):
    """
    Generates positions and velocities for particles in a Plummer distribution.

    Args:
        num_particles (int): The number of particles to generate.
        total_mass (float): The total mass of the system.
        scale_radius (float): The Plummer scale radius.
        G (float, optional): The gravitational constant. Defaults to 1.0.

    Returns:
        tuple: A tuple containing:
            - positions (np.ndarray): Array of shape (num_particles, 3) with x, y, z coordinates.
            - velocities (np.ndarray): Array of shape (num_particles, 3) with vx, vy, vz components.
    """

    # Generate random numbers for radial positions (r) and velocities (v)
    # based on the Plummer model's cumulative distribution functions.
    # This involves inverse transform sampling.

    # Radial positions
    u = np.random.rand(num_particles)
    r = scale_radius * (u**(-2/3) - 1)**(-0.5)

    # Velocities (using the method by Aarseth, Hénon, and Wiyanto)
    if option == 'original':
        v_esc_sq = 2 * G * total_mass / (scale_radius) # Escape velocity squared at r=0
    #Generate random numbers for velocities
        v_rand_u = np.random.rand(num_particles)
    # Calculate velocity magnitudes
        v_sq = v_esc_sq * (1 - r**2 / (r**2 + scale_radius**2)) * v_rand_u**(2/3)
        v = np.sqrt(v_sq)

    # Velocities (using dispersion profile)
    if option == 'disp':
        v_sig_sq = G*total_mass/6./np.sqrt(r*r+scale_radius*scale_radius)
        vx = np.sqrt(v_sig_sq)*np.random.normal(0.,1.,num_particles)
        vy = np.sqrt(v_sig_sq)*np.random.normal(0.,1.,num_particles)
        vz = np.sqrt(v_sig_sq)*np.random.normal(0.,1.,num_particles)


    # Generate random directions for positions (spherical coordinates)
    phi = 2 * np.pi * np.random.rand(num_particles)
    costheta = 2 * np.random.rand(num_particles) - 1
    sintheta = np.sqrt(1 - costheta**2)

    # Convert spherical coordinates to Cartesian positions
    x = r * sintheta * np.cos(phi)
    y = r * sintheta * np.sin(phi)
    z = r * costheta
    positions = np.array([x, y, z]).T

    # Generate random directions for velocities
    if option == "original":
        phi_v = 2 * np.pi * np.random.rand(num_particles)
        costheta_v = 2 * np.random.rand(num_particles) - 1
        sintheta_v = np.sqrt(1 - costheta_v**2)

    # Convert spherical coordinates to Cartesian velocities
        vx = v * sintheta_v * np.cos(phi_v)
        vy = v * sintheta_v * np.sin(phi_v)
        vz = v * costheta_v

    velocities = np.array([vx, vy, vz]).T

    return positions, velocities

def frog_step(dt, pos, masses, softening, vel, accel, option):
    # first a half-step kick
    vel = vel + 0.5 * dt * accel  # note that you must slice arrays to modify them in place in the function!
    # then full-step drift
    pos = pos + dt * vel
    # then recompute accelerations
    if option == 'tree':
        accel = Accel(pos, masses, softening, parallel=True, theta=theta)
    elif option == 'plum':
        accel = plum_accel(pos, M, a, G)      
    elif option == 'direct':
        accel = Accel(pos, masses, softening, parallel=True, method='bruteforce')
    # then another half-step kick
    vel = vel + 0.5 * dt * accel
    return pos, vel, accel

def run_sim(nsteps, 
            dt, 
            x, 
            v, 
            masses, 
            softening, 
            option, 
            t_outputs: list[float], 
            output_dir: str):
    

    if option == 'tree':
        accel = Accel(x, masses, softening, parallel=True, theta=theta)  # initialize acceleration
        pot0 = Potential(x, masses, softening, parallel=True, theta=theta)
    elif option == 'plum':
        accel = plum_accel(x, M, a, G)
        r = np.sqrt(x[:,0]*x[:,0]+x[:,1]*x[:,1]+x[:,2]*x[:,2])
        pot0 = plum_pot(r, M, a, G) 
        
    elif option == 'direct':
        accel = Accel(x, masses, softening, parallel=True, method="bruteforce")  # initialize acceleration
        pot0 = Potential(x, masses, softening, parallel=True, method="bruteforce")
    
    K0 = 0.5 * np.sum(v**2,axis=1) 

    t = 0.  # initial time
    istep = 0
    Tmax = dt*float(nsteps)  # final/max time

    # save system state as function of time
    energies = []  # energies
    vrat = []  # virial ratio
    rave = [] # average radius
    ts = []  # times

    while t <= Tmax:  # actual simulation loop - this may take a couple minutes to run
        # save system properties
        r = np.sqrt(x[:,0]*x[:,0]+x[:,1]*x[:,1]+x[:,2]*x[:,2])
        v2 = (v[:,0]*v[:,0]+v[:,1]*v[:,1]+v[:,2]*v[:,2])
        if option == 'tree':
            pot = Potential(x, masses, softening, parallel=True, theta=theta)
        elif option == 'plum':
            r = np.sqrt(x[:,0]*x[:,0]+x[:,1]*x[:,1]+x[:,2]*x[:,2])
            pot = plum_pot(r, M, a, G) 
        elif option == 'direct':
            pot = Potential(x, masses, softening, parallel=True, method='bruteforce')
            
            
        K = 0.5 * np.sum(v**2,axis=1) 
        # Total kinetic and potential energy of the system
        ke = 0.5 * np.sum(masses*v2)
        pe = 0.5 * np.sum(masses*pot)
        #
        rave.append(np.mean(r))
        energies.append(ke+pe)
        vrat.append(-pe/2./ke)
        ts.append(t)
        
        if istep in t_outputs:
            # save particle subset
            np.savez(f"{output_dir}Plummer_outs_t{istep:03}",
                     x=x,
                     v=v,
                     m=masses,
                     pot=pot
                     )
                        

        # advance 1 step
        x,v,accel = frog_step(dt, x, masses, softening, v, accel,option)
        t += dt
        istep += 1
    
    return rave, energies, vrat, ts, pot0, pot, K0, K, x, v
    
def plum_pot(r,M,a,G):
    pot = - G*M/np.sqrt(r*r+a*a)
    return pot

def plum_accel(x,M,a,G):
    r = np.sqrt(x[:,0]*x[:,0]+x[:,1]*x[:,1]+x[:,2]*x[:,2])
    ax = - G*M*x[:,0]/(r*r+a*a)**1.5
    ay = - G*M*x[:,1]/(r*r+a*a)**1.5
    az = - G*M*x[:,2]/(r*r+a*a)**1.5
    accel = np.array([ax, ay, az]).T
    return accel

def plum_rho(r,M,a,G):
    rho = (3.*M/4./np.pi/a**3)/(1.+(r/a)**2)**2.5
    return rho

def plum_sig(r,M,a,G):
    sig = G*M/6./np.sqrt(r*r+a*a)
    sig = np.sqrt(sig)
    return sig

# Plummer parameters
M = 1.0 # mass of the system
a = 1.0 # Plummer scale radius
G = 1.0
init_opt = "disp"

# Simulation parameters
eps = 0.1 # softening length
dt = 0.1
nsteps = 1000
theta = 0.7 # aperture of tree for force calculation

output_dir = "/mnt/home/asante/ceph/PlummerNbody/N6/"


N = 1000000
# Generate ICs
x0, v0 = generate_plummer(N,M,a,init_opt,G=1.0)

# Run N-body sim
masses = np.full(N,M/N)
softening = np.full(N, eps)

r_nb, e_nb, vir_nb, time_nb, pot0_nb, pot_nb, KE0_nb, KE_nb, xf, vf = run_sim(
        nsteps=nsteps,
        dt=dt,
        x=x0,
        v=v0,
        masses=masses,
        softening=softening,
        option="tree",
        t_outputs=[i for i in range(0,nsteps+1, 20)],
        output_dir=output_dir
    )


fig,axs = plt.subplots(1,2, layout="constrained")
fig.suptitle(f"N={N:,}")
# Total energy of the system
axs[0].plot(time_nb, (e_nb-e_nb[0])/e_nb[0])
axs[0].set_xlim([min(time_nb), max(time_nb)])
axs[0].set_ylim([-0.04, 0.01])
axs[0].set_aspect((max(time_nb)-min(time_nb))/(0.05))
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Fractional change in total energy")
#
ei_ps = pot0_nb + KE0_nb
ef_ps = pot_nb + KE_nb
axs[1].scatter(ei_ps, ef_ps-ei_ps, s=0.1)
axs[1].set_ylim([-0.5,0.5])
axs[1].set_xlim([-1,0])
axs[1].set_aspect(1)
axs[1].set_xlabel("Particle Initial Energy")
axs[1].set_ylabel("Change in Particle Energy")
fig.savefig(f"{output_dir}energy_change.pdf")


# Plot density profile
logr0 = np.log10(np.sqrt(np.sum(x0**2,axis=1)))
logrf = np.log10(np.sqrt(np.sum(xf**2,axis=1)))
density_params_df = {"bins": 100,
                     "rangevals": [0.5,1.5]}

rbins, dvals = DREAMS_utils.return_density(logr=logr0, weights=masses, **density_params_df, smooth=True)
rbinsf, dvalsf = DREAMS_utils.return_density(logr=logrf, weights=masses, **density_params_df, smooth=True)

# Plot inferred density vs analytical density
plum_dens = plum_rho(r=rbins, M=M, a=a, G=G)
fig,axs = plt.subplots(2,1, gridspec_kw={"hspace":0, "height_ratios":[3,1]}, sharex=True)
# ax.plot(rbins, dvals*(1/rbins**2)*4*np.pi, label="numerical")
axs[0].plot(rbins, dvals, label="numerical (initial)", c="r")
axs[0].plot(rbins, dvalsf, label="numerical (final)", c="b")
axs[0].plot(rbins, plum_dens, label="analytical", c="k")
axs[0].set_yscale("log")
axs[0].set_xscale("log")
axs[0].set_xlabel("$r/r_{scale}$")
axs[0].set_ylabel("Density")
axs[0].set_xlim([0,10])
axs[0].legend()

axs[1].plot(rbins, plum_dens-dvals, c="r")
axs[1].plot(rbins, plum_dens-dvalsf, c="b")
fig.savefig(f"{output_dir}density_plot.pdf")

# Construct basis
import os
import DREAMS_utils
import pyEXP
import yaml
import k3d
from gala.units import SimulationUnitSystem
import astropy.units as u

outputs = sorted(os.listdir(output_dir))
outputs = [o for o in outputs if ".npz" in o]

lims = 10

# Load position, velocity, and masses of particles at last snapshot
data_final = np.load(f"{output_dir}{outputs[-1]}", allow_pickle=True)

logr = np.log10(np.sqrt(np.sum(data_final["x"]**2, axis=1)))
masses = data_final["m"]

# 1 - Calculate base density
density_params_df = {"bins": 100,
                     "rangevals": [0,np.log10(lims)]}

rbins, dvals = DREAMS_utils.return_density(logr=logr, weights=masses, **density_params_df, smooth=True)

# 2 - Set up basis
# Create an EXP-compatible spherical basis function table 
model_file = f"{output_dir}basis_empirical_PlummerTest.txt" 
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
                            "rmin": 1,
                            "rmax": 100,
                            "Lmax": 6,
                            "nmax": 20,
                            "rmapping": 0.067,
                            "modelname": model_file,
                            "cachename": model_file.replace(".txt",".cache.run0")
                            }
            }


# Save yaml file for constructing gala potential
yaml_file = f"{output_dir}basis_yaml_PlummerTest.yml"

with open(yaml_file, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
    
    
# Construct basis
with open(yaml_file, "r") as f:
    yaml_config = f.read()

# Build the basis
basis = pyEXP.basis.Basis.factory(yaml_config) 

import astropy.units as u

# 3 - Calculate coefficients

t_outputs=[i*dt for i in range(0,nsteps+1, 20)]
coefs_container = None

for i,output_file in enumerate(outputs):
    
    data_file = np.load(f"{output_dir}{output_file}")

    coefs = basis.createFromArray(data_file["m"], 
                                  data_file["x"], 
                                  time=t_outputs[i])
    
    if coefs_container is None:
        coefs_container = pyEXP.coefs.Coefs.makecoefs(coefs)
        coefs_container.add(coefs)
    else:
        coefs_container.add(coefs)
        
# Save the coefficients
coefs_file = f"{output_dir}coefs_PlummerTest.h5"
if os.path.exists(coefs_file):
    os.remove(coefs_file)
    coefs_container.WriteH5Coefs(coefs_file) 
else:
    coefs_container.WriteH5Coefs(coefs_file)
    
    
# Build stream simulation
from gala.units import SimulationUnitSystem
import gala.potential as gp
import gala.dynamics as gd
import astropy.units as u
from gala.dynamics import mockstream as ms
import gala.integrate as gi
import math 

exp_units = SimulationUnitSystem(mass=10**10*u.Msun, # Can decide here how big your system is
                                 length=10*u.kpc, # Specify what's the scale radius
                                 G=1)

pot = gp.EXPPotential(units=exp_units,
                      config_file=yaml_file,
                      coef_file=coefs_file,
                      snapshot_time_unit=exp_units["time"])

pot_analytical = gp.PlummerPotential(m=M,
                                     b=a,
                                     units=exp_units)

# Orbits
v_xyz = data_file["v"]
x_lim = 5
v_abs = np.sqrt(np.sum(v_xyz**2,axis=1))
idx_list = np.random.randint(len(v_xyz), size=5)
fig2,axs = plt.subplots(1,2, layout="constrained", sharex=True, sharey=True)

pos = dict(zip(idx_list, [[] for i in range(len(idx_list))]))
vel = dict(zip(idx_list, [[] for i in range(len(idx_list))]))

for output_file in outputs:
    
    data_file = np.load(f"{output_dir}{output_file}")
    for idx in idx_list:
        x_idx = data_file["x"][idx]
        v_idx = data_file["v"][idx]
        pos[idx].append(x_idx)
        vel[idx].append(v_idx)

for idx in idx_list:
    p_xyz = np.vstack(pos[idx])
    axs[0].plot(p_xyz[:,0],p_xyz[:,1])
    axs[1].plot(p_xyz[:,0],p_xyz[:,2])
    
axs[0].set_aspect(1)
axs[1].set_aspect(1)
axs[0].set_xlim([-x_lim,x_lim])
axs[0].set_ylim([-x_lim,x_lim])
fig.savefig(f"{output_dir}orbits1.pdf")

for i,k in enumerate(pos.keys()):
    
    color = [np.random.uniform(), np.random.uniform(), np.random.uniform()]
    orbit_xyz = np.vstack(pos[k])
    
    orbit_x0 = pos[k][0]*exp_units["length"]
    orbit_v0 = vel[k][0]*exp_units["velocity"]
    orbit_w0 = gd.PhaseSpacePosition(pos=orbit_x0,
                                     vel=orbit_v0)

    orbit_r = np.sqrt(np.sum(np.power(orbit_x0,2)))

    orbit_period = 5*2*math.pi*orbit_r/pot_analytical.circular_velocity(orbit_x0)[0]
    
    if orbit_period.value >= max(t_outputs):
        continue 

    orbit_analytical = pot_analytical.integrate_orbit(w0=orbit_w0, 
                                                      Integrator=gi.LeapfrogIntegrator,
                                                      t1=0, 
                                                      t2=orbit_period, n_steps=1000)

    orbit_EXP = pot.integrate_orbit(w0=orbit_w0, 
                                    Integrator=gi.LeapfrogIntegrator, 
                                    t1=0, 
                                    t2=orbit_period, n_steps=1000)

    fig,ax = plt.subplots()
    ax.plot(orbit_analytical.xyz[0], orbit_analytical.xyz[1], c=color, ls="-")
    ax.plot(orbit_EXP.xyz[0], orbit_EXP.xyz[1], c=color, ls="--")
    ax.scatter(orbit_xyz[:,0], orbit_xyz[:,1], c=color, marker="^")
    fig.savefig(f"{output_dir}orbits_comp_{k}.pdf")
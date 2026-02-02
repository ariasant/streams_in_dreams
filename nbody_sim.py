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
dt = 0.01
nsteps = 10000
theta = 0.7 # aperture of tree for force calculation

output_dir = "/mnt/home/asante/ceph/PlummerNbody/N6/"


t_outputs=[i for i in range(0,nsteps)]

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
        t_outputs=t_outputs,
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
density_params_df = {"bins": 200,
                     "rangevals": [0.01,50]}

rbins, dvals = DREAMS_utils.return_density(r=np.sqrt(np.sum(x0**2,axis=1)), weights=masses, **density_params_df, smooth=True)
rbinsf, dvalsf = DREAMS_utils.return_density(r=np.sqrt(np.sum(xf**2,axis=1)), weights=masses, **density_params_df, smooth=True)

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


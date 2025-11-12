import numpy as np
from pytreegrav import Accel, Potential
import pickle
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
    vel[:] = vel + 0.5 * dt * accel  # note that you must slice arrays to modify them in place in the function!
    # then full-step drift
    pos[:] = pos + dt * vel
    # then recompute accelerations
    if option == 'tree':
        accel[:] = Accel(pos, masses, softening, parallel=True)
    if option == 'plum':
        accel[:] = plum_accel(pos, M, a, G)        
    # then another half-step kick
    vel[:] = vel + 0.5 * dt * accel
    return

def run_sim(nsteps, nsave, dt, x, v, masses, softening, option):

    if option == 'tree':
        accel = Accel(x, masses, softening, parallel=True, theta=theta)  # initialize acceleration
        pot0 = Potential(x, masses, softening, parallel=True, theta=theta)
    if option == 'plum':
        accel = plum_accel(x, M, a, G)
        r = np.sqrt(x[:,0]*x[:,0]+x[:,1]*x[:,1]+x[:,2]*x[:,2])
        pot0 = plum_pot(r, M, a, G) 
    
    K0 = 0.5*m1*np.sum(v**2,axis=1) 

    t = 0.  # initial time
    istep = 0
    Tmax = dt*float(nsteps)  # final/max time

# save full time series for subset of particles
    xsave = np.empty((nsteps+1,nsave,3))
    vsave = np.empty((nsteps+1,nsave,3))

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
        if option == 'plum':
            r = np.sqrt(x[:,0]*x[:,0]+x[:,1]*x[:,1]+x[:,2]*x[:,2])
            pot = plum_pot(r, M, a, G) 
            
        K = 0.5 * m1 * v2
        # Total kinetic and potential energy of the system
        ke = 0.5 * m1 * np.sum(v2)
        pe = 0.5 * m1 * np.sum(pot)
        #
        rave.append(np.mean(r))
        energies.append(ke+pe)
        vrat.append(-pe/2./ke)
        ts.append(t)
        # save particle subset
        xsave[istep,0:nsave,:]=x[0:nsave,:]
        vsave[istep,0:nsave,:]=v[0:nsave,:]

        # advance 1 step
        frog_step(dt, x, masses, softening, v, accel,option)
        t += dt
        istep += 1
    
    return rave, energies, vrat, ts, xsave, vsave, pot0, pot, K0, K
    
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

def shell_properties(N,nbin,x,v):

    r = np.sqrt(x[:,0]*x[:,0]+x[:,1]*x[:,1]+x[:,2]*x[:,2])
    vr = (x[:,0]*v[:,0]+x[:,1]*v[:,1]+x[:,2]*v[:,2])/r

# define parameters
    nshell = int(N/nbin)
# initialize variables
    vol0 = 0.
# set-up arrays
    r_shell = np.empty(nshell)
    n_shell = np.empty(nshell)
    sig_shell = np.empty(nshell)
    
# sort the particles
    isort = np.argsort(r)

# define shell properties
    for ishell in range(0,nshell):
        ibin = isort[nbin*ishell:nbin*(ishell+1)]
        # select particles in bin
        rbin = r[ibin]
        vbin = vr[ibin]
        #
        r_shell[ishell] = np.sum(rbin)/nbin
        #
        rmax = r[isort[(ishell+1)*nbin-1]]
        vol1 = rmax**3
        n_shell[ishell] = 1./(vol1 - vol0)
        vol0 = vol1
        #         
        v_shell = np.sum(vbin)/nbin
        sig_shell[ishell] = np.sqrt(np.sum((vbin - v_shell)**2)/nbin)
                                                                    
    n_shell = 3.*float(nbin)*n_shell/(4.*np.pi)
                 
    return r_shell,n_shell,sig_shell




# tree case - n = 100000.
run_opt = 'tree'

# parameters
M = 1.0
a = 1.0
G = 1.0
init_opt = 'disp'
# time
dt = 0.1  # adjust this to control integration error
nsteps = 1000

theta = 0.1

for exp_N in [4,5,6]:
    # set up
    N = 10**exp_N
    m1 = M /float(N)
    masses = np.full(N,m1)
    eps = 0.1
    softening = np.full(N,eps)
    nsave = 1000

    x0, v0 = generate_plummer(N,M,a,init_opt,G=1.0)

    x = x0[0:N,:]+0.
    v = v0[0:N,:]+0.
    v2 = (v[:,0]*v[:,0]+v[:,1]*v[:,1]+v[:,2]*v[:,2])

    # --- Run analytical profile (Plummer) ---
    r_an, e_an, vir_an, time_an, xsv_an, vsv_an, pot0_an, pot_an, KE0_an, KE_an = run_sim(
        nsteps, nsave, dt, x, v, masses, softening, "plum"
    )

    # --- Run N-body simulation ---
    r_nb, e_nb, vir_nb, time_nb, xsv_nb, vsv_nb, pot0_nb, pot_nb, KE0_nb, KE_nb = run_sim(
        nsteps, nsave, dt, x, v, masses, softening, run_opt
    )

    # --- Create Dictionaries ---
    ## 1. Analytical Profile Dictionary
    analytical_results = {
        'r': r_an,
        'e': e_an,
        'vir': vir_an,
        'time': time_an,
        'xsv': xsv_an,
        'vsv': vsv_an,
        'pot0': pot0_an,
        'pot': pot_an,
        'KE0': KE0_an,
        'KE': KE_an
    }

    ## 2. N-body Simulation Dictionary
    nbody_results = {
        'r': r_nb,
        'e': e_nb,
        'vir': vir_nb,
        'time': time_nb,
        'xsv': xsv_nb,
        'vsv': vsv_nb,
        'pot0': pot0_nb,
        'pot': pot_nb,
        'KE0': KE0_nb,
        'KE': KE_nb
    }

    # --- Save Dictionaries as Pickle Files ---
    def save_as_pickle(data_dict, filename):
        """Saves a Python dictionary as a pickle file."""
        # 'wb' mode means Write Binary, which is required for pickle
        try:
            with open(filename, 'wb') as file:
                pickle.dump(data_dict, file)
            print(f"Successfully saved data to **{filename}**")
        except Exception as e:
            print(f"An error occurred while saving {filename}: {e}")

    # Save the two dictionaries
    save_as_pickle(analytical_results, f'/mnt/home/asante/ceph/parc/analytical_results_theta_{theta}_N{exp_N}.pkl')
    save_as_pickle(nbody_results, f'/mnt/home/asante/ceph/parc/nbody_results_theta_{theta}_N{exp_N}.pkl')
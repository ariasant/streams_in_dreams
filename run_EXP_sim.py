from DREAMS_utils import makemodel
import numpy as np
import os
from ruamel.yaml import YAML
import shutil
import time

def plummer_density(radius,scale_radius=1.0,mass=1.0,astronomicalG=1.0):
        """basic plummer density profile"""
        return ((3.0*mass)/(4*np.pi))*(scale_radius**2.)*((scale_radius**2 + radius**2)**(-2.5))

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
  
def create_ics_yaml(
        output_filename,
        halofile1,
        disk_mass=0.01,
        ndisk=100,
        nhalo=1000,
        ToomreQ=1.8,               # Toomre Q value for disc (constant across disc)
        scale_length=0.01,              # Scale length of the realised disc
        scale_height=0.002
    ):
    """
    Generates a YAML configuration file from a template, filling in
    key simulation parameters.
    
    Args:
        output_filename (str): The name of the .yaml file to create (e.g., "my_run.yaml").
        HSCALE (float): Vertical scale length for disk basis construction (e.g. 0.002)
        SEED (int): Random number seed.
        ToomreQ (float): Toomre Q parameter for stellar disk generation (e.g. 1.4)
        cachefile (str): Name of EOF cache file (e.g. "eof.cache.fileF")
        disk_mass (float): Mass of stellar disk (e.g. 0.0125)
        halofile1 (str): File with input halo model (e.g., "my_halo.model")
        ndisk (int): Number of disk particles (e.g. 1000000)
        nhalo (int): Number of halo particles (e.g. 10000000)
        runtag (str): Label prefix for diagnostic images (e.g. "run1")
        scale_height (float): Scale height for disk realization (e.g. 0.002)
        scale_length (float): Scale length for disk realization (e.g. 0.01)
    """
    
    output_dir = os.path.dirname(output_filename) + "/"

    # This is the provided YAML template for the ICs
    yaml_template_dict = {
        # disk basis parameters
        "ASCALE"      : 0.01,              # Radial scale length for disk basis construction
        "HSCALE"      : 0.002,            # Vertical scale length for disk basis construction
        "cachefile"   : ".eof.cache.file",   # The cache file for the cylindrical basis
        "ignore"      : True,             # If true, will force remaking the disc basis
        # disk realization parameters
        "disk_mass"   : disk_mass,             # Disk mass
        "ndisk"       : ndisk,       # Number of disk particles
        "nhalo"       : nhalo,    # Number of halo particles
        "ToomreQ"     : ToomreQ,               # Toomre Q value for disc (constant across disc)
        "scale_length": scale_length,              # Scale length of the realised disc
        "scale_height": scale_height,            # Scale height of the realised disc

        # halo basis parameters
        "halofile1"   : halofile1 # Halo spherical model table
            }
    
    with open(output_filename, "w") as f:
        yaml.dump(yaml_template_dict, f)
       

def create_ics_file(m,x0,v0, output_dir):
    
    N = len(x0)
    
    with open(f"{output_dir}halo.bods","w", encoding="ascii") as f:
        
        f.write(f"{N}\t0\t0\n")
        for i in range(N):
            f.write(f"{m}\t{x0[i,0]}\t{x0[i,1]}\t{x0[i,2]}\t{v0[i,0]}\t{v0[i,1]}\t{v0[i,2]}\n")


    return     
    
yaml = YAML()
yaml.default_flow_style = False


def run_simulation(N: int, 
                   output_dir: str):
    
    M = 1.0
    
    start = time.time()

    # STEP 1: create density model assuming uniform density
    model_file = f"{output_dir}Nbody.model"

    plummer_b = 1.0
    R,D,M,P = makemodel(plummer_density,1.,
                        [plummer_b],
                        rvals = 10.**np.linspace(-3.,1.,2000),
                        pfile=model_file)


    # STEP 2: create body file with initial conditions of the simulation
    
    x0, v0 = generate_plummer(N,1.0,plummer_b,"disp",G=1.0)
    
    """ics_yaml_file = f"{output_dir}Nbody_ICs.yml"
     
    create_ics_yaml(output_filename=ics_yaml_file,
                    halofile1=model_file,
                    ndisk=0,
                    nhalo=N)

    os.system(f"mpirun /mnt/home/asante/streams_in_dreams/EXP/install/bin/gendisk --config {ics_yaml_file}")"""
    
    create_ics_file(m=M/N,
                    x0=x0,
                    v0=v0,
                    output_dir=output_dir)

    # Step 3: Run the simulation
    # Beware that if a disk component is present, than an extra preliminary step
    # needs to be performed to create "quiet" ICs

    # Create an EXP yaml file
    exp_yaml_dict = {
    "Global": {
        "outdir": f"{output_dir}",
        "nthrds": 1,
        "dtime": 0.005, # virial units, i.e. 1 is the period of an orbit at the virial radius [~Gyr]
        "ldlibdir": "/mnt/home/asante/streams_in_dreams/EXP/src/user",
        "runtag": "run1",
        "nsteps": 200, # i.e. 2 orbital periods at virial radius
        "multistep": 5,
        "infile": "OUT.run1.chkpt",
        "VERBOSE": 1
    },
    "Components": [
        {
            "name": "dark halo",
            "parameters": {
                "nlevel": 1,
                "indexing": True,
                "EJ": 2,
                "nEJkeep": 256,
                "EJdryrun": True
            },
            "bodyfile": f"{output_dir}halo.bods",
            "force": {
                "id": "direct"
            }
        },
    ],
    "Output": [ # nint specifies the number of time steps between running the output process
        {'id': 'outlog', 'parameters': {'nint': 1}}, # Global simulation properties
        {'id': 'outpsn', 'parameters': {'nint': 1}}, # Phase-space dumps
        ]
    }

    # Save the yaml file
    exp_yaml_file = f"{output_dir}Nbody_EXP.yml"
    with open(exp_yaml_file, "w") as f:
            yaml.dump(exp_yaml_dict, f)
            
            
    # Run the simulation
    os.system(f"mpirun -v /mnt/home/asante/streams_in_dreams/EXP/install/bin/exp --config {exp_yaml_file}")
    
    t = (time.time() - start) / 60
    
    print(f"Time taken: {t:.1f} minutes", flush=True)
    
    
for exp_N in [4,5,6]:
    
    N = int(10**exp_N)
    output_run = f"/mnt/home/asante/ceph/EXP_Nbody/n_particles_{exp_N}/"
    
    if os.path.exists(output_run):
        # Delete all the files in the folder
        shutil.rmtree(output_run)
        
    os.mkdir(output_run)
    
    print(f"N particles: {N} \noutput_dir: {output_run}", flush=True)
    print("="*100)
    run_simulation(N=N,
                   output_dir=output_run)
    print("#"*100, flush=True)

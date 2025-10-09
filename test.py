import astropy.units as u
import astropy.coordinates as coord
import EXP4DREAMS
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic
import matplotlib.pyplot as plt
import numpy as np



# Create simulation object
MW_sim = EXP4DREAMS.DREAMSMW(box=34,
                             snap_path="/mnt/home/dreams/ceph/Sims/CDM/MW_zooms/SB5/",
                             group_path="/mnt/home/dreams/ceph/FOF_Subfind/CDM/MW_zooms/SB5/")


# Make EXP BFE approximation
basis_dict = {# PartType : basis params 
              1: {"Lmax": 3, "nmax": 10}}
EXP_gen = EXP4DREAMS.EXPBFE_builder(sim=MW_sim,
                                    basis_params_dict=basis_dict,
                                    snapshots = [61, 67, 73, 81, 90], # snapshots from z=1 to z=0)
                                    output_dir="/mnt/home/asante/ceph/parc/")

# Plot projections
for field in ["dens", "dens m=0", "dens m>0", "potl"]:
    fig = EXP_gen.surface_projection(EXP_gen.basis[1],
                                    EXP_gen.coefs[1],
                                    field="dens m=0", #potl
                                    time=EXP_gen.coefs[1].Times()[-1],
                                    extent=[[-1,-1,0],[1,1,0]],  # x y z lims
                                    grid=[100,100,0]) 
    plt.show()


# Build gala potential
pot, exp_units = EXP_gen.build_gala_potential()

# Integrate Pal5-like orbit
pal5_w0 = gd.PhaseSpacePosition(pos=[20., 0, 10.] * u.kpc, 
                                vel=([0, 50., 0] * u.km/u.s) )
orbit = pot.integrate_orbit(
    pal5_w0, dt=1*u.Myr, t1=9.*u.Gyr, t2=13*u.Gyr, Integrator=gi.DOPRI853Integrator
)
orbit.plot()

# Integrate solar orbit
orbit = pot.integrate_orbit(
    gd.PhaseSpacePosition(pos=[8., 0, 0.] * u.kpc, 
                          vel=[100, 0., 0] * u.km/u.s) , 
    dt=1*u.Myr, 
    t1=9.*u.Gyr, 
    t2=13*u.Gyr, 
    Integrator=gi.DOPRI853Integrator
)
orbit.plot()
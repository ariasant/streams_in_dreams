import sys
sys.path.append("/mnt/home/asante/ceph/streams_in_dreams")
import EXP4DREAMS
import astropy.units as u
import pyEXP

import matplotlib.pyplot as plt
import numpy as np
import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic
import astropy.coordinates as coord
import gala.integrate as gi


# Create simulation object
MW_sim = EXP4DREAMS.DREAMSMW(box=695,
                             snap_path="/mnt/home/dreams/ceph/Sims/CDM/MW_zooms/SB5/",
                             group_path="/mnt/home/dreams/ceph/FOF_Subfind/CDM/MW_zooms/SB5/")


# Create EXPpotential
# 695
basis_dict = {# PartType : basis params 
              1: {"Lmax": 6, "nmax": 10, "numr": 4000}}
density_dict = {1: {"bins": 400, 
                    "rangevals": [0, 2.5] # log10 scale
                    }
                }
EXP_gen = EXP4DREAMS.EXPBFE_builder(sim=MW_sim,
                                    basis_params_dict=basis_dict,
                                    density_dict=density_dict,
                                    snapshots = [90], # snapshots from z=1 to z=0)
                                    output_dir="/mnt/home/asante/ceph/parc/")

density_dict = {1: {"bins": 400, 
                    "rangevals": [1, 2.5] # log10 scale
                    }
                }
EXP_gen_2 = EXP4DREAMS.EXPBFE_builder(sim=MW_sim,
                                      basis_params_dict=basis_dict,
                                      density_dict=density_dict,
                                      snapshots = [90], # snapshots from z=1 to z=0)
                                      output_dir="/mnt/home/asante/ceph/parc/mod2/")


basis_dict = {# PartType : basis params 
              1: {"Lmax": 10, "nmax": 48, "numr": 4000, "rmapping": 7}}
density_dict = {1: {"bins": 400, 
                    "rangevals": [np.log10(0.3), np.log10(400)] # log10 scale
                    }
                }

EXP_gen_JH = EXP4DREAMS.EXPBFE_builder(sim=MW_sim,
                                      basis_params_dict=basis_dict,
                                      density_dict=density_dict,
                                      snapshots = [90], # snapshots from z=1 to z=0)
                                      output_dir="/mnt/home/asante/ceph/parc/JH/")



for bfe in [EXP_gen, EXP_gen_2, EXP_gen_JH]:

    fig = bfe.surface_projection(bfe.basis[1],
                                 bfe.coefs[1],
                                 field="dens", #potl
                                 time=bfe.coefs[1].Times()[-1],
                                 extent=[[-1,-1,0],[1,1,0]],  # x y z lims
                                 grid=[100,100,0]) 
    
    fig.savefig(f"{bfe.__output_dir__}exp_dens.pdf")



    fig,axs = plt.subplots(1,2,figsize=(10,5), sharex=True, sharey=True)

    plot = bfe.plot_field(basis=bfe.basis[1],
                          coefs=bfe.coefs[1].getCoefStruct(bfe.coefs[1].Times()[-1]),
                          field="dens",
                          lim=1000,
                          level=0,
                          ax=axs[0],
                          projection="faceon",
                          norm="log")

    plot = bfe.plot_field(basis=bfe.basis[1],
                          coefs=bfe.coefs[1].getCoefStruct(bfe.coefs[1].Times()[-1]),
                          field="dens",
                          lim=1000,
                          level=0,
                          ax=axs[1],
                          projection="edgeon",
                          norm="log")

    cbar = fig.colorbar(plot,ax=axs[:], orientation="horizontal")

    fig.savefig(f"{bfe.__output_dir__}my_dens.pdf")

    pot, exp_units = bfe.build_gala_potential()

    v_circ = pot.circular_velocity([8.0,0,0]*u.kpc, t=13*u.Gyr).to(u.km / u.s)
    pot_sun = pot.energy([8.0,0,0]*u.kpc, t=13*u.Gyr).to((u.km/u.s)**2)
    acc = pot.acceleration([8.0,0,0]*u.kpc, t=13*u.Gyr).to((u.km/(u.s*u.Gyr)))
    print(v_circ, flush=True)
    print(pot_sun, flush=True)
    print(acc, flush=True)



    pal5_w0 = gd.PhaseSpacePosition(pos=[20., 0, 10.] * u.kpc, 
                                    vel=[0, 50., 0] * u.km/u.s)

    orbit = pot.integrate_orbit(
        pal5_w0, 
        dt=100*u.Myr, 
        t1=9.*u.Gyr, 
        t2=13*u.Gyr, 
        Integrator=gi.DOPRI853Integrator
    )
    orbit.plot()

    plt.savefig(f"{bfe.__output_dir__}orbit.pdf")

    print("#"*100, flush=True)
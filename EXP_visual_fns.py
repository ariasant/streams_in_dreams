import astropy.units as u
import corner
from DREAMS_utils import return_density
import k3d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
import numpy as np
import pyEXP




def shell_average(basis, 
                  coefs,
                  r_min: float, 
                  field: str = None, 
                  n_points=1000):
        
    # Load basis coefficients
    basis.set_coefs(coefs)

    theta = np.arccos(np.random.uniform(-1, 1, size=n_points))
    phi = np.random.uniform(0, 2*np.pi, size=n_points)
    r = r_min

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    sph_avg = np.mean([basis.getFields(xi, yi, zi) for xi, yi, zi in zip(x, y, z)], axis=0)

    output_dict = dict(zip(basis.getFieldLabels(), sph_avg))

    if field is None:
        return output_dict
    else:
        return output_dict[field]
        
        
        
def plot_reconstructed_density_profile(r,
                                       mass,
                                       basis, 
                                       coefs
                                       ):
        
    rbins, dvals = return_density(r=r,
                                  weights=mass, 
                                  bins=300,
                                  rangevals=[0.1,300]) 
    
    # Get density contribution from different m values
    exp_dens, exp_densm0, exp_densml0 = [],[],[]
    rbins_exp = np.zeros(len(rbins)+1)
    rbins_exp[1:] = rbins
    for i in range(len(rbins_exp)-1):
        out_dict = shell_average(field="dens", 
                                 basis=basis,
                                 coefs=coefs,
                                 r_min=rbins_exp[i])
        exp_dens.append(out_dict["dens"])
        exp_densm0.append(out_dict["dens m=0"])
        exp_densml0.append(out_dict["dens m>0"])
    exp_dens = np.array(exp_dens)
    
    fig,ax = plt.subplots(figsize=(4,4))
    ax.plot(rbins, dvals, label="Data")
    ax.plot(rbins, exp_dens, label="EXP")
    ax.plot(rbins, exp_densm0, label="EXP m=0")
    ax.plot(rbins, exp_densml0, label="EXP m>0")
    ax.set_yscale("log")
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("Density [$\\rm{M}_{\\odot}/\\rm{kpc}^3$]")
    ax.set_xscale("log")
    ax.legend()
    
    return fig,ax
    
    
    
def plot_2D_integrated_field(basis,
                             coefs,
                             field: str,
                             x_bins: np.array,
                             y_bins: np.array,
                             z_bins: np.array,
                             ax: plt.axes,
                             integrate_over: str = "z",
                             **kwargs):
    
    basis.set_coefs(coefs)

    # Understand which field to plot
    field_labels = basis.getFieldLabels()
    idx = field_labels.index(field)


    # Understand which dimension to integrate on
    if integrate_over=="z":
        # Create array to hold the 2D field projection
        exp_2d = np.zeros((len(x_bins),len(y_bins)))
        dz = z_bins[1] - z_bins[0]
        extent = [x_bins[0], x_bins[1], y_bins[0], y_bins[1]]
        for i,xi in enumerate(x_bins):
            for j, yj in enumerate(y_bins):
                field_sum = 0
                for zi in z_bins:
                    outs = basis.getFields(xi, yj, zi)
                    field_sum += outs[idx]*dz
                exp_2d[i,j] = field_sum

    elif integrate_over=="y":
        exp_2d = np.zeros((len(x_bins),len(z_bins)))
        dy = y_bins[1] - y_bins[0]
        extent = [x_bins[0], x_bins[1], z_bins[0], z_bins[1]]
        for i,xi in enumerate(x_bins):
            for j, zj in enumerate(z_bins):
                field_sum = 0
                for yi in y_bins:
                    outs = basis.getFields(xi, yi, zj)
                    field_sum += outs[idx]*dy
                exp_2d[i,j] = field_sum

    elif integrate_over=="x":
        exp_2d = np.zeros((len(y_bins),len(z_bins)))
        dx = x_bins[1] - x_bins[0]
        extent = [y_bins[0], y_bins[1], z_bins[0], z_bins[1]]
        for i,yi in enumerate(y_bins):
            for j, zj in enumerate(z_bins):
                field_sum = 0
                for xi in x_bins:
                    outs = basis.getFields(xi, yi, zj)
                    field_sum += outs[idx]*dx
                exp_2d[i,j] = field_sum
    

    # Plot the 2D projection
    plot = ax.imshow(exp_2d, origin="lower",
                        extent=extent, **kwargs)

    return plot
   
    
    
def surface_projection(basis,
                       coefs,
                       exp_units,
                       field: str, # dens, dens m=0, dens m>0, potl, potl m-0, ...
                       time: float,
                       extent: list, # e.g. [[xmin, ymin, 0.],[xmax, ymax, 0.]]
                       grid: list, # [bins_x, bins_y, 0.],
                       ax=None,
                       circles=False):
    
    # Initialise surface field generator
    times = coefs.Times()

    generator = pyEXP.field.FieldGenerator(times, 
                                            [el.to(exp_units["length"]).value if el!=0 else 0. for el in extent[0]], 
                                            [el.to(exp_units["length"]).value if el!=0 else 0. for el in extent[1]], 
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

    if field in ["dens", "dens m=0", "dens m>0"]:

        # Convert to M_sun / kpc^2
        surface = surface*(exp_units["mass"]/exp_units["length"]**2).to(u.Msun / u.pc**2)
        surface = np.log10(surface)
            
        cbar_label = "$\\log_{10}(\\Sigma) \\; [\\rm{M}_{\\odot} \\, \\rm{pc}^{-2}]$"

            
    if field in ["potl", "potl m=0", "potl m>0"]:
        # Convert to (km/s)^2
        surface = surface*(exp_units["length"]**2 / exp_units["time"]**2).to(u.km**2 / u.s**2)
        cbar_label = "$\\Phi \\; [\\rm{km}^2 \\, \\rm{s}^{-2}]$"

    
    if ax is None:
        
        fig, ax = plt.subplots()
        ax.set_xlim([min(x.value),max(x.value)])
        ax.set_ylim([min(y.value), max(y.value)])
        cbar_label = field
    
    cont1 = ax.contour(xv, yv, surface, colors='k')
    cont1.clabel(fontsize=9, inline=True)
    cont2 = ax.contourf(xv, yv, surface)
    
    if ax is None:
        cbar = fig.colorbar(cont2)
        cbar.set_label(cbar_label)
    
    if circles:
        # Plot circles showing 10 and 100 kpc
        for r in [10, 100]:
            circle = patches.Circle(
                        (0,0), # center coordinates
                        r,  # Radius
                        color="red",
                        fill=False, 
                        linewidth=2,
                        linestyle='-'
                    )
            ax.add_patch(circle)
        
    

    return ax



def volume_render(basis,
                  coefs,
                  time: float, 
                  field: str,
                  grid_lim: int = 100,
                  n_points: int = 100):

    # Calculate 3D distribution of fields
    times = coefs.Times()
    pmin  = [-grid_lim, -grid_lim, -grid_lim]
    pmax  = [grid_lim, grid_lim, grid_lim]
    grid  = [n_points, n_points, n_points]

    generator = pyEXP.field.FieldGenerator(times, pmin, pmax, grid)
    volumes = generator.volumes(basis, coefs)

    volume = volumes[time][field]
    
    if field=="potl":
        volume=np.abs(volume)

    # Initialise plot
    plot = k3d.plot()

    value_range = [np.percentile(volume, 5), np.percentile(volume, 95)]
    size = [-grid_lim, grid_lim, -grid_lim, grid_lim, -grid_lim, grid_lim]

    volume = k3d.volume(volume.astype(np.float32), 
                        alpha_coef=5,
                        color_range=value_range,  
                        color_map=k3d.matplotlib_color_maps.Viridis, 
                        compression_level=7)
    
    volume.transform.bounds = [-size[0], size[0], -size[1], size[1], -size[2], size[2]]

    plot += volume

    return plot
    
   
    
def plot_coefs_evolution(coefs, 
                         l, 
                         m,
                         ax):

    coefs_values = coefs.getAllCoefs()
    times = coefs.Times()
    
    spherical_index = (l * (l + 1)) // 2 + m
    n_radial_terms = coefs_values.shape[1]

    cmap = mpl.colormaps["inferno"]
    norm = mpl.colors.Normalize(vmin=0, vmax=n_radial_terms)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    for n_radial in range(n_radial_terms):
        ax.plot(times, 
                coefs_values[spherical_index, n_radial, :],
                c=cmap(norm(n_radial)))
        
    ax.set_yscale("log")
    ax.set_ylabel("Coefficients amplitude")
    ax.set_xlabel("Time")

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Radial order")

    ax.set_title(f"l={l}, m={m}")
    return ax
    
 
   
def plot_SNR_mesh(SNR_mesh,
                  ax,
                  vmin=1e-3,
                  vmax=1e3):
    
    plot = ax.pcolormesh(SNR_mesh, 
                        norm="log", 
                        cmap="RdYlBu",
                        vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(plot,ax=ax)
    cbar.set_label("SNR")

    ax.set_xlabel("Radial nth mode")
    ax.set_ylabel("Angular (l,m) mode")
    ax.set_aspect(SNR_mesh.shape[1]/SNR_mesh.shape[0])

    return ax
 

    
def plot_acceleration_field_xy(pot,
                               r_vir,
                               t=13*u.Gyr,
                               grid_lim: float = 1.,
                               n_points: int = 100,
                               z_level: float = 0.0):
    
    x = np.linspace(-grid_lim, grid_lim, n_points)
    y = np.linspace(-grid_lim, grid_lim, n_points)

    Fx = np.zeros((n_points,n_points))
    Fy = np.zeros((n_points,n_points))

    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            pos = [xi,yj,z_level]*u.kpc
            acc = pot.acceleration(pos, t=t).to(u.km/u.s/u.Gyr).value
            Fx[i,j] = acc[0][0]
            Fy[i,j] = acc[1][0]

    fig,axs = plt.subplots(1,2, sharex=True, sharey=True, layout="constrained")

    axs[0].pcolormesh(x,y, Fx, 
                    norm=mpl.colors.Normalize(vmin=-1000, vmax=1000),
                    cmap="seismic")
    plot = axs[1].pcolormesh(x,y, Fy, 
                    norm=mpl.colors.Normalize(vmin=-1000, vmax=1000),
                    cmap="seismic")

    cbar = plt.colorbar(plot, ax=axs[:], orientation="horizontal", shrink=0.8)
    cbar.set_label("Acceleration $[\\rm{km} \, \\rm{s}^{-1} \, \\rm{Gyr}^{-1}]$")
    axs[0].set_xlabel("x [kpc]")
    axs[1].set_xlabel("x [kpc]")
    axs[0].set_ylabel("y [kpc]")
    axs[0].set_aspect(1)
    axs[1].set_aspect(1)
    fig.suptitle(f"z = {z_level} [kpc]")

    axs[0].set_title("$a_{X}$")
    axs[1].set_title("$a_{Y}$")
    
    
    for ax in axs:
        # Plot circles showing 10 and 100 kpc
        r_10kpc = 10 / r_vir
        r_100kpc = 100 / r_vir
        for r in [r_10kpc, r_100kpc]:
            circle = patches.Circle(
                        (0,0), # center coordinates
                        r,  # Radius
                        color="red",
                        fill=False, 
                        linewidth=2,
                        linestyle='-'
                    )
            ax.add_patch(circle)
    
    
    return fig,axs



def plot_acceleration_field_xz(pot,
                               r_vir,
                               t=13*u.Gyr,
                               grid_lim: float = 1.,
                               n_points: int = 100,
                               y_level: float = 0.0):
    

    x = np.linspace(-grid_lim, grid_lim, n_points)
    z = np.linspace(-grid_lim, grid_lim, n_points)

    Fx = np.zeros((n_points,n_points))
    Fz = np.zeros((n_points,n_points))

    for i, xi in enumerate(x):
        for j, zj in enumerate(z):
            pos = [xi,y_level,zj]*u.kpc
            acc = pot.acceleration(pos, t=t).to(u.km/u.s/u.Gyr).value
            Fx[i,j] = acc[0][0]
            Fz[i,j] = acc[1][0]

    fig,axs = plt.subplots(1,2, sharex=True, sharey=True, layout="constrained")

    axs[0].pcolormesh(x,y_level, Fx, 
                    norm=mpl.colors.Normalize(vmin=-1000, vmax=1000),
                    cmap="seismic")
    plot = axs[1].pcolormesh(x,y_level, Fz, 
                    norm=mpl.colors.Normalize(vmin=-1000, vmax=1000),
                    cmap="seismic")

    cbar = plt.colorbar(plot, ax=axs[:], orientation="horizontal", shrink=0.8)
    cbar.set_label("Acceleration $[\\rm{km} \, \\rm{s}^{-1} \, \\rm{Gyr}^{-1}]$")
    axs[0].set_xlabel("x [kpc]")
    axs[1].set_xlabel("x [kpc]")
    axs[0].set_ylabel("z [kpc]")
    axs[0].set_aspect(1)
    axs[1].set_aspect(1)
    fig.suptitle(f"y = {y_level} [R_vir]")

    axs[0].set_title("$a_{X}$")
    axs[1].set_title("$a_{Z}$")
    
    for ax in axs:
        # Plot circles showing 10 and 100 kpc
        r_10kpc = 10 / r_vir
        r_100kpc = 100 / r_vir
        for r in [r_10kpc, r_100kpc]:
            circle = patches.Circle(
                        (0,0), # center coordinates
                        r,  # Radius
                        color="red",
                        fill=False, 
                        linewidth=2,
                        linestyle='-'
                    )
            ax.add_patch(circle)
    
    return fig,axs
 
 

#### ORBIT RECONSTRUCTION 

def plot_colored_line(ax, x, y, c, cmap='viridis', linewidth=2, **kwargs):
    """
    Plots a line with a color gradient based on a third variable 'c'.
    
    Parameters
    ----------
    ax : Matplotlib Axes
        The axes to plot on.
    x, y : array-like
        The x and y coordinates of the data points.
    c : array-like
        The color values (e.g., time) that determine the color of each segment.
    cmap : str, optional
        The colormap to use (default is 'viridis').
    linewidth : int, optional
        The width of the line (default is 2).
    """
    # Create segments by pairing up consecutive points
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map the 'c' values to colors
    norm = mpl.colors.Normalize(c.min(), c.max())
    
    # Create the LineCollection
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, **kwargs)
    
    # Set the color array
    # We use the 'c' values for the first point of each segment
    lc.set_array(c[:-1]) 
    
    # Add the collection to the axes and get the minimum/maximum limits
    ax.add_collection(lc)
    ax.autoscale_view() # Adjust the plot limits to encompass the new line
    
    return lc # Return the LineCollection object to use with a Colorbar 
 
 
 
def plot_orbit_reconstruction(pid,
                              sim_tracks,
                              bfe_tracks,
                              axs,
                              cmap="plasma"):
    
    # Extract data, converting to numpy array if it has a '.value' attribute
    times = bfe_tracks[pid]["times"].value
    x_bfe = bfe_tracks[pid]["xyz"][0]
    y_bfe = bfe_tracks[pid]["xyz"][1]
    z_bfe = bfe_tracks[pid]["xyz"][2]
    
    #  Find plotting limits based on the combined range of both tracks
    x_min = min(np.concatenate([x_bfe, sim_tracks[pid]["xyz"][0]]))
    x_max = max(np.concatenate([x_bfe, sim_tracks[pid]["xyz"][0]]))
    y_min = min(np.concatenate([y_bfe, sim_tracks[pid]["xyz"][1]]))
    y_max = max(np.concatenate([y_bfe, sim_tracks[pid]["xyz"][1]]))
    z_min = min(np.concatenate([z_bfe, sim_tracks[pid]["xyz"][2]]))
    z_max = max(np.concatenate([z_bfe, sim_tracks[pid]["xyz"][2]]))   
    
    min_val = min([x_min.value, y_min.value, z_min.value])
    max_val = max([x_max.value, y_max.value, z_max.value])
    
    sim_times = [t.value for t in sim_tracks[pid]["times"]]
    
    # Plot the simulation positions (unchanged)
    axs[0].scatter(sim_tracks[pid]["xyz"][0], 
                   sim_tracks[pid]["xyz"][1].value, 
                   c=sim_times, 
                   norm=mpl.colors.Normalize(times.min(), times.max()),
                   edgecolors="k",
                   cmap=cmap,
                   s=30, label='Simulation', zorder=100)
    axs[1].scatter(sim_tracks[pid]["xyz"][0], 
                   sim_tracks[pid]["xyz"][2].value, 
                   c=sim_times, 
                   norm=mpl.colors.Normalize(times.min(), times.max()),
                   edgecolors="k",
                   cmap=cmap,
                   s=30, label='Simulation', zorder=100)
    axs[2].scatter(sim_tracks[pid]["xyz"][1], 
                   sim_tracks[pid]["xyz"][2].value, 
                   c=sim_times, 
                   norm=mpl.colors.Normalize(times.min(), times.max()),
                   edgecolors="k",
                   cmap=cmap,
                   s=30, label='Simulation', zorder=100)
    
    # Plot the orbit reconstruction using the multicolored line function
    lc0 = plot_colored_line(axs[0], x_bfe, y_bfe, times, cmap=cmap, linewidth=1)
    lc1 = plot_colored_line(axs[1], x_bfe, z_bfe, times, cmap=cmap, linewidth=1)
    lc2 = plot_colored_line(axs[2], y_bfe, z_bfe, times, cmap=cmap, linewidth=1)
    
    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
    # Add colorbar for the time dimension
    plt.colorbar(lc0, ax=axs, label='Time [Gyr]', shrink=0.5) 
    
    return



def compare_distributions(data1, data2, labels=None, title=None):
    """
    Overlays two multidimensional distributions on a corner plot.
    
    Parameters:
    -----------
    data1 : np.ndarray
        First distribution (N_samples, N_dim).
    data2 : np.ndarray
        Second distribution (N_samples, N_dim).
    labels : list of str, optional
        Labels for the dimensions (e.g., ['x', 'y', 'z', 'vx']).
    title : str, optional
        Title for the figure.
    """
    # 1. Define colors and range
    # Get the combined range to ensure axes are identical for both
    # We compute min/max across both datasets for each dimension
    mins = np.min([data1.min(axis=0), data2.min(axis=0)], axis=0)
    maxs = np.max([data1.max(axis=0), data2.max(axis=0)], axis=0)
    # Add a small buffer (5%)
    ranges = [[mn - 0.05*(mx-mn), mx + 0.05*(mx-mn)] for mn, mx in zip(mins, maxs)]

    # 2. Plot the first distribution (Blue)
    # We save the figure object to pass it to the second call
    figure = corner.corner(data1, 
                           labels=labels,
                           range=ranges,
                           color='tab:blue', 
                           smooth=1.0,           # Smooths contours slightly
                           plot_datapoints=False, # Don't plot scatter points if N is large
                           plot_density=False,    # Don't fill contours (optional style)
                           fill_contours=True,
                           hist_kwargs={'density': True, 'linewidth': 2},
                           alpha=0.4)            # Transparency for fill

    # 3. Overplot the second distribution (Orange) on the SAME figure
    corner.corner(data2, 
                  fig=figure,                # Pass the existing figure
                  range=ranges,              # MUST use same ranges
                  color='tab:orange',
                  smooth=1.0,
                  plot_datapoints=False,
                  plot_density=False,
                  fill_contours=True,
                  hist_kwargs={'density': True, 'linewidth': 2},
                  alpha=0.4)

    # 4. Add Legend (Manual, since corner doesn't support easy auto-legends)
    blue_line = mlines.Line2D([], [], color='tab:blue', label='Dist 1')
    orange_line = mlines.Line2D([], [], color='tab:orange', label='Dist 2')
    plt.legend(handles=[blue_line, orange_line], 
               bbox_to_anchor=(0., 1.0, 1., .0), 
               loc="upper right",
               fontsize=12)
    
    if title:
        plt.suptitle(title, fontsize=16)

    return figure

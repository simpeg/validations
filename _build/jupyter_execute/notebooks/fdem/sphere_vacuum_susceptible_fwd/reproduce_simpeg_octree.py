#!/usr/bin/env python
# coding: utf-8

# Reproduce: SimPEG OcTree
# =======================
# 
# ## Simulating Secondary Magnetic Field Data over a Conductive and Susceptible Sphere
# 
# Secondary magnetic fields are simulated over conductive sphere in a vacuum. The sphere has a conductivity of $\sigma$ = 0.25 S/m and a magnetic susceptibility of $\chi$ = 4 SI. The center of the sphere is located at (0,0,-50) and has a radius of $a$ = 8 m. The background conductivity is defined as 1e-8 S/m. 
# 
# Secondary magnetic fields are simulated for an x, y and z oriented magnetic dipoles sources at (-5,0,10). The x, y and z components of the response are simulated for each source at (5,0,10). We plot only the horizontal coaxial, horizontal coplanar and vertical coplanar data.

# ## SimPEG Package Details
# 
# **Link to the docstrings for the simulation.** The docstrings will have a citation and show the integral equation.

# ## Reproducing the Forward Simulation Result
# 
# We begin by loading all necessary packages and setting any global parameters for the notebook.

# In[1]:


from SimPEG import dask
from discretize.utils import mkvc, refine_tree_xyz
from discretize import CylMesh, TreeMesh
import SimPEG.electromagnetics.frequency_domain as fdem
from SimPEG import maps
from SimPEG.utils import model_builder

from pymatsolver import Pardiso

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import mu_0

mpl.rcParams.update({"font.size": 10})
write_output = True


# A compressed folder containing the assets required to run the notebook is then downloaded. This includes mesh and model files for the forward simulation.

# In[2]:


# Download .tar files


# Extracted files are then loaded into the SimPEG framework.

# In[3]:


rootdir = './../../../assets/fdem/sphere_vacuum_susceptible_fwd_simpeg/'
meshfile = rootdir + 'octree_mesh.txt'
conmodelfile = rootdir + 'model.con'
susmodelfile = rootdir + 'model.sus'

mesh = TreeMesh.readUBC(meshfile)
sigma_model = TreeMesh.readModelUBC(mesh, conmodelfile)
chi_model = TreeMesh.readModelUBC(mesh, susmodelfile)


# Here, we define the survey geometry for the forward simulation.

# In[4]:


xyz_tx = np.c_[-5., 0., 10.]         # Transmitter location
xyz_rx = np.c_[5., 0., 10.]          # Receiver location
frequencies = np.logspace(2,5,10)    # Frequencies

source_list = []

for ii in range(0, len(frequencies)):

    receivers_list = [
        fdem.receivers.PointMagneticFluxDensitySecondary(xyz_rx, "x", "real"),
        fdem.receivers.PointMagneticFluxDensitySecondary(xyz_rx, "x", "imag"),
        fdem.receivers.PointMagneticFluxDensitySecondary(xyz_rx, "y", "real"),
        fdem.receivers.PointMagneticFluxDensitySecondary(xyz_rx, "y", "imag"),
        fdem.receivers.PointMagneticFluxDensitySecondary(xyz_rx, "z", "real"),
        fdem.receivers.PointMagneticFluxDensitySecondary(xyz_rx, "z", "imag")
        ]

    for jj in ['x','y','z']:
        source_list.append(
            fdem.sources.MagDipole(receivers_list, frequencies[ii], mkvc(xyz_tx), orientation=jj)
        )

survey = fdem.Survey(source_list)


# Below, we plot the discretization and conductivity model used in the forward simulation.

# In[5]:


fig = plt.figure(figsize=(11, 8))
ind_active = mesh.cell_centers[:, 2] < 0
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
log_model = np.log10(sigma_model[ind_active])

ax1 = fig.add_axes([0.14, 0.1, 0.6, 0.85])
mesh.plot_slice(
    plotting_map * log_model, normal="Y", ax=ax1, ind=int(mesh.hy.size / 2),
    clim=(np.min(log_model), np.max(log_model)), grid=True
)

ax1.set_xlim([-50, 50])
ax1.set_ylim([-80, 20])
ax1.set_title("OcTree Conductivity Model: {} Cells".format(mesh.nC))

ax2 = fig.add_axes([0.76, 0.1, 0.05, 0.85])
norm = mpl.colors.Normalize(
    vmin=np.min(log_model), vmax=np.max(log_model)
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", format="$10^{%.1f}$"
)
cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)


# Below, we plot the discretization and susceptibility model used in the forward simulation.

# In[6]:


fig = plt.figure(figsize=(11, 8))
plotting_chi_model = chi_model[ind_active]

ax1 = fig.add_axes([0.14, 0.1, 0.6, 0.85])
mesh.plot_slice(
    plotting_map * plotting_chi_model,
    normal="Y", ax=ax1, ind=int(mesh.hy.size / 2),
    clim=(np.min(chi_model), np.max(chi_model)), grid=True,
    pcolorOpts={'cmap':'plasma'}
)

ax1.set_xlim([-50, 50])
ax1.set_ylim([-80, 20])
ax1.set_title("OcTree Susceptibility Model: {} Cells".format(mesh.nC))

ax2 = fig.add_axes([0.76, 0.1, 0.05, 0.85])
norm = mpl.colors.Normalize(
    vmin=np.min(chi_model), vmax=np.max(chi_model)
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.plasma
)
cbar.set_label("Susceptibility [SI]", rotation=270, labelpad=15, size=12)


# Here we define the mapping from the model to the mesh, extract the survey from the data object and define the forward simulation.

# In[7]:


mu0 = 4*np.pi*1e-7
mu_model = mu0 * (1 + chi_model)

sigma_map = maps.IdentityMap()

sim = fdem.simulation.Simulation3DMagneticFluxDensity(
    mesh, survey=survey, sigmaMap=sigma_map, verbose=True, solver=Pardiso, forward_only=True
)

sim.mu = mu_model


# Finally, we predict the secondary magnetic field data for the model provided.

# In[8]:


Hs_octree = mu_0**-1 * sim.dpred(sigma_model)

Hs_octree = Hs_octree[0::2] + 1.j*Hs_octree[1::2]
Hs_octree = np.reshape(Hs_octree, (len(frequencies), 3, 3))
Hs_octree = [Hs_octree[:, 0, :], Hs_octree[:, 1, :], Hs_octree[:, 2, :]]


# If desired, we can export the data to a simple text file.

# In[9]:


if write_output:
    
    fname_octree = rootdir + 'dpred_octree.txt'
    
    header = 'FREQUENCY HX_REAL HX_IMAG HY_REAL HY_IMAG HZ_REAL HZ_IMAG'
    f_column = np.kron(np.ones(3), frequencies)
    
    out_array = np.vstack(Hs_octree)
    out_array = np.c_[
        f_column,
        np.real(out_array[:, 0]),
        np.imag(out_array[:, 0]),
        np.real(out_array[:, 1]),
        np.imag(out_array[:, 1]),
        np.real(out_array[:, 2]),
        np.imag(out_array[:, 2])
    ]

    fid = open(fname_octree, 'w')
    np.savetxt(fid, out_array, fmt='%.6e', delimiter=' ', header=header)
    fid.close()


# ## Simulated Data Plot
# 
# Here, we plot only the horizontal coaxial, horizontal coplanar and vertical coplanar data.

# In[10]:


fig = plt.figure(figsize=(16, 7))
lw = 2
ms = 6

ax = 3*[None]

legend_str = ['Real', 'Imag']

for ii, src in enumerate(['X','Y','Z']):
    
        ax[ii] = fig.add_axes([0.05 + 0.3*ii, 0.1, 0.25, 0.8])
        ax[ii].semilogx(frequencies, np.real(Hs_octree[ii][:, ii]), 'r-o', lw=lw, markersize=ms)
        ax[ii].semilogx(frequencies, np.imag(Hs_octree[ii][:, ii]), 'r--s', lw=lw, markersize=ms)
        ax[ii].grid()
        ax[ii].set_xlabel('Frequency (Hz)')
        ax[ii].set_ylabel('Secondary field (A/m)')
        ax[ii].set_title(src + ' dipole source, ' + src + ' component')
        ax[ii].legend(legend_str)


# In[ ]:





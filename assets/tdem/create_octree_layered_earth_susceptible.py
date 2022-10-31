#%% IMPORT PACKAGES
from discretize.utils import mkvc, refine_tree_xyz
from discretize import CylMesh, TreeMesh
import SimPEG.electromagnetics.time_domain as tdem
import matplotlib.pyplot as plt
from SimPEG import maps
from SimPEG.utils import model_builder
import numpy as np
import os

#%% SURVEY PARAMETERS
xyz_tx = np.c_[0., 0., 5.]         # Transmitter location
xyz_rx = np.c_[10., 0., 5.]         # Receiver location
times = np.logspace(-5,-2,10)       # Times channels
time_steps = [(5e-07, 40), (2.5e-06, 40), (1e-5, 40), (5e-5, 40), (2.5e-4, 40)]

#%% MODEL PARAMETERS
thicknesses = np.r_[64., 64]         # thicknesses
sigma = np.r_[0.01, 0.01, 0.01]       # conductivity
chi = np.r_[0., 9., 0.]              # susceptibility

mu0 = 4*np.pi*1e-7
min_diff_dist = np.sqrt(2*times.min() / np.max(mu0 * (1+chi) * sigma) )
max_diff_dist = np.sqrt(2*times.max() / np.min(mu0 * (1+chi) * sigma) )

print('Minimum Diffusion Distance: {} m'.format(min_diff_dist))
print('Maximum Diffusion Distance: {} m'.format(max_diff_dist))


#%% CREATE TREE MESH
phi = np.linspace(0, 2*np.pi, 17)
phi[-1] = phi[0]

dh = 0.5
L = 3000.0
nbc = 2 ** int(np.round(np.log(L / dh) / np.log(2.0)))
h = [(dh, nbc)]
mesh = TreeMesh([h, h, h], x0="CCC")

pts = np.r_[xyz_tx, xyz_rx]
mesh = refine_tree_xyz(
    mesh, pts, octree_levels=[4, 4, 4, 4, 4, 4], method="radial", finalize=False
)

xp, yp, zp = np.meshgrid(np.linspace(-5, 15, 3), np.linspace(-5, 5, 3), np.array([0.]))
pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
mesh = refine_tree_xyz(
    mesh, pts,
    octree_levels=[4, 6, 6, 8, 10, 8], 
    octree_levels_padding=[4, 6, 6, 8, 10, 8],
    method="surface", finalize=False
)

mesh.finalize()

#%% REGULARIZE MESH

mesh.write_UBC('octree_mesh_unregularized.txt')

os.system('regularizeOctreeMesh.exe octree_mesh_unregularized.txt octree_mesh.txt')

mesh = TreeMesh.read_UBC('./octree_mesh.txt')
print('Number of cells: {}'.format(mesh.nC))

#%% DEFINE CONDUCTIVITY MODEL
ind_active = mesh.cell_centers[:, 2] < 0

sigma_model = 1e-8 * np.ones(mesh.nC)
chi_model = np.zeros(mesh.nC)

sigma_model[mesh.cell_centers[:,2]<0] = sigma[-1]
chi_model[mesh.cell_centers[:,2]<0] = chi[-1]

k = (mesh.cell_centers[:, 2] < 0) & (mesh.cell_centers[:, 2] > -np.sum(thicknesses))
sigma_model[k] = sigma[1]
chi_model[k] = chi[1]

k = (mesh.cell_centers[:, 2] < 0) & (mesh.cell_centers[:, 2] > -thicknesses[0])
sigma_model[k] = sigma[0]
chi_model[k] = chi[0]

mesh.write_model_UBC('model.con', sigma_model)
mesh.write_model_UBC('model.sus', chi_model)

#%% PLOT MODEL

fig = plt.figure(figsize=(10,4))

ax1 = fig.add_axes([0.14, 0.1, 0.6, 0.85])

mesh.plot_slice(
    sigma_model,
    normal="Y", ax=ax1, ind=int(mesh.hy.size / 2), grid=True
)

ax1.set_xlim([-800, 800])
ax1.set_ylim([-800, 50])
ax1.set_title("OcTree Conductivity Model: {} Cells".format(mesh.nC))

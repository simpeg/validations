#%% IMPORT PACKAGES
from discretize.utils import mkvc, refine_tree_xyz, ndgrid
from discretize import CylMesh, TreeMesh
import SimPEG.electromagnetics.time_domain as tdem
import matplotlib.pyplot as plt
from SimPEG import maps
from SimPEG.utils import model_builder
import numpy as np
import os

#%% SURVEY PARAMETERS
xyz_tx = np.c_[-5., 0., 10.]        # Transmitter location
xyz_rx = np.c_[5., 0., 10.]         # Receiver location
times = np.logspace(-5,-2,10)       # Times channels
time_steps = [(5e-07, 40), (2.5e-06, 40), (1e-5, 40), (5e-5, 40), (2.5e-4, 40)]

#%% MODEL PARAMETERS
a = 8                                # radius of sphere
sigma0 = 1e-8                        # background conductivity
sigma = 1e2                          # electrical conductivity of sphere
chi = 0                              # susceptibility of sphere
xyzs = np.c_[0., 0., -50.]           # xyz location of the sphere

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
    mesh, pts, octree_levels=[4, 4, 4, 4, 4], method="radial", finalize=False
)

theta = np.linspace(-np.pi/2, np.pi/2, 21)
phi, theta = np.meshgrid(phi, theta)
phi = mkvc(phi)
theta = mkvc(theta)

pts = np.c_[
    xyzs[0, 0] + a*np.cos(theta)*np.cos(phi),
    xyzs[0, 1] + a*np.cos(theta)*np.sin(phi),
    xyzs[0, 2] + a*np.sin(theta)
]
mesh = refine_tree_xyz(
    mesh, pts, octree_levels=[4, 4, 4], method="radial", finalize=False
)

pts = ndgrid(np.r_[-5, 5], np.r_[-5, 5], 0.)
mesh = refine_tree_xyz(
    mesh, pts,
    octree_levels=[0, 0, 0, 0, 10, 4, 4], 
    octree_levels_padding=[0, 0, 0, 0, 10, 4, 4],
    max_distance=2., method="surface", finalize=False
)

mesh.finalize()

#%% REGULARIZE MESH

mesh.write_UBC('octree_mesh_unregularized.txt')

os.system('regularizeOctreeMesh.exe octree_mesh_unregularized.txt octree_mesh.txt')

mesh = TreeMesh.read_UBC('./octree_mesh.txt')
print('Number of cells: {}'.format(mesh.nC))

#%% DEFINE CONDUCTIVITY MODEL
ind_active = mesh.cell_centers[:, 2] < 0

sigma_model = sigma0 * np.ones(mesh.nC)
ind_sphere = model_builder.getIndicesSphere(mkvc(xyzs), a, mesh.cell_centers)
sigma_model[ind_sphere] = sigma

mesh.write_model_UBC('model.con', sigma_model)

#%% PLOT MODEL

fig = plt.figure(figsize=(10,4))

ax1 = fig.add_axes([0.14, 0.1, 0.6, 0.85])

mesh.plot_slice(
    sigma_model,
    normal="Y", ax=ax1, ind=int(mesh.hy.size / 2), grid=True
)

ax1.set_xlim([-300, 300])
ax1.set_ylim([-250, 50])
ax1.set_title("OcTree Conductivity Model: {} Cells".format(mesh.nC))

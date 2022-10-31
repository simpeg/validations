# IMPORT PACKAGES
from discretize.utils import mkvc, refine_tree_xyz, ndgrid
from discretize import CylMesh, TreeMesh
import SimPEG.electromagnetics.frequency_domain as fdem
from SimPEG import maps
from SimPEG.utils import model_builder
import numpy as np
import os

# SURVEY PARAMETERS
xyz_tx = np.c_[-5., 0., 10.]         # Transmitter location
xyz_rx = np.c_[5., 0., 10.]          # Receiver location
frequencies = np.logspace(2,5,10)    # Frequencies
tx_moment = 1.                       # Dipole moment of the transmitter

# MODEL PARAMETERS
a = 8                                # radius of sphere
sig0 = 1e-8                          # background conductivity
sig = 0.25                           # electrical conductivity of sphere
chi = 4                              # relative permeability of sphere
xyzs = np.c_[0., 0., -50.]           # xyz location of the sphere

mu0 = 4*np.pi*1e-7
min_skin_depth = (np.pi * mu0 * (1+chi) * sig * frequencies.min())**-0.5
max_skin_depth = (np.pi * mu0 * (1+chi) * sig * frequencies.max())**-0.5

print('Minimum Skin Depth: {} m'.format(min_skin_depth))
print('Maximum Skin Depth: {} m'.format(max_skin_depth))

#%% CREATE TREE MESH
phi = np.linspace(0, 2*np.pi, 17)
phi[-1] = phi[0]

dh = 0.25
L = 1000.0
nbc = 2 ** int(np.round(np.log(L / dh) / np.log(2.0)))
h = [(dh, nbc)]
mesh = TreeMesh([h, h, h], x0="CCC")

pts = np.r_[xyz_tx, xyz_rx]
mesh = refine_tree_xyz(
    mesh, pts, octree_levels=[4, 4, 4, 4, 2, 4], method="radial", finalize=False
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
    mesh, pts, octree_levels=[0, 4, 4, 4], method="radial", finalize=False
)

pts = ndgrid(np.r_[-10, 10], np.r_[-5, 5], 0.)
mesh = refine_tree_xyz(
    mesh, pts,
    octree_levels=[0, 0, 0, 0, 0, 12, 4, 4], 
    octree_levels_padding=[0, 0, 0, 0, 0, 12, 4, 4],
    max_distance=2., method="surface", finalize=False
)

mesh.finalize()

mesh.write_UBC('octree_mesh_unregularized.txt')


# REGULARIZE MESH
os.system('regularizeOctreeMesh.exe octree_mesh_unregularized.txt octree_mesh.txt')



mesh = TreeMesh.read_UBC('./octree_mesh.txt')
print('Number of cells: {}'.format(mesh.nC))



# DEFINE CONDUCTIVITY AND SUSCEPTIBILITY MODEL
ind_active = mesh.cell_centers[:, 2] < 0

sigma_model = sig0 * np.ones(mesh.nC)
ind_sphere = model_builder.getIndicesSphere(mkvc(xyzs), a, mesh.cell_centers)
sigma_model[ind_sphere] = sig

chi_model = np.zeros(mesh.nC)
chi_model[ind_sphere] = chi

mesh.write_model_UBC('model.con', sigma_model)
mesh.write_model_UBC('model.sus', chi_model)


# IMPORT PACKAGES
from discretize.utils import mkvc, refine_tree_xyz
from discretize import CylMesh, TreeMesh
import SimPEG.electromagnetics.frequency_domain as fdem
from SimPEG import maps
from SimPEG.utils import model_builder
import numpy as np
import os

# SURVEY PARAMETERS
xyz_tx = np.c_[0., 0., 5.]           # Transmitter location
xyz_rx = np.c_[10., 0., 5.]          # Receiver location
frequencies = np.logspace(2,5,10)    # Frequencies
tx_moment = 1.                       # Dipole moment of the transmitter

# MODEL PARAMETERS
thicknesses = np.r_[64., 64]         # thicknesses
sigma = np.r_[0.05, 0.5, 0.05]       # conductivity

mu0 = 4*np.pi*1e-7
min_skin_depth = (np.pi * mu0 * sigma.max() * frequencies.min())**-0.5
max_skin_depth = (np.pi * mu0 * sigma.min() * frequencies.max())**-0.5

print('Minimum Skin Depth: {} m'.format(min_skin_depth))
print('Maximum Skin Depth: {} m'.format(max_skin_depth))

# CREATE TREE MESH
dh = 0.25
L = 1000.0
nbc = 2 ** int(np.round(np.log(L / dh) / np.log(2.0)))
h = [(dh, nbc)]
mesh = TreeMesh([h, h, h], x0="CCC")

pts = np.r_[xyz_tx, xyz_rx]
mesh = refine_tree_xyz(
    mesh, pts, octree_levels=[4, 4, 6, 4, 4, 4], method="radial", finalize=False
)

xp, yp, zp = np.meshgrid(np.linspace(-5, 15, 3), np.linspace(-5, 5, 3), np.array([0.]))
pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
mesh = refine_tree_xyz(
    mesh, pts,
    octree_levels=[0, 4, 8, 8, 8, 8, 8], 
    octree_levels_padding=[0, 8, 8, 8, 8, 8, 8],
    method="surface", finalize=False
)

mesh.finalize()

mesh.write_UBC('octree_mesh_unregularized.txt')


# REGULARIZE MESH
os.system('regularizeOctreeMesh.exe octree_mesh_unregularized.txt octree_mesh.txt')



mesh = TreeMesh.read_UBC('./octree_mesh.txt')
print('Number of cells: {}'.format(mesh.nC))

# DEFINE CONDUCTIVITY MODEL
ind_active = mesh.cell_centers[:, 2] < 0

sigma_model = 1e-8 * np.ones(mesh.nC)

sigma_model[mesh.cell_centers[:,2]<0] = sigma[-1]

k = (mesh.cell_centers[:, 2] < 0) & (mesh.cell_centers[:, 2] > -np.sum(thicknesses))
sigma_model[k] = sigma[1]

k = (mesh.cell_centers[:, 2] < 0) & (mesh.cell_centers[:, 2] > -thicknesses[0])
sigma_model[k] = sigma[0]

mesh.write_model_UBC('model.con', sigma_model)

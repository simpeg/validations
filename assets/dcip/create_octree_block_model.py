# IMPORT PACKAGES
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG import maps, data
from SimPEG.utils import model_builder
from SimPEG.utils.io_utils.io_utils_electromagnetics import write_dcipoctree_ubc
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.static_utils import generate_dcip_sources_line

# SURVEY PARAMETERS
survey_type = "pole-dipole"
data_type = "volt"
dimension_type = "3D"
station_separation = 100.0
num_rx_per_src = 8
northing = np.linspace(-1000, 1000, 9)


# MODEL PARAMETERS
background_cond = 1e-2
conductor_cond = 1e-1
resistor_cond = 1e-3

background_charge = 1e-6
conductor_charge = 1e-1
resistor_charge = 0.

# DEFINE SOURCE
source_list = []
for ii in range(0, len(northing)):

    end_locations = np.r_[-1000., 1000., northing[ii], northing[ii]]

    source_list += generate_dcip_sources_line(
        survey_type,
        data_type,
        dimension_type,
        end_locations,
        0,
        num_rx_per_src,
        station_separation,
    )

survey = dc.survey.Survey(source_list)
data_object = data.Data(survey=survey)

fname = "./survey.loc"
write_dcipoctree_ubc(
    fname,
    data_object,
    'volt',
    'survey',
    format_type="general",
    comment_lines=""
)



# CREATE TREE MESH
dh = 25.0  # base cell width
dom_width_x = 10000.  # domain width x
dom_width_y = 10000.  # domain width y
dom_width_z = 6000.   # domain width z
nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))  # num. base cells x
nbcy = 2 ** int(np.round(np.log(dom_width_y / dh) / np.log(2.0)))  # num. base cells y
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))  # num. base cells z

# Define the base mesh
hx = [(dh, nbcx)]
hy = [(dh, nbcy)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hy, hz], x0="CCN")

# Mesh refinement near sources and receivers.
electrode_locations = np.r_[
    survey.locations_a,
    survey.locations_b,
    survey.locations_m,
    survey.locations_n,
]
unique_locations = np.unique(electrode_locations, axis=0)
mesh = refine_tree_xyz(
    mesh, unique_locations, octree_levels=[10, 8, 4, 4], method="radial", finalize=False
)

# Finalize the mesh
mesh.finalize()

mesh.write_UBC('octree_mesh_unregularized.txt')


# REGULARIZE MESH
os.system('regularizeOctreeMesh.exe octree_mesh_unregularized.txt octree_mesh.txt')



mesh = TreeMesh.read_UBC('./octree_mesh.txt')
print('Number of cells: {}'.format(mesh.nC))

# DEFINE CONDUCTIVITY MODEL
conductivity_model = background_cond * np.ones(mesh.nC)
chargeability_model = background_charge * np.ones(mesh.nC)

ind_conductor = model_builder.getIndicesBlock(
    np.r_[-600., -800., -450.], np.r_[-200., 800., -150.], mesh.cell_centers
)
conductivity_model[ind_conductor] = conductor_cond
chargeability_model[ind_conductor] = conductor_charge

ind_resistor = model_builder.getIndicesBlock(
    np.r_[200., -800., -450.], np.r_[600., 800., -150.], mesh.cell_centers
)
conductivity_model[ind_resistor] = resistor_cond
chargeability_model[ind_resistor] = resistor_charge

mesh.write_model_UBC('true_model.con', conductivity_model)
mesh.write_model_UBC('true_model.chg', chargeability_model)

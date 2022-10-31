# IMPORT PACKAGES
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG import maps, data
from SimPEG.utils import model_builder
from SimPEG.utils.io_utils.io_utils_electromagnetics import read_dcipoctree_ubc, write_dcipoctree_ubc
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.static_utils import generate_dcip_sources_line


rootdir = './block_model_ip_fwd_dcipoctree/'

dc_data = read_dcipoctree_ubc(rootdir+'data_dc.txt', 'volt')
ip_data = read_dcipoctree_ubc(rootdir+'data_ip.txt', 'apparent_chargeability')

rootdir = './block_model_dc_inv_dcoctree/'
np.random.seed(83)
standard_deviation = 1e-6 + 0.05*np.abs(dc_data.dobs)
dobs = dc_data.dobs + standard_deviation * np.random.rand(len(standard_deviation))
dc_data.dobs = dobs
dc_data.standard_deviation = standard_deviation
outname = rootdir + 'dc_data.dobs'
write_dcipoctree_ubc(outname, dc_data, 'volt', 'dobs')

rootdir = './block_model_ip_inv_ipoctree/'
np.random.seed(83)
standard_deviation = 0.001
dobs = ip_data.dobs + standard_deviation * np.random.rand(len(ip_data.dobs))
ip_data.dobs = dobs
ip_data.standard_deviation = standard_deviation
outname = rootdir + 'ip_data.dobs'
write_dcipoctree_ubc(outname, ip_data, 'apparent_chargeability', 'dobs')

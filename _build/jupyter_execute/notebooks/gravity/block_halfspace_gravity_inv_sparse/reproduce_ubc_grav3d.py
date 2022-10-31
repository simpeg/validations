#!/usr/bin/env python
# coding: utf-8

# Reproduce: GRAV3D v6.0.1
# =======================
# 
# ## Inverting Gravity Data Over a Block in a Halfspace: Compact and Blocky Model
# 
# Here, we invert gravity anomaly data collected over a block within a homogeneous halfspace. We invert for a compact and blocky model using an iteratively re-weighted least-squares inversion approach.
# 
# The true model consists of a denser block (0.1 $g/cm^3$) within a halfspace (0 $g/cm^3$).
# The dimensions of the block in the x, y and z directions are all 200 m. The block is buried at a depth of 200 m.
# 
# The data being inverted were generated using the [UBC-GIF GRAV3D v6.0.1 code](https://grav3d.readthedocs.io/en/v6.0/). Synthetic gravity data were simulated at a heigh 1 m above the surface within a 1000 m by 1000 m region; the center of which lied directly over the center of the block. Gaussian noise with a standard deviation of 0.002 mGal were added to the synthetic data. Uncertainties of 0.002 mGal were assigned to the data before inverting.

# ## UBC-GIF GRAV3D v6.0.1 Package Details
# 
# [GRAV3D v6.0.1](https://grav3d.readthedocs.io/en/v6.0/) is a voxel cell gravity forward modeling and inversion package developed by the UBC Geophysical Inversion Facility. This software is proprietary and can ONLY be acquired through appropriate [academic](https://gif.eos.ubc.ca/software/licenses_academic) or [commerical](https://gif.eos.ubc.ca/software/licenses_commercial) licenses. The numerical approach of the forward simulation is described in the [online manual's theory section](https://grav3d.readthedocs.io/en/v6.0/content/theory.html). If you have a valid license, the steps for reproducing the simulating results are explained below. 

# ## Running the Inversion

# ### Step 1: Acquiring a commercial or academic license.
# 
# - Explain and link to where you would get licenses. Make ABSOLUTELY CLEAR we don't just give out the licenses willy-nilly.

# ### Step 2: Downloading and extracting assets
# 
# - Link and instructions for download

# ### Step 3: Running the forward modeling executable
# 
# - Brief description of what files are needed
# - Link to the manual page where we run the forward simulation

# In[ ]:





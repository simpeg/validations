#!/usr/bin/env python
# coding: utf-8

# Reproduce: MAG3D v6.0.1
# ===============================
# 
# ## Inverting TMI Data Over a Susceptible Block in a Halfspace: Smoothest Least-Squares
# 
# Here, we inverted total magnetic intensity (TMI) data collected over a block within a homogeneous halfspace. We invert for a compact and blocky model using an iteratively re-weighted least-squares inversion approach. The problem is bounded to enforce positivity in the recovered susceptibility model.
# 
# The true model consists of a susceptible block (0.025 SI) within a minimally susceptible halfspace (0.0001 SI).
# The dimensions of the block in the x, y and z directions were are all 200 m. The block was buried at a depth of 200 m. The Earth's inducing field had an inclination of 65 degrees, a declination of 25 degrees and an intensity of 50,000 nT.
# 
# The data being inverted were generated using the [UBC MAG3D v6.0.1 code](https://mag3d.readthedocs.io/en/v6/). Synthetic TMI data were simulated within a 1000 m by 1000 m region at an elevation of 30 m; the center of which lied directly over the center of the block. The station spacing was 50 m in both the X and Y directions. Observed data were sythnetically created by adding Gaussian noise with a standard deviation of 1 nT to the simulated data. A floor uncertainty of 1 nT was assigned to the observed data.

# ## UBC-GIF MAG3D v6.0 Package Details
# 
# [MAG3D v6.0.1](https://mag3d.readthedocs.io/en/v6/) is a voxel cell magnetic forward modeling and inversion package developed by the UBC Geophysical Inversion Facility. This software is proprietary and can ONLY be acquired through appropriate [academic](https://gif.eos.ubc.ca/software/licenses_academic) or [commerical](https://gif.eos.ubc.ca/software/licenses_commercial) licenses. The numerical approach of the forward simulation is described in the [online manual's theory section](https://mag3d.readthedocs.io/en/v6/content/theory.html). If you have a valid license, there are instructions for reproducing the results (add link). 

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





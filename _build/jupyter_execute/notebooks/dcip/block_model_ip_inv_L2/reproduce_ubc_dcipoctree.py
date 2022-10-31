#!/usr/bin/env python
# coding: utf-8

# Reproduce: UBC DCIP OcTree
# =========================
# 
# ## Inverting Pole-Dipole IP Data over a Conductive (and Chargeable) and a Resistive Block
# 
# Here, we invert pole-dipole IP data collected over a conductive and a resistive block. The conductive block is also chargeable. We invert for an intrinsic chargeability model using a least-squares inversion approach.
# 
# For the true model, the background conductivity $\sigma_0$ = 0.01 S/m. The conductor has a conductivity of $\sigma_c$ = 0.1 S/m and an intrinsic chargeability of 0.1 V/V. The resistor has a conductivity of $\sigma_r$ = 0.001 S/m and is non-chargeable. Both blocks are oriented along the Northing direction and have x, y and z dimensions of 400 m, 1600 m and 320 m. Both blocks are buried at a depth of 160 m.
# 
# The data being inverted were generated using the [UBC-GIF DCIP OcTree](https://dcipoctree.readthedocs.io/en/latest/). Synthetic apparent chargeability data were simulated with a pole-dipole configuration. The survey consisted of 9 West-East survey lines, each with a length of 2000 m. The line spacing was 250 m and the electrode spacing was 100 m. Gaussian noise with a standard deviation of 1e-6 V + 5% the absolute value were added to each datum. Uncertainties of 1e-6 V + 5% were assigned to the data for inversion.

# **UBC-GIF DCIP OcTree:** [DCIP OcTree](https://dcipoctree.readthedocs.io/en/latest/) is a voxel cell DC/IP forward modeling and inversion package developed by the UBC Geophysical Inversion Facility. This software is proprietary and can ONLY be acquired through appropriate [commerical](https://gif.eos.ubc.ca/software/licenses_commercial) licenses. The numerical approach of the forward simulation is described in the [online manual's theory section](https://dcipoctree.readthedocs.io/en/latest/content/theory.html). If you have a valid license, there are instructions for reproducing the results (add link).

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





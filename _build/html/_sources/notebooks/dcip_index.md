DC / IP
=======

In this chapter, we publish code comparisons and validations for direct current resistivity (DC) and induced polarization (IP) modeling packages.
In SimPEG the *SimPEG.electromagnetics.static.resistivity* module is used for modeling DC resistivity data, while the *SimPEG.electromagnetics.static.induced_polarization* module is used for modeling IP data.

```{figure} ../assets/section_images/dcip_physics.PNG
:width: 800px
:align: center

Schematic illustrating the physics of the DC/IP method ([image source](https://em.geosci.xyz/content/geophysical_surveys/ip/physics.html))
```

**Content:**

* [Forward simulation of DC data for blocks in a halfspace](./dcip/block_model_dc_fwd/code_comparison.ipynb)
* [Forward simulation of IP data for blocks in a halfspace](./dcip/block_model_ip_fwd/code_comparison.ipynb)
* [Least-squares inversion of DC data to recover blocks in a halfspace](./dcip/block_model_dc_inv_L2/code_comparison.ipynb)
* [Least-squares inversion of IP data to recover blocks in a halfspace](./dcip/block_model_ip_inv_L2/code_comparison.ipynb)



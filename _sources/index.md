SimPEG Code Validations Book
============================

The *SimPEG Code Validations Book* is a space for publishing notebooks used to validate the SimPEG coding package. The goal of this project is to assess the accuracy and benchmark the performance of SimPEG against analytic solutions and other coding packages.


```{figure} ./assets/section_images/title_image.PNG
:width: 600px
:align: center
```


Project Summary
---------------

Geophysical forward simulation and inversion algorithms have proven to be powerful tools for characterizing subsurface geologies. However the accuracy, efficiency, computational cost and stability of these algorithms depend on many factors; e.g. the numerical formulation used to define the physics and the numerical grid (mesh) on which the formulation is discretized. Prior to practical implementation, it is therefore beneficial to have confidence in an algorithm and to know how various factors influence its output. Here, we discuss a public code validation project for SimPEG; an open-source Python-based package for simulation and parameter estimation in geophysics. This project aims to promote continual improvement to SimPEG’s code base and increase the efficacy of practitioners in using SimPEG to solve applied problems.

For each geophysical method (e.g. gravity, DC-resistivity, time/frequency domain electromagnetics), we use SimPEG to generate forward modeling or inversion outputs for a given geological scenario. Each output is compared to an analytic solution (when available) and/or outputs from other coding packages. Using this approach, we can determine whether SimPEG’s algorithm works as expected, expose how different implementation choices impact the algorithm’s output, and whether improvements to the algorithm are necessary prior to practical implementation. For most geophysical methods available within SimPEG, forward simulation algorithms have been validated against UBC-GIF coding packages; as UBC-GIF codes have been used extensively within industry for many years.

To increase efficacy amongst geophysical practitioners and generate feedback, we are publishing our code validation project publicly as a Jupyter-book website; where the instructions and resources required to reproduce all results are made available for download. Automated testing of the website’s underlying notebooks will be run periodically to ensure validations are up to date with SimPEG’s latest software release. Preliminary validations have already resulted in notable improvements to SimPEG’s gravity, magnetics and DC/IP forward simulations; with more improvements expected as the project matures.

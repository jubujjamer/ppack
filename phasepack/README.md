# Phasepack for Python

This is a python implementation of the Phasepack library (https://www.cs.umd.edu/~tomg/projects/phasepack/).
The Phasepack module is a very complete set of methods for the phase retrieval problem, wich arise in many physical applications. It  was originally written in MATLAB and primarily devised to easily benchmark many state of the art phase retrieval algorithms.
This is a work in progress and just a few of the methods are currently implemented, but it's structure will easily allow the translation of the missing ones. The name of the files and methods were mantained, just changed to underscore notation and phasepack has become a package with initializers and solvers as submodules. Fineup's algorithm and Truncated Wirtinger Flow method are working, along with a few test problems to test them. 

## Some details

The original package used function handlers to perform least squares, eigenvectors and matrix vector products in an efficient way. In a similar fashion, this version uses scipy's LinearOperators, which allow the same operations to be performed with the same philosophy.

## Dependencies
Run this module using a conda instalation of numpy and scipy (for efficiency sake, don't use pip). The scikit-image module and imageio should also be installed, as long as numba.
```
conda install numpy scipy scikit-image numba imageio
```
After this, you can run `image_reonstruction.py` and `signal_reconstruction.py` from the 'examples' folder to test all is working ok.

## Who created Phasepack?

Rohan Chandra - University of Maryland 
Ziyuan Zhong - Columbia University 
Justin Hontz - University of Maryland 
Val McCulloch - Smith College

…and faculty advisors
Christoph Studer - Cornell University 
Tom Goldstein - University of Maryland 

## Who am I?

Juan M. Bujjamer - University of Buenos Aires

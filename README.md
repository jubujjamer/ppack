# Phasepack for Python

This is a python implementation of the Phasepack library (https://www.cs.umd.edu/~tomg/projects/phasepack/).

The Phasepack module is a very complete set of methods for the phase retrieval problem, which arise in many physical applications. It  was originally written in MATLAB and primarily devised to easily benchmark many state of the art phase retrieval algorithms.

This is a work in progress and just a few of the methods are currently implemented, but it's structure will easily allow the translation of the missing ones. The name of the files and methods were roughly kept the same, just changed to underscore notation and some names were pythonized. Also, phasepack has become a package with initializers and solvers as submodules.

Up to now Fineup's algorithm and Truncated Wirtinger Flow method are working, along with a few test problems to test them, the rest of the modules will be soon implemented.

## Some details

The original package used function handlers to perform least squares, eigenvectors and matrix vector products in an efficient way. In a similar fashion, this version uses scipy's LinearOperators, which allow the same operations to be performed with the same philosophy. Results and options have been placed in containers along with some methods to manage them and generic math operations and algorithms lie in its own file.

## Dependencies
Run this module using a binary installation of numpy and scipy (for efficiency sake, don't use pip). It's recommended to create a new conda environment with it's clean dependencies

```
conda create -n phasepack-env python=3.6 numpy scipy scikit-image imageio numba
```
And you'll be free to test the examples `image_reconstruction.py` and `signal_reconstruction.py` after activating the environment.

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

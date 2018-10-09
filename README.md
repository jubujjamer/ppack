# Phasepack for Python

This is a python implementation of the Phasepack library (https://www.cs.umd.edu/~tomg/projects/phasepack/).
The Phasepack module is a great and very complete set of methods for the phase retrieval problem, wich arise in many physical applications. It  was originally written in MATLAB and primarily deviced to easily benchmark many state of the art phase retrieval algorithms.
This is a work in progress and just a few of the methods are currently implemented, but it's structure will easily allow the translation of the missing ones. 

# Some details

The original package used function handlers to perform least squares and matrix vector products in an efficient way. In a similar fashion, this version uses scipy's LinearOperators, which allow the same operations to be performed with the same philosophy.

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

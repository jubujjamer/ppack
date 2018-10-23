"""
Module ppck.
Phasepack implementation for python for phase retrieval applications.

Based on MATLAB implementation by Rohan Chandra, Ziyuan Zhong, Justin Hontz,
Val McCulloch, Christoph Studer & Tom Goldstein.
Copyright (c) University of Maryland, 2017.
Python version of the phasepack module by Juan M. Bujjamer.
University of Buenos Aires, 2018.
"""
# It is required to have a binary installation of numpy and scipy.
# pip installation is too slow.
try:
    import numpy
except:
    raise ImportError('Module numpy is not installed.\n\
    Please run \'conda install numpy\' from a conda environment.')

try:
    import scipy
except:
    raise ImportError('Module scipy is not installed.\n\
    Please run \'conda install scipy\' from a conda environment.')

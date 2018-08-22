# runInitSpectral.py
#
# This test file implements the Spectral as well as the Truncated Spectral
# initializer. The code builds a synthetic formulation of the Phase
# Retrieval problem, b = |Ax| and computes an estimate to x. The code
# finally outputs the correlation achieved.
#
# PAPER TITLE:
#              Phase Retrieval via Wirtinger Flow: Theory and Algorithms.
#
# ARXIV LINK:
#              https://arxiv.org/abs/1407.1065
#
#
# This is a test script for the spectral initializer.
#
# 1.) Each test script for an initializer starts out by defining the length
# of the unknown signal, n and the number of measurements, m. These
# mesurements can be complex by setting the isComplex flag to be true.
#
# 2.) We then build the test problem by generating random gaussian
# measurements and using b0 = abs(Ax) where A is the measurement matrix.
#
# 3.) We run x0 = initSpectral(A,[],b0,n,isTruncated,isScaled) which runs
# the initialier and and recovers the test vector with high correlation.
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017
# Translated into python by Juan M. Bujjamer

import numpy as np
from numpy.random import randn
from numpy.linalg import norm
from phasepack.util import Options, ConvMatrix
from phasepack.initializers import initSpectral
from numpy.random import multivariate_normal as mvnrnd


# Parameters
n = 500            # number of unknowns
m = 8*n            # number of measurements
isComplex = True   # use complex matrices? or just stick to real?

# Build the test problem
xt = randn(n, 1) + isComplex*randn(n,1)*1j # true solution
A = randn(m, n) + isComplex*randn(m, n)*1j # matrix
A = ConvMatrix(A)
b0 = abs(A*xt) # data
# Set up Parameters
isTruncated = True
isScaled = True

# Invoke the truncated spectral initial method
x0 = initSpectral(A=A, b0=b0, n=n, isTruncated=isTruncated, isScaled=isScaled)
# Calculate the correlation between the recovered signal and the true signal
correlation = np.abs(x0.conjugate().T@xt/norm(x0)/norm(xt))
print('correlation: %.3f\n' % correlation)

#  runSignalReconstructionDemo.py
#
# This script will create phaseless measurements from a 1d test signal, and
# then recover the image using phase retrieval methods.  We now describe
# the details of the simple recovery problem that this script implements.
#
#                         Recovery Problem
# This script creates a complex-valued random Gaussian signal. Measurements
# of the signal are then obtained by applying a linear operator to the
# signal, and computing the magnitude (i.e., removing the phase) of
# the results.
#
#                       Measurement Operator
# Measurement are obtained using a linear operator, called 'A', that
# contains random Gaussian entries.
#
#                      The Recovery Algorithm
# The image is recovered by calling the method 'solvePhaseRetrieval', and
# handing the measurement operator and linear measurements in as arguments.
# A struct containing options is also handed to 'solvePhaseRetrieval'.
# The entries in this struct specify which recovery algorithm is used.
#
# For more details, see the Phasepack user guide.
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

from numpy.linalg import norm
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt

from phasepack.util import Options, ConvMatrix
from phasepack.solvers import solvePhaseRetrieval

# Parameters
n = 100 # Dimension of unknown vector
m = 8*n # Number of measurements
# Build the target signal
x_true = randn(n, 1)+1j*randn(n, 1)

# Create the measurement operator
# Note: we use a dense matrix in this example, but PhasePack also supports
# function handles.  See the more complex 'runImageReconstructionDemo.m'
# script for an example using the fast Fourier transform.
A = ConvMatrix(randn(m, n) + 1j*randn(m,n))

# Compute phaseless measurements
b = np.abs(A*x_true)

## Set options for PhasePack - this is where we choose the recovery algorithm
opts = Options(algorithm='Fienup',
               initMethod='optimal',
               tol=1E-3,
               verbose=2)

# Run the Phase retrieval Algorithm
print('Running %s algorithm' % opts.algorithm)
# Call the solver using the measurement operator 'A', the
# measurements 'b', the length of the signal to be recovered, and the
# options.  Note, the measurement operator can be either a function handle
# or a matrix.   Here, we use a matrix.  In this case, we have omitted the
# second argument. If 'A' had been a function handle, we would have
# handed the transpose of 'A' in as the second argument.

x, outs, opts = solvePhaseRetrieval(A=A, b0=b, n=n, opts=opts)
# Remove phase ambiguity
# Phase retrieval can only recover images up to a phase ambiguity.
# Let's apply a phase rotation to align the recovered signal with the
# original so they look the same when we plot them.
rotation = (x.conjugate().T@x_true)/np.abs(x.conjugate().T@x_true)
x = x*rotation
print('Signal recovery required %d iterations (%f secs)\n' % (outs.iterationCount, outs.solveTimes[-1]))

# Plot the true vs recovered signal.  Ideally, this scatter plot should be clustered around the 45-degree line.
fig, axes = plt.subplots(1,2)
axes[0].scatter(np.real(x_true), np.real(x))
axes[0].set_xlabel('Original signal value')
axes[0].set_ylabel('Recovered signal value')
axes[0].set_title('Original vs recovered signal')
# Plot a convergence curve
axes[1].semilogy(outs.solveTimes, outs.residuals, linewidth=1.75)
axes[1].grid()
axes[1].set_xlabel('Time (sec)')
axes[1].set_ylabel('Error')
axes[1].set_title('Convergence Curve');
# set(gcf,'units','points','position',[0,0,1200,300]);
plt.show()

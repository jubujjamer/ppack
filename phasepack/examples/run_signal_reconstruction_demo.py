"""
This script will create phaseless measurements from a 1d test signal, and then recover the signal using phase retrieval methods.  We now describe the details of the simple recovery problem that this script implements.

                        Recovery Problem
This script creates a complex-valued random Gaussian signal. Measurements of the signal are then obtained by applying a linear operator to the signal, and computing the magnitude (i.e., removing the phase) of the results.

                      Measurement Operator
Measurement are obtained using a linear operator, called 'A', that contains random Gaussian entries.

                     The Recovery Algorithm
First, the recovery options should be given to the 'Options' class, which manages them and applies the default cases for each algorithm or initializer, in case of not being explicitly provided by the user.
All the information about the problem is collectted in the 'Retrieval' class and
the image is recovered by calling its method solve_phase_retrieval().

For more details, see the Phasepack user guide.

Based on MATLAB implementation by Rohan Chandra, Ziyuan Zhong, Justin Hontz,
Val McCulloch, Christoph Studer & Tom Goldstein.
Copyright (c) University of Maryland, 2017.
Python version of the phasepack module by Juan M. Bujjamer.
University of Buenos Aires, 2018.
"""
from numpy.linalg import norm
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt

from phasepack.containers import Options
from phasepack.matops import ConvolutionMatrix
from phasepack.retrieval import Retrieval

# Parameters
n = 100 # Dimension of unknown vector
m = 8*n # Number of measurements
# Build the target signal
x_true = randn(n, 1)+1j*randn(n, 1)
# Create the measurement operator
A = ConvolutionMatrix(randn(m, n) + 1j*randn(m,n))
# Compute phaseless measurements
b = np.abs(A*x_true)
# Set options for PhasePack - this is where we choose the recovery algorithm
opts = Options(algorithm='fienup', init_method='optimal', tol=1E-4,
               verbose=2)
# Create an instance of the phase retrieval class, which manages initializers
# and selection of solvers acording to the options provided.
retrieval = Retrieval(A, b, opts)
# Run the Phase retrieval Algorithm
print('Running %s algorithm' % opts.algorithm)
# Call the solver using the measurement operator 'A', the measurements 'b', the
# length of the signal to be recovered, and the options.
# Note the measurement operator can be either a function handle or a matrix.
# Here, we use a matrix.  In this case, we have omitted the second argument. If # 'A' had been a  function handle, we would have handed the transpose of 'A' in # as the second argument.
x, outs, opts = retrieval.solve_phase_retrieval()
# Remove phase ambiguity
# Phase retrieval can only recover images up to a phase ambiguity.
rotation = (x.conjugate().T@x_true)/np.abs(x.conjugate().T@x_true)
# Let's apply a phase rotation to align the recovered signal with the original
# so they look the same when we plot them.
x = x*rotation
print('Signal recovery required %d iterations (%f secs)\n' % (outs.iteration_count, outs.solve_times[-1]))
# Plot the true vs recovered signal.  Ideally, this scatter plot should be
# clustered around the 45-degree line.
fig, axes = plt.subplots(1, 2)
axes[0].scatter(np.real(x_true), np.real(x))
axes[0].set_xlabel('Original signal value')
axes[0].set_ylabel('Recovered signal value')
axes[0].set_title('Original vs recovered signal')
# Plot a convergence curve
axes[1].semilogy(outs.solve_times, outs.residuals, linewidth=1.75)
axes[1].grid()
axes[1].set_xlabel('Time (sec)')
axes[1].set_ylabel('Error')
axes[1].set_title('Convergence Curve');
# set(gcf,'units','points','position',[0,0,1200,300]);
plt.figure()
plt.plot(np.real(x_true))
plt.show()

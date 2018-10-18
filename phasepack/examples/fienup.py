# run_fienup.py
#
# This test file implements the Fienup solver. The code builds
# a synthetic formulation of the Phase Retrieval problem, b = |Ax| and
# computes an estimate to x. The code finally plots a convergence curve
# and also makes a scatter plot of the true vs recovered solution.
#
# PAPER TITLE:
#              .Phase retrieval algorithms: a comparison
#
# ARXIV LINK:
#          https://www.osapublishing.org/ao/abstract.cfm?uri=ao-21-15-2758
#
#
# 1) Each test script starts out by defining the length of the unknown
# signal, n and the number of measurements, m. These mesurements can be
# made complex by setting the is_complex flag to be true.
#
# 2) We then build the test problem by invoking the function
# 'build_test_problem' which generates random gaussian measurements according
# to the user's choices in step(1). The function returns the measurement
# matrix 'A', the true signal 'xt' and the measurements 'b0'.
#
# 3) We set the options for the PR solver. For example, the maximum
# number of iterations, the tolerance value, the algorithm and initializer
# of choice. These options are controlled by setting the corresponding
# entries in the 'opts' struct.  Please see the user guide for a complete
# list of options.
#
# 4) We solve the phase retrieval problem by running the following line
# of code:
#   >>  [x, outs, opts] = solve_phase_retrieval(A, A', b0, n, opts)
# This solves the problem using the algorithm and initialization scheme
# specified by the user in the struct 'opts'.
#
# 5) Determine the optimal phase rotation so that the recovered solution
# matches the true solution as well as possible.
#
# 6) Report the relative reconstruction error. Plot residuals (a measure
# of error) against the number of iterations and plot the real part of the
# recovered signal against the real part of the original signal.
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

from numpy.linalg import norm

from phasepack.util import Options, build_test_problem, plot_error_convergence, plot_recovered_vs_original
from phasepack.solvers import solve_phase_retrieval

# Parameters
n = 100 # Dimension of unknown vector
m = 5*n # Number of measurements
is_complex = True # If the signal and measurements are complex

# Build a test problem
print('Building test problem...');
[A, xt, b0, At] = build_test_problem(m, n ,is_complex)

# Options
opts = Options(algorithm='Fienup',
               init_method='truncated_spectral',
               tol=1E-6,
               verbose=2,
               is_complex = is_complex,
               record_recon_errors=False,
               record_measurement_errors=True,
               fienup_tuning = 0.5)
# [x, outs, opts] =
x, outs, opts = solve_phase_retrieval(A=A, At=A.T, b0=b0, n=n, opts=opts)
# Determine the optim al phase rotation so that the recovered signal matches the true signal as well
# as possible.
alpha = (x.conjugate().T@xt)/(x.conjugate().T@x)
# print('alpha', alpha)
x = alpha*x
# Determine the relative reconstruction error
recon_error = norm(xt-x) /norm(xt)
print('relative recon error = %.8f\n' % recon_error)
# Finally, the user can plot the convergence results and the recovered signal x against the true signal xt
# using two helper functions.

# Plot a graph or error versus the number of iterations.
plot_error_convergence(outs, opts)
# Plot a graph of the recovered signal x against the true signal xt.
plot_recovered_vs_original(x, xt)

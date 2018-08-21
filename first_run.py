from numpy.linalg import norm

from phasepack.util import Options, buildTestProblem, plotErrorConvergence
from phasepack.solvers import solvePhaseRetrieval

# Parameters
n = 50 # Dimension of unknown vector
m = 5*n # Number of measurements
isComplex = True # If the signal and measurements are complex
# Build a random test problem
print('Building test problem...');
[A, xt, b0, At] = buildTestProblem(m, n ,isComplex);
opts = Options(algorithm='Fienup', initMethod='truncatedSpectral', tol=1E-10, maxIters=500,
               recordReconErrors=True, recordMeasurementErrors=True, xt=xt)
# [x, outs, opts] =
x, outs, opts = solvePhaseRetrieval(A, A.T, b0 , n, opts)
# Determine the optim al phase rotation so that the recovered signal matches the true signal as well
# as possible.
alpha = (x.conjugate().T*xt)/(x.conjugate().T*x)
x = alpha*x
# Determine the relative reconstruction error
reconError = norm(xt-x) /norm(xt)
print('relative recon error = %.8f\n' % reconError)
# Finally, the user can plot the convergence results and the recovered signal x against the true signal xt
# using two helper functions
# Plot a graph or error versus the number of iterations
plotErrorConvergence(outs, opts)
# % Pl o t a graph o f the r e c o v e r e d s i g n a l x a g ai n s t the t r u e s i g n a l xt .
# pl o tR e c o v e r e dV SO ri gi n al ( x , xt ) ;

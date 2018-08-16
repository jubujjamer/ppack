from phasepack.util import Options, buildTestProblem
from phasepack.solvers import solvePhaseRetrieval

# Parameters
n = 10 # Dimension of unknown vector
m = 5*n # Number of measurements
isComplex = True # If the signal and measurements are complex
# Build a random test problem
print('Building test problem...');
[A, xt, b0, At] = buildTestProblem(m, n ,isComplex);
opts = Options(algorithm='Fienup', initMethod='truncatedSpectral', tol=1E-10, maxIters=500)
# [x, outs, opts] =
solvePhaseRetrieval(A, A.T, b0 , n, opts)

#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File solvers.py

Last update: 15/08/2018

Usage:

"""
__version__ = "1.0.0"
__author__ = 'Juan M. Bujjamer'
__all__ = ['solvePhaseRetrieval']

import time
import warnings
import numpy as np
from numpy.linalg import norm

from phasepack.util import Options, Container, ConvMatrix, stopNow, displayVerboseOutput
from phasepack.initializers import initSpectral, initOptimalSpectral
from phasepack.gdescent import gdOptions, gradientDescentSolver


def validateInput(A, b0, n, opts, At=None):
    assert n>0, 'n must be positive'

    assert (np.abs(b0) == b0).all, 'b must be real-valued and non-negative'

    if callable(A) and type(At) == np.ndarray:
        raise Exception('If A is a function handle, then At must also be a function handle')

    if opts.customx0:
        assert np.shape(opts.customx0) == (n, 1), 'customx0 must be a column vector of length n'

def checkAdjoint(A, b, At=None):
    """ Check that A and At are indeed ajoints of one another
    """
    y = np.random.randn(*b.shape);
    # Aty = At(y) # Check
    Aty = At*y #At@y
    x = np.random.randn(*Aty.shape)
    # Ax = A(x) # check
    Ax = A*x #Ax = A@x
    innerProduct1 = Ax.conjugate().T@y
    innerProduct2 = x.conjugate().T@Aty
    error = np.abs(innerProduct1-innerProduct2)/np.abs(innerProduct1)

    assert error<1e-3, 'Invalid measurement operator:  At is not the adjoint of A.  Error = %.1f' % error


def initX(A, b0, n, opts, At=None):
    # initMethods = {'truncatedspectral': initSpectral(A=A, At=At, b0=b0, n=n,
    #                                     isTruncated=True, isScaled=True, verbose=opts.verbose),
    #                 'truncated': initSpectral(A=A, At=At, b0=b0, n=n,
    #                                     isTruncated=True, isScaled=True, verbose=opts.verbose),
    #                 'spectral': initSpectral(A=A, At=At, b0=b0, n=n,
    #                                     isTruncated=False, isScaled=True, verbose=opts.verbose),
    #                 'optimal':  initOptimalSpectral(A=A, At=At, b0=b0, n=n,
    #                                     isScaled=True, verbose=opts.verbose),
    #                 'optimalspectral':  initOptimalSpectral(A=A, At=At, b0=b0, n=n,
    #                                     isScaled=True, verbose=opts.verbose)}
    # x0 = initMethods[opts.initMethod.lower()]
    chosen_opt = opts.initMethod.lower()
    if chosen_opt == 'truncatedspectral' or chosen_opt == 'truncated':
        return initSpectral(A=A, At=At, b0=b0, n=n, isTruncated=True, isScaled=True, verbose=opts.verbose)
    elif chosen_opt == 'spectral':
        return initSpectral(A=A, At=At, b0=b0, n=n, isTruncated=False, isScaled=True, verbose=opts.verbose)
    elif chosen_opt == 'optimal' or chosen_opt == 'optimalspectral':
            return initOptimalSpectral(A=A, At=At, b0=b0, n=n, isScaled=True, verbose=opts.verbose)
    else:
        raise Exception('Unknown initialization option.')
    #
    # case {'amplitudespectral','amplitude'}
    #     x0 = initAmplitude(A,At,b0,n,opts.verbose);
    # case {'weightedspectral','weighted'}
    #     x0 = initWeighted(A,At,b0,n,opts.verbose);
    # case {'orthogonalspectral','orthogonal'}
    #     x0 = initOrthogonal(A,At,b0,n,opts.verbose);

    # case 'angle'
    #     assert(isfield(opts,'xt'),'The true solution, opts.xt, must be specified to use the angle initializer.')
    #     assert(isfield(opts,'initAngle'),'An angle, opts.initAngle, must be specified (in radians) to use the angle initializer.')
    #     x0 = initAngle(opts.xt, opts.initAngle);
    # case 'custom'
    #     x0 = opts.customx0;
    # otherwise
    #     error('Unknown initialization method "%s"', opts.initMethod);
    return x0

def optsCustomAlgorithm(A, At, b0, x0, opts):
    return

def solveAmplitudeFlow(A, At, b0, x0, opts):
    return

def solveCoordinateDescent(A, At, b0, x0, opts):
    return

def solveFienup(A, At, b0, x0, opts):
    """ Solver for Fienup algorithm.

   Inputs:
      A:    m x n matrix or a function handle to a method that
            returns A*x.
      At:   The adjoint (transpose) of 'A'. If 'A' is a function handle,
            'At' must be provided.
      b0:   m x 1 real,non-negative vector consists of all the measurements.
      x0:   n x 1 vector. It is the initial guess of the unknown signal x.
      opts: A struct consists of the options for the algorithm. For details,
            see header in solvePhaseRetrieval.m or the User Guide.

      Note: When a function handle is used, the
      value of 'At' (a function handle for the adjoint of 'A') must be
      supplied.

   Outputs:
      sol:  n x 1 vector. It is the estimated signal.
      outs: A struct consists of the convergence info. For details,
            see header in solvePhaseRetrieval.m or the User Guide.


   See the script 'testFienup.m' for an example of proper usage of this
   function.

  % Notations
   The notations mainly follow those used in Section 2 of the Fienup paper.
   gk:    g_k   the guess to the signal before the k th round
   gkp:   g_k'  the approximation to the signal after the k th round of
          iteration
   gknew: g_k+1 the guess to the signal before the k+1 th round
   Gkp:   G_k'  the approximation to fourier transfor of the signal after
                satisfying constraints on fourier-domain
   beta:  \beta the Tuning parameter for object-domain update

  % Algorithm Description
   Fienup Algorithm is the same as Gerchberg-Saxton Algorithm except when
   the signal is real and non-negative (or has constraint in general). When
   this happens, the update on the object domain is different.

   Like Gerchberg-Saxton, Fienup transforms back and forth between the two
   domains, satisfying the constraints in one before returning to the
   other. The method has four steps (1) Left multipy the current estimation
   x by the measurement matrix A and get Ax. (2) Keep phase, update the
   magnitude using the measurements b0, z = b0.*sign(Ax). (3) Solve the
   least-squares problem
            sol = \argmin ||Ax-z||^2
       to get our new estimation x. We use Matlab built-in solver lsqr()
       for this least square problem.
   (4) Impose temporal constraints on x(This step is ignored when there is
   no constraints)

   For a detailed explanation, see the Fienup paper referenced below.


  % References
   Paper Title:   Phase retrieval algorithms: a comparison
   Place:         Section II for notation and Section V for the
                  Input-Output Algorithm
   Authors:       J. R. Fienup
   Address: https://www.osapublishing.org/ao/abstract.cfm?uri=ao-21-15-2758

   PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
   Christoph Studer, & Tom Goldstein
   Copyright (c) University of Maryland, 2017

    """

    def validateOptions(opts):
        try:
            float(opts.FienupTuning)
        except:
            raise Exception("%s should be a number" % opts.FienupTuning)
#
    # Initialization
    gk = x0                      # Initial Guess, corresponds to g_k in the paper
    gkp = x0                     # corresponds to g_k' in the paper
    gknew = x0                   # corresponds to g_k+1 in the paper
    beta = opts.FienupTuning     # GS tuning parameter
#
    # Initialize values potentially computed at each round.
    currentTime = []
    currentResid = []
    currentReconError = []
    currentMeasurementError = []

    # Initialize vectors for recording convergence information
    # [solveTimes,measurementErrors,reconErrors,residuals] = initializeContainers(opts);
    container = Container(opts)

#     % Build a function handle for matlab's conjugate-gradient solver
#     function y = Afun(x,transp_flag)
#        if strcmp(transp_flag,'transp')       % y = A'*x
#           y = At(x);
#        elseif strcmp(transp_flag,'notransp') % y = A*x
#           y = A(x);
#        end
#     end
#
    startTime = time.time() # Start timer
#
    for iter in range(opts.maxIters):
#
#         Ax = A(gk);            % Intermediate value to save repetitive computation
#         Gkp = b0.*sign(Ax);    % Calculate the initial spectral magnitude, G_k' in the paper.
        Ax = A*gk            # Intermediate value to save repetitive computation
        Gkp = b0*Ax/np.abs(Ax)
        #-----------------------------------------------------------------------
        # Record convergence information and check stopping condition
        # If xt is provided, reconstruction error will be computed and used for stopping
        # condition. Otherwise, residual will be computed and used for stopping
        # condition.
        if len(opts.xt) > 0:
            x = gk
            xt = opts.xt
            # Compute optimal rotation
            alpha = (x.T@xt)/(x.T@x)
            x = alpha*x
            currentReconError = norm(x-xt)/norm(xt);
            if opts.recordReconErrors:
                container.appendReconError(currentReconError)

        if not len(opts.xt) == 0 or opts.recordResiduals:
            currentResid = norm(A.hmul((Ax-Gkp)))/norm(Gkp)

        if opts.recordResiduals:
            container.appendResidual(currentResid)

        currentTime = time.time()-startTime  #Record elapsed time so far
        if opts.recordTimes:
            container.appendRecordTime(currentTime)

        if opts.recordMeasurementErrors:
            currentMeasurementError = norm(np.abs(A*gk)-b0)/norm(b0)
            container.appendMeasurementError(currentMeasurementError)
#
        # Display verbose output if specified
        if opts.verbose == 2:
          displayVerboseOutput(iter, currentTime, currentResid, currentReconError, currentMeasurementError)

        #  Test stopping criteria.
        if stopNow(opts, currentTime, currentResid, currentReconError):
            break
        # Solve the least-squares problem
        # gkp = \argmin ||Ax-Gkp||^2.
        # If A is a matrix,
        # gkp = inv(A)*Gkp
        # If A is a fourier transform( and measurements are not oversampled i.e. m==n),
        # gkp = inverse fourier transform of Gkp
        gkp = A.lsqr(Gkp, opts.tol/100, opts.maxInnerIters, gk)
        # gkp=lsqr(@Afun,Gkp,opts.tol/100,opts.maxInnerIters,[],[],gk)

        # If the signal is real and non-negative, Fienup updates object domain
        # following the constraint
        if not opts.isComplex and opts.isNonNegativeOnly:
            inds = gkp < 0  # Get indices that are outside the non-negative constraints
                            # May also need to check if isreal
            inds2 = not inds # Get the complementary indices
            # hybrid input-output (see Section V, Equation (44))
            gknew[inds] = gk[inds] - beta*gkp[inds]
            gknew[inds2] = gkp[inds2]
        else: # Otherwise, its update is the same as the GerchBerg-Saxton algorithm
            gknew = gkp.reshape(-1,1)
        gk = gknew # update gk
        # print(gk)
    sol = gk
#     % Create output according to the options chosen by user
    container.iterationCount = iter
    # Display verbose output if specified
    if opts.verbose:
        displayVerboseOutput(iter, currentTime, currentResid, currentReconError, currentMeasurementError)
#
#
# % Check the validify of algorithm specific options


    return sol, container

def solveGerchbergSaxton(A, At, b0, x0, opts):
    return

def solveKaczmarzSimple(A, At, b0, x0, opts):
    return
def solvePhaseMax(A, At, b0, x0, opts):
    return

def solvePhaseLamp(A, At, b0, x0, opts):
    return

def solvePhaseLift(A, At, b0, x0, opts):
    return

def solveRAF(A, At, b0, x0, opts):
    return

def solveRWF(A, At, b0, x0, opts):
    return

def solveSketchyCGM(A, At, b0, x0, opts):
    return

def solveTAF(A, At, b0, x0, opts):
    return

def solveTWF(A, At, b0, x0, opts):
    """
    Implementation of the truncated Wirtinger Flow (TWF) algorithm.
    The code below is adapted from implementation of the
    Wirtinger Flow algorithm designed and implemented by E. Candes, X. Li,
    and M. Soltanolkotabi.

    Inputs:
        A:    m x n matrix or a function handle to a method that
              returns A*x.
        At:   The adjoint (transpose) of 'A'. If 'A' is a function handle,
              'At' must be provided.
        b0:   m x 1 real,non-negative vector consists of all the measurements.
        x0:   n x 1 vector. It is the initial guess of the unknown signal x.
        opts: A struct consists of the options for the algorithm. For details,
              see header in solvePhaseRetrieval.m or the User Guide.

        Note: When a function handle is used, the value of 'At' (a function
        handle for the adjoint of 'A') must be supplied.

    Outputs:
        sol:  n x 1 vector. It is the estimated signal.
        outs: A struct containing convergence info. For details,
              see header in solvePhaseRetrieval.m or the User Guide.


     See the script 'testTWF.m' for an example of proper usage of this
     function.

    Notations:
        x is the estimation of the signal. y is the vector of measurements such
        that yi = |<ai,x>|^2 for i = 1,...,m Most of our notations are
        consistent with the notations used in the TWF paper referenced below.

    Algorithm Description:
     Similar to WF, TWF successively refines the estimate via a gradient
     descent scheme.  The loss function is the negative log of the Poisson
     likelihood.

     Unlike WF, TWF regularizes the gradient flow in a data-dependent fashion
     by operating only upon some iteration-varying index subsets that
     correspond to those data yi whose resulting gradient components are in
     some sense not excessively large.

     This gives us a more stable search directions and avoids the overshoot
     problem of the Wirtinger Flow Algorithm.

     We also add a feature: when opts.isComplex==false and
     opts.isNonNegativeOnly==true i.e. when the signal is real and
     non_negative signal, then at each iteration, negative values in the
     latest solution vector will be set to 0. This helps to speed up the
     convergence of errors.

     For a detailed explanation, see the TWF paper referenced below.

     References
     Paper Title:   Solving Random Quadratic Systems of Equations Is Nearly
                    as Easy as Solving Linear Systems
     Place:         Algorithm 1
     Authors:       Yuxin Chen, Emmanuel J. Candes
     arXiv Address: https://arxiv.org/abs/1505.05114

    PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
    Christoph Studer, & Tom Goldstein
    Copyright (c) University of Maryland, 2017

    """
    gdOpts = gdOptions(opts)


    def updateObjective(x, Ax):
        y = b0**2 # The TWF formulation uses y as the measurements rather than b0
        m = y.size # number of Measurements
        Kt = 1/m*norm(y-np.abs(Ax)**2, ord=1);   # 1/m * sum(|y-|a'z|^2|)
        # Truncation rules
        # Unlike what specified in the TWF paper Algorithm1, the
        # term sqrt(n)/abs(x) does not appear in the following equations
        Eub =  np.abs(Ax)/norm(x) <= opts.alpha_ub
        Elb =  np.abs(Ax)/norm(x) >= opts.alpha_lb
        Eh  =  np.abs(y-np.abs(Ax)**2) <= opts.alpha_h*Kt*np.abs(Ax)/norm(x)
        mask = Eub*Elb*Eh
        s = np.sum(mask)
        f = lambda z: (0.5/s)*np.sum(mask*(np.abs(z)**2-y*np.log(np.abs(z)**2) ) )
        gradf = lambda z: (1.0/s)*mask*(np.abs(z)**2-y)/z.conjugate()
        return f, gradf

    sol, outs = gradientDescentSolver(A, At, x0, b0, updateObjective, gdOpts)



    return

def solveWirtFlow(A, At, b0, x0, opts):

    return

def solveX(A, At, b0, x0, opts):
    chooseAlgorithm = {'custom': optsCustomAlgorithm,
                   'amplitudeflow': solveAmplitudeFlow,
                   'coordinatedescent': solveCoordinateDescent,
                   'fienup': solveFienup,
                   'gerchbergsaxton': solveGerchbergSaxton,
                   'kaczmarz': solveKaczmarzSimple,
                   'phasemax': solvePhaseMax,
                   'phaselamp': solvePhaseLamp,
                   'phaselift': solvePhaseLift,
                   'raf': solveRAF,
                   'rwf': solveRWF,
                   'sketchycgm': solveSketchyCGM,
                   'taf': solveTAF,
                   'twf': solveTWF,
                   'wirtflow': solveWirtFlow}
    sol, outs = chooseAlgorithm[opts.algorithm.lower()](A, At, b0, x0, opts)

    return sol, outs, opts


def solvePhaseRetrieval(A, b0, n,  At=None, opts=None):
    """ This method solves the problem:
                          Find x given b0 = |Ax+epsilon|
     Where A is a m by n complex matrix, x is a n by 1 complex vector, b0 is a m by 1 real,non-negative vector and epsilon is a m by 1 vector. The user supplies function handles A, At and measurement b0. Note: The unknown signal to be recovered must be 1D for our interface.
     Inputs:
      A     : A m x n matrix (or optionally a function handle to a method) that returns A*x
      At    : The adjoint (transpose) of 'A.' It can be a n x m matrix or a function handle.
      b0    : A m x 1 real,non-negative vector consists of  all the measurements.
      n     : The size of the unknown signal. It must be provided if A is a function handle.
      opts  : An optional struct with options.  The commonly used fields of 'opts' are:
                 initMethod              : (string,
                 default='truncatedSpectral') The method used
                                           to generate the initial guess x0.
                                           User can use a customized initial
                                           guess x0 by providing value to
                                           the field customx0.
                 algorithm               : (string, default='altmin') The
                                           algorithm used
                                           to solve the phase retrieval
                                           algorithm. User can use a
                                           customized algorithm by providing
                                           a function [A,At,b0,x0,opts]
                                           ->[x,outs,opts] to the field
                                           customAlgorithm.
                 maxIters                : (integer, default=1000) The
                                           maximum number of
                                           iterations allowed before
                                           termination.
                 maxTime                 : (positive real number,
                                           default=120, unit=second)
                                           The maximum seconds allowed
                                           before termination.
                 tol                     : (double, default=1e-6) The
                                           stopping tolerance.
                                           It will be compared with
                                           reconerror if xt is provided.
                                           Otherwise, it will be compared
                                           with residual. A smaller value of
                                           'tol' usually results in more
                                           iterations.
                 verbose                 : ([0,1,2], default=0)  If ==1,
                                           print out
                                           convergence information in the
                                           end. If ==2, print out
                                           convergence information at every
                                           iteration.
                 recordMeasurementErrors : (boolean, default=false) Whether
                                           to compute and record
                                           error(i.e.
                                           norm(|Ax|-b0)/norm(b0)) at each
                                           iteration.
                 recordResiduals         : (boolean, default=true) If it's
                                           true, residual will be
                                           computed and recorded at each
                                           iteration. If it's false,
                                           residual won't be recorded.
                                           Residual also won't be computed
                                           if xt is provided. Note: The
                                           error metric of residual varies
                                           across solvers.
                 recordReconErrors       : (boolean, default=false) Whether
                                           to record
                                           reconstruction error. If it's
                                           true, opts.xt must be provided.
                                           If xt is provided reconstruction
                                           error will be computed regardless
                                           of this flag and used for
                                           stopping condition.
                 recordTimes             : (boolean, default=true) Whether
                                           to record
                                           time at each iteration. Time will
                                           be measured regardless of this
                                           flag.
                 xt                      : A n x 1 vector. It is the true
                                           signal. Its purpose is
                                           to compute reconerror.

              There are other more algorithms specific options not listed
              here. To use these options, set the corresponding field in
              'opts'. For example:
                        >> opts.tol=1e-8; >> opts.maxIters = 100;


     Outputs:
      sol               : The approximate solution outs : A struct with
                          convergence information
      iterationCount    : An integer that is  the number of iterations the
                          algorithm runs.
      solveTimes        : A vector consists  of elapsed (exist when
                          recordTimes==true) time at each iteration.
      measurementErrors : A vector consists of the errors (exist when
                          recordMeasurementErrors==true)   i.e.
                          norm(abs(A*x-b0))/norm(b0) at each iteration.
      reconErrors       : A vector consists of the reconstruction (exist
                          when recordReconErrors==true) errors
                          i.e. norm(xt-x)/norm(xt) at each iteration.
      residuals         : A vector consists of values that (exist when
                          recordResiduals==true)  will be compared with
                          opts.tol for stopping condition  checking.
                          Definition varies across solvers.
      opts              : A struct that contains fields used by the solver.
                          Its possible fields are the same as the input
                          parameter opts.

   For more details and more options in opts, see the Phasepack user guide.

   PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch, Christoph Studer, & Tom Goldstein Copyright (c) University of Maryland, 2017
    """

    # If opts is not provided, create it
    if opts is None:
        opts = Options()

    if type(A) == np.ndarray:
        # n = Am.shape[1]
        # Transform matrix into function form
        # At = ConvMatrix(A.conjugate().T)
        A = ConvMatrix(A)

    # Check that inputs are of valid datatypes and sizes
    validateInput(A=A, b0=b0, n=n, opts=opts)
    # Check that At is the adjoint/transpose of A
    # if At is not None:
    #     checkAdjoint(A=A, b=b0, At=At)
    # # Initialize x0
    x0 = initX(A=A, b0=b0, n=n, opts=opts)
    # % Truncate imaginary components of x0 if working with real values
    if not opts.isComplex:
        x0 = np.real(x0)
    elif opts.isNonNegativeOnly:
        warnings.warn('opts.isNonNegativeOnly will not be used when the signal is complex.');

    [sol, outs, opts] = solveX(A=A, At=None, b0=b0, x0=x0, opts=opts) # Solve the problem using the specified algorithm
    return sol, outs, opts

#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File gdescent.py

Last update: 24/08/2018

Usage:

"""
__version__ = "1.0.0"
__author__ = 'Juan M. Bujjamer'
__all__ = ['gradientDescentSolver']

import time
import warnings
import numpy as np
from numpy.linalg import norm
from numpy.random import randn

from pdb import set_trace as bp
from phasepack.util import Options, Container, ConvMatrix, displayVerboseOutput


class gdOptions(object):
    """ Gradient Descent Algorithm Options manager.
    """
    def __init__(self, opts):
        self.maxIters = opts.maxIters
        self.maxTime = opts.maxTime
        self.tol = opts.tol
        self.verbose = opts.verbose
        self.recordTimes = opts.recordTimes
        self.recordResiduals = opts.recordResiduals
        self.recordMeasurementErrors = opts.recordMeasurementErrors
        self.recordReconErrors = opts.recordReconErrors
        self.xt = opts.xt
        self.updateObjectivePeriod = opts.truncationPeriod
        self.searchMethod = opts.searchMethod
        self.betaChoice = opts.betaChoice

        try:
            self.updateObjectivePeriod = opts.updateObjectivePeriod
        except:
            self.updateObjectivePeriod = -np.inf
        try:
            self.tolerancePenaltyLimit = opts.tolerancePenaltyLimit
        except:
            self.tolerancePenaltyLimit = 3
        try:
            self.searchMethod = opts.searchMethod
        except:
            self.searchMethod = 'steepestDescent'

        if opts.searchMethod.lower() == 'lbfgs':
            try:
                 self.storedVectors = opts.storedVectors
            except:  self.storedVectors= 5
        if opts.searchMethod.lower() == 'ncg':
            try:
                self.betaChoice = opts.betaChoice
            except:
                self.betaChoice = 'HS'
        try:
            self.ncgResetPeriod = opts.ncgResetPeriod
        except:
            self.ncgResetPeriod = 100


def determineSearchDirection(opts, gradf1, index, lastObjectiveUpdateIter, lastNcgResetIter=None,
                             unscaledSearchDir=None, rhoVals=None, sVals=None, yVals=None, Dg=None):
    """
        Determine search direction for next iteration based on specified search
        methodself.

        TODO: Manage None values
    """
    if opts.searchMethod.lower() == 'steepestdescent':
        searchDir = -gradf1
    elif opts.searchMethod.lower() == 'ncg':
        searchDir = -gradf1
        # Reset NCG progress after specified number of iterations have
        # passed
        if index - lastNcgResetIter == opts.ncgResetPeriod:
            unscaledSearchDir = zeros(n)
            lastNcgResetIter = index
        # Proceed only if reset has not just occurred
        if index != lastNcgResetIter:
            if opts.betaChoice.lower() == 'hs':
                # Hestenes-Stiefel
                beta = -np.real(gradf1.congujate().T*Dg)/np.real(unscaledSearchDir.congujate().T*Dg)
            elif opts.betaChoice.lower() == 'fr':
                # Fletcher-Reeves
                beta = norm(gradf1)**2 / norm(gradf0)^2;
            elif opts.betaChoice.lower == 'pr':
                #Polak-Ribiere
                beta = np.real(gradf1.conjugate().T*Dg)/norm(gradf0)**2
            elif opts.betaChoice.lower ==  'dy':
                # Dai-Yuan
                beta = norm(gradf1)**2 / np.real(unscaledSearchDir.conjugate().T*Dg)

            searchDir = searchDir + beta * unscaledSearchDir;
        unscaledSearchDir = searchDir
    elif opts.searchMethod.lower() == 'lbfgs':
        searchDir = -gradf1
        iters = np.min(index-lastObjectiveUpdateIter, opts.storedVectors);
        if iters > 0:
            alphas = np.zeros(iters)
            # First loop
            for j in range(iters):
                alphas[j] = rhoVals[j]*np.real(sVals[:,j].conjugate().T*searchDir)
                searchDir = searchDir - alphas[j] * yVals[:,j]

            # Scaling of search direction
            gamma = np.real(Dg.conjugate().T*Dx)/(Dg.conjugate().T*Dg)
            searchDir = gamma*searchDir
            # Second loop
            for j in range(iters, 1, -1):
                beta = rhoVals[j]*np.real(yVals[:,j].congugate().T*searchDir)
                searchDir = searchDir + (alphas[j]-beta)*sVals[:,j]
            searchDir = 1/gamma*searchDir
            searchDir = norm(gradf1)/norm(searchDir)*searchDir
    # Change search direction to steepest descent direction if current
    # direction is invalid
    if any(np.isnan(searchDir)) or any(np.isinf(searchDir)):
        searchDir = -gradf1
    # Scale current search direction match magnitude of gradient
    searchDir = norm(gradf1) / norm(searchDir)*searchDir
    return searchDir


def determineInitialStepsize(A, x0, gradf):
    """ Determine reasonable initial stepsize of current objective function
        (adapted from FASTA.m)
    """
    x_1 = randn(*x0.shape)
    x_2 = randn(*x0.shape)
    gradf_1 = A.hmul(gradf(A*x_1))
    gradf_2 = A.hmul(gradf(A*x_2))
    L = norm(gradf_1-gradf_2)/norm(x_2-x_1)
    L = max(L, 1.0e-30)
    tau = 25.0/L
    return tau

def updateStepsize(searchDir0, searchDir1, Dx, tau1, tau0):
    """ Update stepsize when objective update has not just occurred (adopted from
        FASTA.m)
    """
    Ds = searchDir0 - searchDir1
    dotprod = np.real(np.vdot(Dx, Ds))
    tauS = norm(Dx)**2/dotprod  # First BB stepsize rule
    tauM = dotprod/norm(Ds)**2 # Alternate BB stepsize rule
    tauM = max(tauM, 0)
    if 2*tauM > tauS:   #  Use "Adaptive"  combination of tau_s and tau_m
        tau1 = tauM
    else:
        tau1 = tauS - tauM/2  # Experiment with this param
    if tau1 <= 0 or np.isinf(tau1) or np.isnan(tau1): #  Make sure step is non-negative
        tau1 = tau0*1.5  # let tau grow, backtracking will kick in if stepsize is too big
    return tau1

def processIteration(index, startTime, Dx, maxDiff, opts, x1, updateObjectiveNow, container,
                     residualTolerance, tolerancePenalty):
    currentSolveTime = time.time()-startTime
    maxDiff = max(norm(Dx), maxDiff)
    currentResidual = norm(Dx)/maxDiff
    stopNow = False
    currentReconError = None
    currentMeasurementError = None
    # Obtain recon error only if true solution has been provided
    if opts.xt:
        reconEstimate = (x1.conjugate().T@opts.xt)/(x1.conjugate().T*x1)*x1
        currentReconError = norm(opts.xt-reconEstimate)/norm(opts.xt)
    if opts.recordTimes:
        container.appendRecordTime(currentSolveTime)
    if opts.recordResiduals:
        container.appendResidual(currentResidual)
    if opts.recordMeasurementErrors:
        currentMeasurementError = norm(abs(d1) - b0) / norm(b0)
        container.appendMeasurementError(currentMeasurementError)
    if opts.recordReconErrors:
        assert opts.xt, 'You must specify the ground truth solution if the "recordReconErrors" flag is set to true.  Turn this flag off, or specify the ground truth solution.'
        container.appendReconError(currentReconError)
    if opts.verbose == 2:

        displayVerboseOutput(index, currentSolveTime, currentResidual, currentReconError, currentMeasurementError)

    # Terminate if solver surpasses maximum allocated timespan
    if currentSolveTime > opts.maxTime:
        stopNow = True
    # If user has supplied actual solution, use recon error to determine
    # termination
    if opts.xt and currentReconError <= opts.tol:
        stopNow = True

    if currentResidual <= residualTolerance:
        # Give algorithm chance to recover if stuck at local minimum by
        # forcing update of objective function
        updateObjectiveNow = True
        # If algorithm continues to get stuck, terminate
        tolerancePenalty = tolerancePenalty + 1

        if tolerancePenalty >= opts.tolerancePenaltyLimit:
            stopNow = True
    print(stopNow)
    print(currentResidual <= residualTolerance)
    print(tolerancePenalty, opts.tolerancePenaltyLimit)
    return stopNow, updateObjectiveNow, maxDiff, tolerancePenalty

def gradientDescentSolver(A, At, x0, b0, updateObjective, opts):
    """% -------------------------gradientDescentSolver.m--------------------------------

    General routine used by phase retrieval algorithms that function by using
    line search methods. This function is internal and should not be called
    by any code outside of this software package.

    The line search approach first finds a descent direction along which the
    objective function f will be reduced and then computes a step size that
    determines how far x  should move along that direction. The descent
    direction can be computed by various methods, such as gradient descent,
    Newton's method and Quasi-Newton method. The step size can be determined
    either exactly or inexactly.

    This line search algorithm implements the steepest descent, non linear
    conjugate gradient, and the LBFGS method. Set the option accordingly as
    described below.

    % Aditional Parameters
    The following are additional parameters that are to be passed as fields
    of the struct 'opts':

    maxIters (required) - The maximum number of iterations that are
    allowed to
        occur.

    maxTime (required) - The maximum amount of time in seconds the
    algorithm
        is allowed to spend solving before terminating.

    tol (required) - Positive real number representing how precise the
    final
        estimate should be. Lower values indicate to the solver that a
        more precise estimate should be obtained.

    verbose (required) - Integer representing whether / how verbose
        information should be displayed via output. If verbose == 0, no
        output is displayed. If verbose == 1, output is displayed only
        when the algorithm terminates. If verbose == 2, output is
        displayed after every iteration.

    recordTimes (required) - Whether the algorithm should store the total
        processing time upon each iteration in a list to be obtained via
        output.

    recordResiduals (required) - Whether the algorithm should store the
        relative residual values upon each iteration in a list to be
        obtained via output.

    recordMeasurementErrors (required) - Whether the algorithm should
    store
        the relative measurement errors upon each iteration in a list to
        be obtained via output.

    recordReconErrors (required) - Whether the algorithm should store the
        relative reconstruction errors upon each iteration in a list to
        be obtained via output. This parameter can only be set 'true'
        when the additional parameter 'xt' is non-empty.

    xt (required) - The true signal to be estimated, or an empty vector
    if the
        true estimate was not provided.

    searchMethod (optional) - A string representing the method used to
        determine search direction upon each iteration. Must be one of
        {'steepestDescent', 'NCG', 'LBFGS'}. If equal to
        'steepestDescent', then the steepest descent search method is
        used. If equal to 'NCG', a nonlinear conjugate gradient method is
        used. If equal to 'LBFGS', a Limited-Memory BFGS method is used.
        Default value is 'steepestDescent'.

    updateObjectivePeriod (optional) - The maximum number of iterations
    that
        are allowed to occur between updates to the objective function.
        Default value is infinite (no limit is applied).

    tolerancePenaltyLimit (optional) - The maximum tolerable penalty
    caused by
        surpassing the tolerance threshold before terminating. Default
        value is 3.

    betaChoice (optional) - A string representing the choice of the value
        'beta' when a nonlinear conjugate gradient method is used. Must
        be one of {'HS', 'FR', 'PR', 'DY'}. If equal to 'HS', the
        Hestenes-Stiefel method is used. If equal to 'FR', the
        Fletcher-Reeves method is used. If equal to 'PR', the
        Polak-Ribi�re method is used. If equal to 'DY', the Dai-Yuan
        method is used. This field is only used when searchMethod is set
        to 'NCG'. Default value is 'HS'.

    ncgResetPeriod (optional) - The maximum number of iterations that are
        allowed to occur between resettings of a nonlinear conjugate
        gradient search direction. This field is only used when
        searchMethod is set to 'NCG'. Default value is 100.

    storedVectors (optional) - The maximum number of previous iterations
    of
        which to retain LBFGS-specific iteration data. This field is only
        used when searchMethod is set to 'LBFGS'. Default value is 5.
    """

    # Length of input signal
    n = len(x0)
    if opts.xt:
        residualTolerance = 1.0e-13
    else:
        residualTolerance = opts.tol

    # Iteration number of last objective update
    lastObjectiveUpdateIter = 0
    # Total penalty caused by surpassing tolerance threshold
    tolerancePenalty = 0
    # Whether to update objective function upon next iteration
    updateObjectiveNow = True
    # Maximum norm of differences between consecutive estimates
    maxDiff = -np.inf

    currentSolveTime = 0
    currentMeasurementError = []
    currentResidual = []
    currentReconError = []
    container = Container(opts)

    x1 = x0
    d1 = A*x1
    Dx = 0

    startTime = time.time()
    for index in range(opts.maxIters):
        # Signal to update objective function after fixed number of iterations
        # have passed
        if index-lastObjectiveUpdateIter == opts.updateObjectivePeriod:
            updateObjectiveNow = True
        # Update objective if flag is set
        if updateObjectiveNow:
            updateObjectiveNow = False
            lastObjectiveUpdateIter = index
            f, gradf = updateObjective(x1, d1)
            f1 = f(d1)
            gradf1 = A.hmul(gradf(d1))
            # if opts.searchMethod.lower() == 'lbfgs':
            #     # Perform LBFGS initialization
            #     yVals = np.zeros((n, opts.storedVectors))
            #     sVals = np.zeros((n, opts.storedVectors))
            #     rhoVals = np.zeros((1, opts.storedVectors))
            # elif opts.searchMethod.lower() == 'ncg':
            #     # Perform NCG initialization
            #     lastNcgResetindex= iter
            #     unscaledSearchDir = zeros((n, 1))

            searchDir1 = determineSearchDirection(opts, gradf1, index, lastObjectiveUpdateIter)

            # Reinitialize stepsize to supplement new objective function
            tau1 = determineInitialStepsize(A, x0, gradf)
        else:
            gradf1 = A.hmul(gradf(d1))
            Dg = gradf1 - gradf0
            # if opts.searchMethod.lower() == 'lbfgs':
            #     # Update LBFGS stored vectors
            #     sVals = np.hstack(( Dx, sVals[:, 1:opts.storedVectors-1] ))
            #     yVals = np.hstack(( Dg, yVals[:, 1:opts.storedVectors-1] ))
            #     rhoVals = np.hstack(( 1/np.real(Dg.conjugate().T@Dx), rhoVals[:, 1:opts.storedVectors-1] ))

            searchDir1 = determineSearchDirection(opts, gradf1, index, lastObjectiveUpdateIter)
            updateStepsize(searchDir0, searchDir1, Dx, tau1, tau0)
        x0 = x1
        f0 = f1
        gradf0 = gradf1
        tau0 = tau1
        searchDir0 = searchDir1

        x1 = x0 + tau0*searchDir0
        Dx = x1 - x0
        d1 = A*x1
        f1 = f(d1)
        # We now determine an appropriate stepsize for our algorithm using
        # Armijo-Goldstein condition
        backtrackCount = 0
        for backtrackCount in range(20):
            tmp = f0 + 0.1*tau0*np.real(searchDir0.conjugate().T@gradf0)
            # Break if f1 < tmp or f1 is sufficiently close to tmp (determined by error)
            # Avoids division by zero
            if f1 <= tmp:
                break
            # Stepsize reduced by factor of 5
            tau0 = tau0*0.2
            x1 = x0 + tau0*searchDir0
            Dx = x1 - x0
            d1 = A*x1
            f1 = f(d1)
        # from pdb import set_trace as bp
        # bp()
        # maxDiff = max(norm(Dx), maxDiff)
        # currentResidual = norm(Dx)/maxDiff
        container.iterationCount = index

        stopNow, updateObjectiveNow, maxDiff, tolerancePenalty= processIteration(index, startTime, Dx, maxDiff, opts, x1, updateObjectiveNow, container, residualTolerance, tolerancePenalty)
        if stopNow:
            break
    sol = x1
    return sol, container

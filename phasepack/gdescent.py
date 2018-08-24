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

from phasepack.util import Options, Container, ConvMatrix


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
            self.updateObjectivePeriod = 3
        try:
            self.searchMethod = opts.searchMethod
        except:
            self.updateObjectivePeriod = 'steepestDescent'

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


def determineSearchDirection():
    return np.array([5, 4])


def determineInitialStepsize():
    return 5

def updateStepsize():
    return 5


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


    x1 = x0
    d1 = A*x1
    Dx = 0

    startime = time.time()

    for iter in range(opts.maxIters):
        # Signal to update objective function after fixed number of iterations
        # have passed
        if iter-lastObjectiveUpdateIter == opts.updateObjectivePeriod:
            updateObjectiveNow = True
        # Update objective if flag is set
        if updateObjectiveNow:
            updateObjectiveNow = False
            lastObjectiveUpdateIter = iter
            f, gradf = updateObjective(x1, d1)
            f1 = f(d1)
            gradf1 = A.hmul(gradf(d1))
            if opts.searchMethod.lower() == 'lbfgs':
                # Perform LBFGS initialization
                yVals = np.zeros((n, opts.storedVectors))
                sVals = np.zeros((n, opts.storedVectors))
                rhoVals = np.zeros((1, opts.storedVectors))
            elif opts.searchMethod.lower() == 'ncg':
                # Perform NCG initialization
                lastNcgResetIter = iter
                unscaledSearchDir = zeros((n, 1))
            searchDir1 = determineSearchDirection()
            # Reinitialize stepsize to supplement new objective function
            tau1 = determineInitialStepsize()
        else:
            gradf1 = A.hmul(gradf(d1))
            Dg = gradf1 - gradf0

            if opts.searchMethod.lower() == 'lbfgs':
                # Update LBFGS stored vectors
                sVals = np.hstack(( Dx, sVals[:, 1:opts.storedVectors-1] ))
                yVals = np.hstack(( Dg, yVals[:, 1:opts.storedVectors-1] ))
                rhoVals = np.hstack(( 1/np.real(Dg.conjugate().T@Dx), rhoVals[:, 1:opts.storedVectors-1] ))

        searchDir1 = determineSearchDirection()
        updateStepsize()
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
        while backtrackCount <= 20:
            tmp = f0 + 0.1*tau0*np.real(searchDir0.conjugate().T * gradf0)
            # Break if f1 < tmp or f1 is sufficiently close to tmp (determined by error)
            # Avoids division by zero
            if f1 <= tmp:
                break

            backtrackCount = backtrackCount + 1
            # Stepsize reduced by factor of 5
            tau0 = tau0*0.2
            x1 = x0 + tau0*searchDir0
            Dx = x1 - x0
            d1 = A(x1)
            f1 = f(d1)
        stopNow = processIteration()
        if stopNow:
            break
    return

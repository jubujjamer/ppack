#!/usr/bin/python
# -*- coding: utf-8 -*-
""" File util.py

Last update: 15/08/2018

Usage:

"""
__version__ = "1.0.0"
__author__ = 'Juan M. Bujjamer'
__all__ = ['buildTestProblem']

import numpy as np
from numpy.random import multivariate_normal as mvnrnd

class Options(object):
    """ Option managing class.
    """
    def __init__(self, **kwargs):

        # Following options are relevant to every algorithm
        GeneralOptsDefault = {'algorithm': 'gerchbergsaxton',
                          'initMethod': 'optimal',
                          'isComplex': True,
                          'isNonNegativeOnly': False,
                          'maxIters':1E4,
                          'maxTime':300, # The maximum time the solver can run (unit: second). Note: since elapsed time will be checked at the end of each iteration,the real time the solver takes is the time of the iteration it goes beyond this maxTime.
                          'tol': 1E-4,
                          'verbose': 0, # Choose from [0, 1, 2]. If 0, print out nothing. If 1, print out status information in the end. If 2, print print out status information every round.
                          'recordTimes': True, # If the solver record time at each iteration.
                          'recordMeasurementErrors': False, #  If the solver compute and record measurement errors i.e. norm(abs(A*x-b0))/norm(b0) at each iteration.
                          'recordReconErrors':False, # If the solver record reconstruction errors i.e. norm(xt-x)/norm(x) at each iteration.
                          'recordResiduals': True, # If the solver record residuals (metric varies across solvers) at each iteration
                          'label': None, #  Can be used to choose a label for the algorithm to show up in the legend of a plot.  This is used when plotting results of benchmarks. The algorithm name is used by default if no label is specified.
                          # Following three options are unused by default
                          'xt': [], # The true signal. If it is provided, reconstruction error will be computed and used for stopping condition
                          'customAlgorithm': None, # Custom algorithm provided by user
                          'customx0': None, # Custom initializer provided by user
                          'initAngle': None # When the angle initializer is used, you must specify the angle between the true signal and the initializer
                          }

        SpecDefaults = {'custom': {},
                        'amplitudeflow': {
                            'searchMethod': 'steepestDescent', # Specifies how search direction for line search is chosen upon each iteration
                            'betaChoice': [] # Specifies how beta value is chosen (only used when search method is NCG)
                            },
                        'coordinatedescent': {
                            'indexChoice': 'greedy' #the rule for picking up index, choose from 'cyclic','random' ,'greedy'].
                            },
                        'fienup': {
                            'FienupTuning': 0.5, # Tunning parameter for Gerchberg-Saxton algorithm. It influences the update of the fourier domain value at each iteration.
                            'maxInnerIters': 10 # The max number of iterations the inner-loop solver  will have.
                            },

                        'gerchbergsaxton': {
                            'maxInnerIters': 10 # The max number of iterations  the inner-loop solver will have.
                            },
                        'kaczmarz': {
                            'indexChoice': 'cyclic' # the rule for picking up index, choose from ['cyclic','random']
                            },
                        'phasemax': {},
                        'phaselamp': {},
                        'phaselift': {
                            'regularizationPara' : 0.1 # This controls the weight of trace(X), where X=xx' in the objective function (see phaseLift paper for details)
                            },
                        'raf': {
                            'reweightPeriod': 20, # The maximum number of iterations that are allowed to occurr between reweights of objective function
                            'searchMethod': 'steepestDescent', # Specifies how search direction for line search is chosen upon each iteration
                            'betaChoice': 'HS' #  Specifies how beta value is chosen (only used when search
                            # method is NCG). Used only for NCG solver.
                        },
                        'rwf': {
                            'eta': 0.9, # Constant used to reweight objective function (see RWF paper for details)
                            'reweightPeriod': 20, # The maximum number of iterations that are allowed to occurr between reweights of objective function
                            'searchMethod': 'steepestDescent', # Specifies how search direction for line search is chosen upon each iteration
                            'betaChoice': 'HS' # Specifies how beta value is chosen (only used when search method is NCG) Used only for NCG solver
                        },
                        'sketchycgm': {
                            'rank': 1, # rank parameter. For details see Algorithm1 in the sketchyCGM paper.
                            'eta': 1 # stepsize parameter
                            },
                        'taf': {
                            'gamma': 0.7, # Constant used to truncate objective function (see paper for details). The maximum number of iterations that are allowed to occurr between truncations of objective function
                            'truncationPeriod': 20, # Specifies how search direction for line search is chosen upon each iteration
                            'searchMethod': 'steepestDescent',
                            'betaChoice': 'HS' # Specifies how beta value is chosen (only used when search method is NCG)Used only for NCG solver
                            },
                        'twf': {
                            'truncationPeriod': 20, # The maximum number of iterations that are allowed to occur between truncations of objective function
                            'searchMethod': 'steepestDescent', # Specifies how search direction for line search is chosen upon each iteration
                            'betaChoice': 'HS', # Specifies how beta value is chosen (only used when search method is NCG). Used only for NCG solver.
                            # Truncation parameters. These default values are defined as in the proposing paper for the case where line search is used.
                            'alpha_lb': 0.1,
                            'alpha_ub': 5,
                            'alpha_h': 6
                            },
                        'wirtflow': {
                            'searchMethod': 'steepestDescent', # Specifies how search direction for line search is chosen upon each iteration
                            'betaChoice': 'HS' # Specifies how beta value is chosen (only used when search method is NCG). Used only for NCG solver.
                            }
                        }
        for key, val in GeneralOptsDefault.items():
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, val)

        for key, val in SpecDefaults[self.algorithm.lower()].items():
            setattr(self, key, val)

    def getDefaultOpts(self):
        """ Obtain and apply default options that relevant to the specified algorithm
        """
        return

    def getExtraOpts(self):
        return

    def applyOpts(self):
        return


def buildTestProblem(m, n, isComplex=True, isNonNegativeOnly=False, dataType='Gaussian'):
    """ Creates and outputs random generated data and measurements according to user's choice. It is invoked in test*.m in order to build a test problem.

    Inputs:
      m(integer): number of measurements.
      n(integer): length of the unknown signal.
      isComplex(boolean, default=true): whether the signal and measurement
        matrix is complex. isNonNegativeOnly(boolean, default=false): whether
        the signal is real and non-negative.
      dataType(string, default='gaussian'): it currently supports
        ['gaussian', 'fourier'].

    Outputs:
      A: m x n measurement matrix/function handle.
      xt: n x 1 vector, true signal.
      b0: m x 1 vector, measurements.
      At: A n x m matrix/function handle that is the transpose of A.


    PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
    Christoph Studer, & Tom Goldstein
    Copyright (c) University of Maryland, 2017
    """
    if dataType.lower() == 'gaussian':
        mvnrnd(np.zeros(n), np.eye(n)/2, m)
        A = mvnrnd(np.zeros(n), np.eye(n)/2, m) + isComplex*1j*mvnrnd(np.zeros(n), np.eye(n)/2, m)
        At = A.T;
        x = mvnrnd(np.zeros(n), np.eye(n)/2) + isComplex*1j*mvnrnd(np.zeros(n), np.eye(n)/2)
        xt = x.reshape((-1, 1))
        print(A)
        print(xt)
        b0 = np.abs(A@xt);

    # elif dataType.lower() is 'fourier':
    # """Define the Fourier measurement operator.
    #    The operator 'A' maps an n-vector into an m-vector, then computes the fft on that m-vector to produce m measurements.
    # """
    #     # rips first 'length' entries from a vector
    #     rip = @(x,length) x(1:length);
    #     A = @(x) fft([x;zeros(m-n,1)]);
    #     At = @(x) rip(m*ifft(x),n);     % transpose of FM
    #     xt = (mvnrnd(zeros(1, n), eye(n)/2) + isComplex * 1i * ...
    #         mvnrnd(zeros(1, n), eye(n)/2))';
    #     b0 = abs(A(xt)); % Compute the phaseless measurements

    else:
        raise Exception('invalid dataType: %s', dataType);

    return [A, xt, b0, At]

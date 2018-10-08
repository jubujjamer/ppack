"""
This module provides classes to be used as containers of matrix operations.
It is deviced to keep efficient implementations of the different methods, so
the idea is to explicitly state here hoy left and right matrix multiplications
are calculated, eigenvectors and other useful matrix operations.

Classes
-------

ConvMarix           A Class containing typical convolution matrix operations.

Python version of the phasepack module by Juan M. Bujjamer, University of
Buenos Aires, 2018. Based on MATLAB implementation by Rohan Chandra,
Ziyuan Zhong, Justin Hontz, Val McCulloch, Christoph Studer,
& Tom Goldstein.
Copyright (c) University of Maryland, 2017
"""
__version__ = "1.0.0"
__author__ = 'Juan M. Bujjamer'

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs, lsqr
from numpy.random import multivariate_normal as mvnrnd

class ConvMatrix(object):
    """ Convolution matrix container.
    """
    def __init__(self, A=None, mv=None, rmv=None, shape=None):
        self.A = A
        if A is not None:
            self.shape = A.shape
            def mv(v):
                return A@v
            def rmv(v):
                return A.conjugate().T@v
        elif any([mv, rmv]):
            if shape:
                self.shape = shape
            else:
                raise Exception('If A is not given, its shape must be provided.')
            if not callable(mv):
                raise Exception('Input mv was not a function. Both mv and rmv shoud be functions, or both empty.')
            elif not callable(rmv):
                raise Exception('Input rmv was not a function. Both mv and rmv shoud be functions, or both empty.')
        else:
            # One of both inputs are needed for ConvMatrix creation
            raise Exception('A was not an ndarray, and both multiplication functions A(x) and At(x) were not provided.')
        self.m = self.shape[0]
        self.n = self.shape[1]
        self.matrix = LinearOperator(self.shape, matvec=mv, rmatvec=rmv)
        self.checkAdjoint()

    def checkAdjoint(self):
        """ Check that A and At are indeed ajoints of one another
        """
        y = np.random.randn(self.m);
        Aty = self.matrix.rmatvec(y)
        x = np.random.randn(self.n)
        Ax = self.matrix.matvec(x)
        innerProduct1 = Ax.conjugate().T@y
        innerProduct2 = x.conjugate().T@Aty
        error = np.abs(innerProduct1-innerProduct2)/np.abs(innerProduct1)
        assert error<1e-3, 'Invalid measurement operator:  At is not the adjoint of A.  Error = %.1f' % error
        print('Both matrices were adjoints', error)

    def hermitic(self):
        return

    def lsqr(self, b, tol, maxit, x0):
        """ Solution of the least squares problem for ConvMatrix
        Gkp, opts.tol/100, opts.maxInnerIters, gk
        """
        if b.shape[1]>0:
            b = b.reshape(-1)
        if x0.shape[1]>0:
            x0 = x0.reshape(-1)
        # x, istop, itn, r1norm = lsqr(self.matrix, b, atol=tol, btol=tol, iter_lim=maxit, x0=x0)
        ret = lsqr(self.matrix, b, atol=tol/100, btol=tol/100, iter_lim=maxit, x0=x0)
        x = ret[0]
        return x

    def hmul(self, x):
        """ Hermitic mutliplication
        returns At*x
        """
        return self.matrix.rmatvec(x)

    def __mul__(self, x):
        return self.matrix.matvec(x)

    def __matmul__(self, x):
        """Implementation of left ConvMatrix multiplication, i.e. A@x"""
        return self.matrix.dot(x)

    def __rmatmul__(self, x):
        """Implementation of right ConvMatrix multiplication, i.e. x@A"""
        return

    def __rmul__(self, x):
        if type(x) is float:
            lvec = np.ones(self.shape[1])*x
        else:
            lvec = x
        return x*self.A # This is not optimal

    def calc_yeigs(self, m, b0, idx):
        v = (idx*b0**2).reshape(-1)
        def ymatvec(x):
            return 1/m*self.matrix.rmatvec(v*self.matrix.matvec(x))
        yfun = LinearOperator((self.n, self.n), matvec=ymatvec)
        [eval, x0] = eigs(yfun, k=1, which='LR',tol=1E-5)
        return x0

def stopNow(opts, currentTime, currentResid, currentReconError):
    """
    Used in the main loop of many solvers (i.e.solve*.m) to
    check if the stopping condition(time, residual and reconstruction error)
    has been met and thus loop should be breaked.


    Note:
    This function does not check for max iterations since the for-loop
    in the solver already gurantee it.

    Inputs:
    opts(struct)                   :  consists of options. It is as
                  defined in solverPhaseRetrieval.
                  See its header or User Guide
                  for details.
    currentResid(real number)      :  Definition depends on the
                  specific algorithm used see the
                  specific algorithm's file's
                  header for details.
    currentReconError(real number) :  norm(xt-x)/norm(xt), where xt
                  is the m x 1 true signal,
                  x is the n x 1 estimated signal
                  at current iteration.
    Outputs:
    ifStop(boolean)                :  If the stopping condition has
                  been met.



    PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
    Christoph Studer, & Tom Goldstein
    Copyright (c) University of Maryland, 2017
    """
    if currentTime >= opts.maxTime:
        return True
    if len(opts.xt)>0:
        assert currentReconError, 'If xt is provided, currentReconError must be provided.'
        ifStop = currentReconError < opts.tol
    else:
        assert currentResid, 'If xt is not provided, currentResid must be provided.'
        ifStop = currentResid < opts.tol
    return ifStop

def  displayVerboseOutput(iter, currentTime, currentResid=None, currentReconError=None, currentMeasurementError=None):
    """ Prints out the convergence information at the current
    iteration. It will be invoked inside solve*.m if opts.verbose is set
    to be >=1.

    Inputs:
      iter(integer)                        : Current iteration number.
      currentTime(real number)             : Elapsed time so far(clock starts
                                             when the algorithm main loop
                                             started).
      currentResid(real number)            : Definition depends on the
                                             specific algorithm used see the
                                             specific algorithm's file's
                                             header for details.
      currentReconError(real number)       : relative reconstruction error.
                                             norm(xt-x)/norm(xt), where xt
                                             is the m x 1 true signal, x is
                                             the n x 1 estimated signal.

      currentMeasurementError(real number) : norm(abs(Ax)-b0)/norm(b0), where
                                             A is the m x n measurement
                                             matrix or function handle
                                             x is the n x 1 estimated signal
                                             and b0 is the m x 1
                                             measurements.

    PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
    Christoph Studer, & Tom Goldstein
    Copyright (c) University of Maryland, 2017
    """
    print('Iteration = %d' % iter, end=' |')
    print('IterationTime = %f' % currentTime, end=' |')
    if currentResid:
        print('Residual = %.1e' % currentResid, end=' |')
    if currentReconError:
        print('currentReconError = %.3f' %currentReconError, end=' |')
    if currentMeasurementError:
        print('MeasurementError = %.1e' %currentMeasurementError, end=' |')
    print()

def plotErrorConvergence(outs, opts):
    """
    This function plots some convergence curve according to the values of
    options in opts specified by user. It is used in all the test*.m scripts.
    Specifically,
    If opts.recordReconErrors is true, it plots the convergence curve of
    reconstruction error versus the number of iterations.
    If opts.recordResiduals is true, it plots the convergence curve of
    residuals versus the number of iterations.
    The definition of residuals is algorithm specific. For details, see the
    specific algorithm's solve*.m file.
    If opts.recordMeasurementErrors is true, it plots the convergence curve
    of measurement errors.

    Inputs are as defined in the header of solvePhaseRetrieval.m.
    See it for details.


    PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
    Christoph Studer, & Tom Goldstein
    Copyright (c) University of Maryland, 2017

    """

    # Plot the error convergence curve
    if opts.recordReconErrors:
        plt.figure()
        plt.semilogy(outs.reconErrors)
        plt.xlabel('Iterations')
        plt.ylabel('ReconErrors')
        plt.title('Convergence curve: %s' % opts.algorithm)
    if opts.recordResiduals:
        plt.figure()
        plt.semilogy(outs.residuals)
        plt.xlabel('Iterations')
        plt.ylabel('Residuals')
        plt.title('Convergence curve: %s' % opts.algorithm)
    if opts.recordMeasurementErrors:
        plt.figure()
        plt.semilogy(outs.measurementErrors);
        plt.xlabel('Iterations');
        plt.ylabel('MeasurementErros');
        plt.title('Convergence curve: %s' % opts.algorithm)
    plt.show()

def plotRecoveredVSOriginal(x,xt):
    """Plots the real part of the recovered signal against
    the real part of the original signal.
    It is used in all the test*.m scripts.

    Inputs:
          x:  a n x 1 vector. Recovered signal.
          xt: a n x 1 vector. Original signal.
    """
    plt.figure()
    plt.scatter(np.real(x), np.real(xt))
    plt.plot([-3, 3], [-3, 3], 'r')
    plt.title('Visual Correlation of Recovered signal with True Signal')
    plt.xlabel('Recovered Signal')
    plt.ylabel('True Signal')
    plt.show()

def buildTestProblem(m, n, isComplex=True, isNonNegativeOnly=False, dataType='Gaussian'):
    """ Creates and outputs random generated data and measurements according to user's choice.

    Inputs:
      m(integer): number of measurements.
      n(integer): length of the unknown signal.
      isComplex(boolean, default=true): whether the signal and measurement matrix is complex. isNonNegativeOnly(boolean, default=false): whether the signal is real and non-negative.
      dataType(string, default='gaussian'): it currently supports ['gaussian', 'fourier'].

    Outputs:
      A: m x n measurement matrix/function handle.
      xt: n x 1 vector, true signal.
      b0: m x 1 vector, measurements.
      At: A n x m matrix/function handle that is the transpose of A.
    """
    if dataType.lower() == 'gaussian':
        # mvnrnd(np.zeros(n), np.eye(n)/2, m)
        A = mvnrnd(np.zeros(n), np.eye(n)/2, m) + isComplex*1j*mvnrnd(np.zeros(n), np.eye(n)/2, m)
        At = A.conjugate().T
        x = mvnrnd(np.zeros(n), np.eye(n)/2) + isComplex*1j*mvnrnd(np.zeros(n), np.eye(n)/2)
        xt = x.reshape((-1, 1))
        b0 = np.abs(A@xt)

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

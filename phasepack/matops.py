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

class ConvolutionMatrix(object):
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
                print('If A is not given, its shape must be provided.')
                raise Exception(RuntimeError)
            if not callable(mv):
                print('Input mv was not a function. Both mv and rmv shoud be functions, or both empty.')
                raise Exception(RuntimeError)
            elif not callable(rmv):
                print('Input rmv was not a function. Both mv and rmv shoud be functions, or both empty.')
                raise Exception(RuntimeError)
        else:
            # One of both inputs are needed for ConvolutionMatrix creation
            print('A was not an ndarray, and both multiplication functions A(x) and At(x) were not provided.')
            raise Exception(RuntimeError)
        self.m = self.shape[0]
        self.n = self.shape[1]
        self.matrix = LinearOperator(self.shape, matvec=mv, rmatvec=rmv)
        self.check_adjoint()

    def validate_input(self, b0, opts):
        assert (np.abs(b0) == b0).all, 'b must be real-valued and non-negative'

        if opts.customx0:
            assert np.shape(opts.customx0) == (n, 1), 'customx0 must be a column vector of length n'

    def check_adjoint(self):
        """ Check that A and At are indeed ajoints of one another
        """
        y = np.random.randn(self.m);
        Aty = self.matrix.rmatvec(y)
        x = np.random.randn(self.n)
        Ax = self.matrix.matvec(x)
        inner_product1 = Ax.conjugate().T@y
        inner_product2 = x.conjugate().T@Aty
        error = np.abs(inner_product1-inner_product2)/np.abs(inner_product1)
        assert error<1e-3, 'Invalid measurement operator:  At is not the adjoint of A.  Error = %.1f' % error
        print('Both matrices were adjoints', error)

    def hermitic(self):
        return

    def lsqr(self, b, tol, maxit, x0):
        """ Solution of the least squares problem for ConvolutionMatrix
        Gkp, opts.tol/100, opts.max_inner_iters, gk
        """
        if b.shape[1]>0:
            b = b.reshape(-1)
        if x0.shape[1]>0:
            x0 = x0.reshape(-1)
        # x, istop, itn, r1norm = lsqr(self.matrix, b, atol=tol, btol=tol, iter_lim=maxit, x0=x0)
        ret = lsqr(self.matrix, b, damp=0.01, atol=tol/100, btol=tol/100, iter_lim=maxit, x0=x0)
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
        """Implementation of left ConvolutionMatrix multiplication, i.e. A@x"""
        return self.matrix.dot(x)

    def __rmatmul__(self, x):
        """Implementation of right ConvolutionMatrix multiplication, i.e. x@A"""
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

def build_test_problem(m, n, is_complex=True, is_non_negative_only=False, data_type='Gaussian'):
    """ Creates and outputs random generated data and measurements according to user's choice.

    Inputs:
      m(integer): number of measurements.
      n(integer): length of the unknown signal.
      isComplex(boolean, default=true): whether the signal and measurement matrix is complex. is_non_negative_only(boolean, default=false): whether the signal is real and non-negative.
      data_type(string, default='gaussian'): it currently supports ['gaussian', 'fourier'].

    Outputs:
      A: m x n measurement matrix/function handle.
      xt: n x 1 vector, true signal.
      b0: m x 1 vector, measurements.
      At: A n x m matrix/function handle that is the transpose of A.
    """
    if data_type.lower() == 'gaussian':
        # mvnrnd(np.zeros(n), np.eye(n)/2, m)
        A = mvnrnd(np.zeros(n), np.eye(n)/2, m) + is_complex*1j*mvnrnd(np.zeros(n), np.eye(n)/2, m)
        At = A.conjugate().T
        x = mvnrnd(np.zeros(n), np.eye(n)/2) + is_complex*1j*mvnrnd(np.zeros(n), np.eye(n)/2)
        xt = x.reshape((-1, 1))
        b0 = np.abs(A@xt)

    # elif data_type.lower() is 'fourier':
    # """Define the Fourier measurement operator.
    #    The operator 'A' maps an n-vector into an m-vector, then computes the fft on that m-vector to produce m measurements.
    # """
    #     # rips first 'length' entries from a vector
    #     rip = @(x,length) x(1:length);
    #     A = @(x) fft([x;zeros(m-n,1)]);
    #     At = @(x) rip(m*ifft(x),n);     % transpose of FM
    #     xt = (mvnrnd(zeros(1, n), eye(n)/2) + is_complex * 1i * ...
    #         mvnrnd(zeros(1, n), eye(n)/2))';
    #     b0 = abs(A(xt)); % Compute the phaseless measurements

    else:
        print('Invalid data_type: %s', data_type)
        raise Exception(TypeError)
    return [A, xt, b0, At]

class FourierOperator(object):
    """ Linear operator creator from problem data.

        Create a measurement operator that maps a vector of pixels into Fourier
        measurements using a collection of PSF's.
    """
    def __init__(self, psf_collection):
        npsf, nrows, ncols = psf_collection.shape
        self.psf_collection = psf_collection
        self.npsf = npsf
        self.nrows = nrows
        self.ncols = ncols

    def mv(self, xvec):
        """The fourier mask matrix operator
        As the reconstruction method stores iterates as vectors, this
        function needs to accept a vector as input.
        """
        xvec2d = xvec.reshape(self.nrows, self.ncols)
        bvec2d = np.fft.fft2(self.psf_collection*xvec2d)
        bvec_array = bvec2d.reshape(self.npsf*self.nrows*self.ncols, 1)
        return bvec_array

    def rmv(self, bvec):
        # The adjoint/transpose of the measurement operator
        # The reconstruction method stores measurements as vectors, so we need
        # to accept a vector input, and convert it back into a 3D array of
        # Fourier measurements.
        bvec2d_array = bvec.reshape(self.npsf, self.nrows, self.ncols)
        conv_images = np.fft.ifft2(bvec2d_array)
        conv_images = conv_images*self.psf_collection*self.nrows*self.ncols
        imagesvec = np.sum(conv_images,axis=0).reshape(-1, 1)
        return imagesvec


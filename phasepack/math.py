""" File gdescent.py

Last update: 24/08/2018

Usage:

"""
__version__ = "1.0.0"
__author__ = 'Juan M. Bujjamer'
__all__ = ['gradient_descent_solver']
import numba
from numba import jit
import time
import warnings
import numpy as np
from numpy.linalg import norm
from numpy.random import randn

from pdb import set_trace as bp
from phasepack.containers import Options, ResultsContainer, display_verbose_output
from phasepack.matops import ConvolutionMatrix

class gd_options(object):
    """ Gradient Descent Algorithm Options manager.
    """
    def __init__(self, opts):
        self.max_iters = opts.max_iters
        self.max_time = opts.max_time
        self.tol = opts.tol
        self.verbose = opts.verbose
        self.record_times = opts.record_times
        self.record_residuals = opts.record_residuals
        self.record_measurement_errors = opts.record_measurement_errors
        self.record_recon_errors = opts.record_recon_errors
        self.xt = opts.xt
        self.update_objective_period = opts.truncation_period
        self.search_method = opts.search_method
        self.beta_choice = opts.beta_choice

        try:
            self.update_objective_period = opts.update_objective_period
        except:
            self.update_objective_period = -np.inf
        try:
            self.tolerance_penalty_limit = opts.tolerance_penalty_limit
        except:
            self.tolerance_penalty_limit = 3
        try:
            self.search_method = opts.search_method
        except:
            self.search_method = 'steepest_descent'

        if opts.search_method.lower() == 'lbfgs':
            try:
                 self.stored_vectors = opts.stored_vectors
            except:  self.stored_vectors= 5
        if opts.search_method.lower() == 'ncg':
            try:
                self.beta_choice = opts.beta_choice
            except:
                self.beta_choice = 'HS'
        try:
            self.ncg_reset_period = opts.ncg_reset_period
        except:
            self.ncg_reset_period = 100

def determine_search_direction(opts, gradf1, index, last_objective_update_iter, last_ncg_reset_iter=None,
                             unscaled_search_dir=None, rho_vals=None, s_vals=None, y_vals=None, Dg=None):
    """
        Determine search direction for next iteration based on specified search
        methodself.

        TODO: Manage None values
    """
    if opts.search_method.lower() == 'steepest_descent':
        search_dir = -gradf1
    elif opts.search_method.lower() == 'ncg':
        search_dir = -gradf1
        # Reset NCG progress after specified number of iterations have
        # passed
        if index - last_ncg_reset_iter == opts.ncg_reset_period:
            unscaled_search_dir = zeros(n)
            last_ncg_reset_iter = index
        # Proceed only if reset has not just occurred
        if index != last_ncg_reset_iter:
            if opts.beta_choice.lower() == 'hs':
                # Hestenes-Stiefel
                beta = -np.real(gradf1.congujate().T*Dg)/np.real(unscaled_search_dir.congujate().T*Dg)
            elif opts.beta_choice.lower() == 'fr':
                # Fletcher-Reeves
                beta = norm(gradf1)**2 / norm(gradf0)^2;
            elif opts.beta_choice.lower == 'pr':
                #Polak-Ribiere
                beta = np.real(gradf1.conjugate().T*Dg)/norm(gradf0)**2
            elif opts.beta_choice.lower ==  'dy':
                # Dai-Yuan
                beta = norm(gradf1)**2 / np.real(unscaled_search_dir.conjugate().T*Dg)

            search_dir = search_dir + beta * unscaled_search_dir;
        unscaled_search_dir = search_dir
    elif opts.search_method.lower() == 'lbfgs':
        search_dir = -gradf1
        iters = np.min(index-last_objective_update_iter, opts.stored_vectors);
        if iters > 0:
            alphas = np.zeros(iters)
            # First loop
            for j in range(iters):
                alphas[j] = rho_vals[j]*np.real(s_vals[:,j].conjugate().T*search_dir)
                search_dir = search_dir - alphas[j] * y_vals[:,j]

            # Scaling of search direction
            gamma = np.real(Dg.conjugate().T*Dx)/(Dg.conjugate().T*Dg)
            search_dir = gamma*search_dir
            # Second loop
            for j in range(iters, 1, -1):
                beta = rho_vals[j]*np.real(y_vals[:,j].congugate().T*search_dir)
                search_dir = search_dir + (alphas[j]-beta)*s_vals[:,j]
            search_dir = 1/gamma*search_dir
            search_dir = norm(gradf1)/norm(search_dir)*search_dir
    # Change search direction to steepest descent direction if current
    # direction is invalid
    # if any(np.isnan(search_dir)) or any(np.isinf(search_dir)):
    if np.isnan(search_dir).any or np.isinf(search_dir).any:
        search_dir = -gradf1
    # Scale current search direction match magnitude of gradient
    search_dir = norm(gradf1) / norm(search_dir)*search_dir
    return search_dir


def determine_initial_stepsize(A, x0, gradf):
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

def update_stepsize(search_dir0, search_dir1, Dx, tau1, tau0):
    """ Update stepsize when objective update has not just occurred (adopted from
        FASTA.m)
    """
    Ds = search_dir0 - search_dir1
    dotprod = np.real(np.vdot(Dx, Ds))
    tau_s = norm(Dx)**2/dotprod  # First BB stepsize rule
    tau_m = dotprod/norm(Ds)**2 # Alternate BB stepsize rule
    tau_m = max(tau_m, 0)
    if 2*tau_m > tau_s:   #  Use "Adaptive"  combination of tau_s and tau_m
        tau1 = tau_m
    else:
        tau1 = tau_s - tau_m/2  # Experiment with this param
    if tau1 <= 0 or np.isinf(tau1) or np.isnan(tau1): #  Make sure step is non-negative
        tau1 = tau0*1.5  # let tau grow, backtracking will kick in if stepsize is too big
    return tau1

def process_iteration(index, start_time, Dx, max_diff, opts, x1, update_objective_now, container,
                     residual_tolerance, tolerance_penalty):
    current_solve_time = time.monotonic()-start_time
    max_diff = max(norm(Dx), max_diff)
    current_residual = norm(Dx)/max_diff
    stop_now = False
    current_recon_error = None
    current_measurement_error = None
    # Obtain recon error only if true solution has been provided
    if opts.xt:
        recon_estimate = (x1.conjugate().T@opts.xt)/(x1.conjugate().T*x1)*x1
        current_recon_error = norm(opts.xt-recon_estimate)/norm(opts.xt)
    if opts.record_times:
        container.append_record_time(current_solve_time)
    if opts.record_residuals:
        container.append_residual(current_residual)
    if opts.record_measurement_errors:
        current_measurement_error = norm(abs(d1) - b0) / norm(b0)
        container.append_measurement_error(current_measurement_error)
    if opts.record_recon_errors:
        assert opts.xt, 'You must specify the ground truth solution if the "record_recon_errors" flag is set to true.  Turn this flag off, or specify the ground truth solution.'
        container.append_recon_error(current_recon_error)
    if opts.verbose == 2:

        display_verbose_output(index, current_solve_time, current_residual, current_recon_error, current_measurement_error)

    # Terminate if solver surpasses maximum allocated timespan
    if current_solve_time > opts.max_time:
        stop_now = True
    # If user has supplied actual solution, use recon error to determine
    # termination
    if opts.xt and current_recon_error <= opts.tol:
        stop_now = True

    if current_residual <= residual_tolerance:
        # Give algorithm chance to recover if stuck at local minimum by
        # forcing update of objective function
        update_objective_now = True
        # If algorithm continues to get stuck, terminate
        tolerance_penalty = tolerance_penalty + 1

        if tolerance_penalty >= opts.tolerance_penalty_limit:
            stop_now = True
    return stop_now, update_objective_now, max_diff, tolerance_penalty

def gradient_descent_solver(A, At, x0, b0, update_objective, opts):
    """% -------------------------gradient_descent_solver.m--------------------------------

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

    max_iters (required) - The maximum number of iterations that are
    allowed to
        occur.

    max_time (required) - The maximum amount of time in seconds the
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

    record_times (required) - Whether the algorithm should store the total
        processing time upon each iteration in a list to be obtained via
        output.

    record_residuals (required) - Whether the algorithm should store the
        relative residual values upon each iteration in a list to be
        obtained via output.

    record_measurement_errors (required) - Whether the algorithm should
    store
        the relative measurement errors upon each iteration in a list to
        be obtained via output.

    record_recon_errors (required) - Whether the algorithm should store the
        relative reconstruction errors upon each iteration in a list to
        be obtained via output. This parameter can only be set 'true'
        when the additional parameter 'xt' is non-empty.

    xt (required) - The true signal to be estimated, or an empty vector
    if the
        true estimate was not provided.

    search_method (optional) - A string representing the method used to
        determine search direction upon each iteration. Must be one of
        {'steepest_descent', 'NCG', 'LBFGS'}. If equal to
        'steepest_descent', then the steepest descent search method is
        used. If equal to 'NCG', a nonlinear conjugate gradient method is
        used. If equal to 'LBFGS', a Limited-Memory BFGS method is used.
        Default value is 'steepest_descent'.

    update_objective_period (optional) - The maximum number of iterations
    that
        are allowed to occur between updates to the objective function.
        Default value is infinite (no limit is applied).

    tolerance_penalty_limit (optional) - The maximum tolerable penalty
    caused by
        surpassing the tolerance threshold before terminating. Default
        value is 3.

    beta_choice (optional) - A string representing the choice of the value
        'beta' when a nonlinear conjugate gradient method is used. Must
        be one of {'HS', 'FR', 'PR', 'DY'}. If equal to 'HS', the
        Hestenes-Stiefel method is used. If equal to 'FR', the
        Fletcher-Reeves method is used. If equal to 'PR', the
        Polak-Ribi�re method is used. If equal to 'DY', the Dai-Yuan
        method is used. This field is only used when search_method is set
        to 'NCG'. Default value is 'HS'.

    ncg_reset_period (optional) - The maximum number of iterations that are
        allowed to occur between resettings of a nonlinear conjugate
        gradient search direction. This field is only used when
        search_method is set to 'NCG'. Default value is 100.

    stored_vectors (optional) - The maximum number of previous iterations
    of
        which to retain LBFGS-specific iteration data. This field is only
        used when search_method is set to 'LBFGS'. Default value is 5.
    """

    # Length of input signal
    n = len(x0)
    if opts.xt:
        residual_tolerance = 1.0e-13
    else:
        residual_tolerance = opts.tol

    # Iteration number of last objective update
    last_objective_update_iter = 0
    # Total penalty caused by surpassing tolerance threshold
    tolerance_penalty = 0
    # Whether to update objective function upon next iteration
    update_objective_now = True
    # Maximum norm of differences between consecutive estimates
    max_diff = -np.inf

    current_solve_time = 0
    current_measurement_error = []
    current_residual = []
    current_recon_error = []
    container = ResultsContainer(opts)

    x1 = x0
    d1 = A*x1
    Dx = 0

    start_time = time.monotonic()
    for index in range(opts.max_iters):
        # Signal to update objective function after fixed number of iterations
        # have passed
        if index-last_objective_update_iter == opts.update_objective_period:
            update_objective_now = True
        # Update objective if flag is set
        if update_objective_now:
            update_objective_now = False
            last_objective_update_iter = index
            f, gradf = update_objective(x1, d1)
            f1 = f(d1)
            gradf1 = A.hmul(gradf(d1))
            # if opts.search_method.lower() == 'lbfgs':
            #     # Perform LBFGS initialization
            #     y_vals = np.zeros((n, opts.stored_vectors))
            #     s_vals = np.zeros((n, opts.stored_vectors))
            #     rho_vals = np.zeros((1, opts.stored_vectors))
            # elif opts.search_method.lower() == 'ncg':
            #     # Perform NCG initialization
            #     last_ncg_resetindex= iter
            #     unscaled_search_dir = zeros((n, 1))

            search_dir1 = determine_search_direction(opts, gradf1, index, last_objective_update_iter)

            # Reinitialize stepsize to supplement new objective function
            tau1 = determine_initial_stepsize(A, x0, gradf)
        else:
            gradf1 = A.hmul(gradf(d1))
            Dg = gradf1 - gradf0
            # if opts.search_method.lower() == 'lbfgs':
            #     # Update LBFGS stored vectors
            #     s_vals = np.hstack(( Dx, s_vals[:, 1:opts.stored_vectors-1] ))
            #     y_vals = np.hstack(( Dg, y_vals[:, 1:opts.stored_vectors-1] ))
            #     rho_vals = np.hstack(( 1/np.real(Dg.conjugate().T@Dx), rho_vals[:, 1:opts.stored_vectors-1] ))

            search_dir1 = determine_search_direction(opts, gradf1, index, last_objective_update_iter)
            update_stepsize(search_dir0, search_dir1, Dx, tau1, tau0)
        x0 = x1
        f0 = f1
        gradf0 = gradf1
        tau0 = tau1
        search_dir0 = search_dir1

        x1 = x0 + tau0*search_dir0
        Dx = x1 - x0
        d1 = A*x1
        f1 = f(d1)
        # We now determine an appropriate stepsize for our algorithm using
        # Armijo-Goldstein condition
        tmp0 = 0.1*tau0*np.real(search_dir0.conjugate().T@gradf0)
        for backtrack_count in range(20):
            # Break if f1 < tmp or f1 is sufficiently close to tmp (determined by error)
            # Avoids division by zero
            if f1 <= (tmp0 + f0):
                break
            tmp0 = tmp0*0.2
            # Stepsize reduced by factor of 5
            tau0 = tau0*0.2
            x1 = x0 + tau0*search_dir0
            Dx = x1 - x0
            d1 = A*x1
            f1 = f(d1)
        # from pdb import set_trace as bp
        # bp()
        # max_diff = max(norm(Dx), max_diff)
        # current_residual = norm(Dx)/max_diff
        container.iteration_count = index

        stop_now, update_objective_now, max_diff, tolerance_penalty= process_iteration(index, start_time, Dx, max_diff, opts, x1, update_objective_now, container, residual_tolerance, tolerance_penalty)
        if stop_now:
            break
    sol = x1
    return sol, container

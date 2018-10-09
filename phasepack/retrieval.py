import numpy as np

from phasepack.matops import ConvMatrix
from phasepack.initializers import init_spectral, init_optimal_spectral
from phasepack.containers import Options
import phasepack.solvers as sv

class Retrieval(object):
    def __init__(self, A, b0, opts):
        self.b0 = b0
        # If opts is not provided, create it
        if opts is None:
            opts = Options()
        # If A was not instantiated as a ConvMatrix
        if type(A) == np.ndarray:
            A = ConvMatrix(A)
        self.A = A
        self.opts = opts

    def _select_initializer(self):
        chosen_opt = self.opts.init_method.lower()
        if chosen_opt == 'truncatedspectral' or chosen_opt == 'truncated':
            return init_spectral(A=self.A, At=None, b0=self.b0, is_truncated=True,
                                is_scaled=True, verbose=self.opts.verbose)
        elif chosen_opt == 'spectral':
            return init_spectral(A=self.A, At=None, b0=self.b0,
                                is_truncated=False, is_scaled=True, verbose=self.opts.verbose)
        elif chosen_opt == 'optimal' or chosen_opt == 'optimalspectral':
            return init_optimal_spectral(A=self.A, At=None, b0=self.b0,
                                is_scaled=True,
                                verbose=self.opts.verbose)
    #    elif chosen_opt == 'amplitudespectral' or chosen_opt == 'amplitude':
    #        return init_amplitude(A=A, At=At, b0=b0, n=n, verbose=opts.verbose)
    #    elif chosen_opt == 'weightedspectral' or chosen_opt == 'weighted':
    #        return init_weighted(A=A, At=At, b0=b0, n=n, verbose=opts.verbose)
    #    elif chosen_opt == 'orthogonalspectral' or chosen_opt == 'orthogonal':
    #        return init_orthogonal(A=A, At=At, b0=b0, n=n, verbose=opts.verbose)
    #    elif chosen_opt == 'angle':
    #        return init_angle(xt=opts.xt, angle=opts.init_angle)
        # case 'angle'
        #     assert(isfield(opts,'xt'),'The true solution, opts.xt, must be specified to use the angle initializer.')
        #     assert(isfield(opts,'init_angle'),'An angle, opts.init_angle, must be specified (in radians) to use the angle initializer.')
        #     x0 = init_angle(opts.xt, opts.init_angle);
        # case 'custom'
        #     x0 = opts.customx0;
        else:
            raise Exception('Unknown initialization option.')
        return x0


    def _select_solver(self, x0):
        choose_algorithm = {'custom':sv.opts_custom_algorithm,
                       'amplitudeflow':sv.solve_amplitude_flow,
                       'coordinatedescent':sv.solve_coordinate_descent,
                       'fienup':sv.solve_fienup,
                       'gerchbergsaxton':sv.solve_gerchberg_saxton,
                       'kaczmarz':sv.solve_kaczmarz_simple,
                       'phasemax':sv.solve_phase_max,
                       'phaselamp':sv.solve_phase_lamp,
                       'phaselift':sv.solve_phase_lift,
                       'raf':sv.solve_raf,
                       'rwf':sv.solve_rwf,
                       'sketchycgm':sv.solve_sketchy_cgm,
                       'taf':sv.solve_taf,
                       'twf':sv.solve_twf,
                       'wirtflow':sv.solve_wirt_flow}
        sol, outs = choose_algorithm[self.opts.algorithm.lower()](self.A, None,
                                    self.b0, x0, self.opts)
        return sol, outs, self.opts


    def solve_phase_retrieval(self):
        """ This method solves the problem:
                              Find x given b0 = |Ax+epsilon|
         Where A is a m by n complex matrix, x is a n by 1 complex vector, b0 is a
         m by 1 real,non-negative vector and epsilon is a m by 1 vector. The user
         supplies function handles A, At and measurement b0. Note: The unknown
         signal to be recovered must be 1D for our interface.
         Inputs:
          A     : A m x n matrix (or optionally a function handle to a method) that returns A*x
          At    : The adjoint (transpose) of 'A.' It can be a n x m matrix or a function handle.
          b0    : A m x 1 real,non-negative vector consists of  all the measurements.
          n     : The size of the unknown signal. It must be provided if A is a function handle.
          opts  : An optional struct with options.  The commonly used fields of 'opts' are:
                     init_method              : (string,
                     default='truncated_spectral') The method used
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
                                               custom_algorithm.
                     max_iters                : (integer, default=1000) The
                                               maximum number of
                                               iterations allowed before
                                               termination.
                     max_time                 : (positive real number,
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
                     record_measurement_errors : (boolean, default=false) Whether
                                               to compute and record
                                               error(i.e.
                                               norm(|Ax|-b0)/norm(b0)) at each
                                               iteration.
                     record_residuals         : (boolean, default=true) If it's
                                               true, residual will be
                                               computed and recorded at each
                                               iteration. If it's false,
                                               residual won't be recorded.
                                               Residual also won't be computed
                                               if xt is provided. Note: The
                                               error metric of residual varies
                                               across solvers.
                     record_recon_errors       : (boolean, default=false) Whether
                                               to record
                                               reconstruction error. If it's
                                               true, opts.xt must be provided.
                                               If xt is provided reconstruction
                                               error will be computed regardless
                                               of this flag and used for
                                               stopping condition.
                     record_times             : (boolean, default=true) Whether
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
                            >> opts.tol=1e-8; >> opts.max_iters = 100;


         Outputs:
          sol               : The approximate solution outs : A struct with
                              convergence information
          iteration_count    : An integer that is  the number of iterations the
                              algorithm runs.
          solve_times        : A vector consists  of elapsed (exist when
                              record_times==true) time at each iteration.
          measurement_errors : A vector consists of the errors (exist when
                              record_measurement_errors==true)   i.e.
                              norm(abs(A*x-b0))/norm(b0) at each iteration.
          recon_errors       : A vector consists of the reconstruction (exist
                              when record_recon_errors==true) errors
                              i.e. norm(xt-x)/norm(xt) at each iteration.
          residuals         : A vector consists of values that (exist when
                              record_residuals==true)  will be compared with
                              opts.tol for stopping condition  checking.
                              Definition varies across solvers.
          opts              : A struct that contains fields used by the solver.
                              Its possible fields are the same as the input
                              parameter opts.

       For more details and more options in opts, see the Phasepack user guide.
        """

        # Check that inputs are of valid datatypes and sizes
        self.A.validate_input(b0=self.b0, opts=self.opts)
        # # Initialize x0
        x0 = self._select_initializer()
        # Truncate imaginary components of x0 if working with real values
        if not self.opts.is_complex:
            x0 = np.real(x0)
        elif self.opts.is_non_negative_only:
            warnings.warn('opts.is_non_negative_only will not be used when the signal is complex.');
        # Solve the problem using the specified algorithm
        [sol, outs, opts] = self._select_solver(x0=x0)
        return sol, outs, opts

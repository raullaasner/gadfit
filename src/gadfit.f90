!!****m* GADfit/gadfit
!! 
!! COPYRIGHT
!! 
!! Copyright (C) 2014-2015 Raul Laasner
!! This file is distributed under the terms of the GNU General Public
!! License, see LICENSE in the root directory of the present
!! distribution or http://gnu.org/copyleft/gpl.txt .
!!
!! FUNCTION
!!
!! Procedures for reading in data, performing the fitting procedure,
!! and printing the results. As a small guide, the following
!! constructs are often encountered in the code:
!!   Loop over all datasets:
!!     do i = 1, size(fitfuncs)
!!   Loop over all data points:
!!     do i = 1, size(x_data)
!!   Loop over data points attributed to this image:
!!     do i = 1, size(fitfuncs)
!!       do j = img_bounds(i), img_bounds(i+1)-1
!!
!! SOURCE
#include <config.h>

module gadfit

  use, intrinsic :: iso_fortran_env, only: real32, real64
  use ad
  use fitfunction,    only: fitfunc, safe_deallocate
  use gadf_constants, only: kp
  use linalg,         only: potr_f08 ! A*X=B with A symmetric
  use messaging
  use misc,           only: string, len, swap, timer, safe_close
#if !defined HAS_CO_SUM || defined QUAD_PRECISION
  use misc,           only: co_sum
#endif
  use numerical_integration

  implicit none
  
  private
  protected :: fitfuncs
  public :: gadf_init, gadf_add_dataset, gadf_set, gadf_set_errors, &
       & gadf_set_verbosity, gadf_fit, gadf_print, gadf_close, GLOBAL, LOCAL, &
       & GLOBAL_AND_LOCAL, NONE, SQRT_Y, PROPTO_Y, INVERSE_Y, USER

  enum, bind(c)
     enumerator:: GLOBAL, LOCAL, GLOBAL_AND_LOCAL         ! For I/O
     enumerator:: NONE, SQRT_Y, PROPTO_Y, INVERSE_Y, USER ! Data errors
  end enum

  interface gadf_set
     module procedure set_int_local_real, set_int_local_real32, &
          & set_int_global_real, set_int_global_real32, set_char_local_real, &
          & set_char_local_real32, set_char_global_real, set_char_global_real32
  end interface gadf_set

  ! FITTING FUNCTION
  
  ! The are as many instances of the fitting function as there are
  ! datasets.
#ifdef POLYM_ARRAY_SUPPORT
  class(fitfunc), allocatable, public :: fitfuncs(:)
#else
  ! Currently gfortran is unable to directly access the allocatable
  ! components of a polymorphic array.
  class(fitfunc), allocatable, target, public :: fitfuncs(:)
  class(fitfunc), pointer :: fitfunc_tmp => null()
#endif
  ! Active parameters are accessed with
  ! fitfuncs(i)%pars(active_pars(j)), where j = 1...<number of active
  ! parameters>.
  integer, allocatable :: active_pars(:)
  ! Information about which active/passive parameters are global
  logical, allocatable :: is_global(:)

  ! INPUT

  ! Input data. 'weights' may be explicitly given by the user or
  ! computed from the input data according to data_error_type.
  real(kp), allocatable :: x_data(:), y_data(:), weights(:)
  integer :: data_error_type
  ! I/O device numbers corresponding to the input data files
  integer, allocatable :: data_units(:)
  ! Information about where one dataset ends and another begins in
  ! x_data, y_data, and weights.
  integer, allocatable :: data_positions(:)
  ! Counts calls to gadf_set and warns if some parameters are not
  ! initialized. (Not terribly important.)
  integer :: set_count

  ! OPTIMIZATION PROCEDURE

  ! Number of iterations performed. This is kept global because it
  ! contains information about whether a fitting procedure has been
  ! performed or not. All other optimization related variables are the
  ! local variables of gadf_fit.
  integer :: iterations

  ! FOR DISPLAYING RESULTS

  ! These are all user controlled variables.
  integer :: show_scope, show_digits, out_unit
  logical :: &
       & show_timing, show_memory, show_workloads, show_delta1, show_delta2, &
       & show_cos_phi, show_grad_chi2, show_uphill, show_acc
  
  ! TIMING
  
  type(timer) :: Jacobian_timer[*], chi2_timer[*], linalg_timer[*]
  type(timer) :: omega_timer[*], main_loop_timer[*]

contains
  !!***

  !!****f* gadfit/gadf_init
  !!
  !! FUNCTION
  !!
  !! Initializes the fitting functions and other work variables and
  !! reserves memory for AD and numerical integration.
  !!
  !! INPUTS
  !!
  !! f - fitting function
  !! num_datasets (optional) - number of datasets
  !!
  !! These are passed directly to ad_init:
  !! ad_memory  (optional)
  !! sweep_size (optional)
  !! trace_size (optional)
  !! const_size (optional)
  !!
  !! These are passed directly to init_integration or
  !! init_integration_dbl:
  !! rel_error        (optional)
  !! rel_error_inner  (optional)
  !! ws_size          (optional)
  !! ws_size_inner    (optional)
  !! integration_rule (optional)
  !!
  !! SOURCE
  subroutine gadf_init(f, num_datasets, sweep_size, trace_size, &
       & const_size, ws_size, ws_size_inner, integration_rule, ad_memory, &
       & rel_error_inner, rel_error)
    use, intrinsic :: iso_fortran_env, only: output_unit
    class(fitfunc), intent(in) :: f
    integer, intent(in), optional :: &
         & num_datasets, sweep_size, trace_size, const_size, ws_size, &
         & ws_size_inner, integration_rule
    character(*), intent(in), optional :: ad_memory
    real(kp), intent(in), optional :: rel_error_inner, rel_error 
    integer :: i
    if (present(num_datasets)) then
       allocate(fitfuncs(num_datasets), mold=f, stat=err_stat, errmsg=err_msg)
    else
       allocate(fitfuncs(1), mold=f, stat=err_stat, errmsg=err_msg)
    end if
    call check_err(__FILE__, __LINE__)
    do i = 1, size(fitfuncs)
       call fitfuncs(i)%init()
    end do
    allocate(data_units(size(fitfuncs)), &
         & active_pars(size(fitfuncs(1)%pars)), &
         & is_global(size(fitfuncs(1)%pars)), &
         & data_positions(size(fitfuncs)+1), &
         & stat=err_stat, errmsg=err_msg)
    call check_err(__FILE__, __LINE__)
    call ad_init_reverse(ad_memory, sweep_size, trace_size, const_size)
    if (present(rel_error_inner) .or. present(ws_size_inner)) then
       call init_integration_dbl(rel_error_inner, rel_error, ws_size_inner, &
            & ws_size, integration_rule)
    else if (present(rel_error) .or. present(ws_size) .or. &
         & present(integration_rule)) then
       call init_integration(rel_error, ws_size, integration_rule)
    end if
    data_positions = 1
    set_count = 0
    ! Since some procedures may never be called, their defaults are
    ! set here.
    iterations = 0
    data_error_type = NONE
    show_scope = GLOBAL_AND_LOCAL
    show_digits = 7
    show_timing    = .false.; show_memory = .false.; show_workloads = .false.
    show_delta1    = .false.; show_delta2 = .false.; show_cos_phi   = .false.
    show_grad_chi2 = .false.; show_uphill = .false.; show_acc       = .false.
    out_unit = output_unit
  end subroutine gadf_init
  !!***

  !!****f* gadfit/gadf_add_dataset
  !!
  !! FUNCTION
  !!
  !! Opens a data file and determines the number of data points. If
  !! successful, the I/O device number is saved and the actual reading
  !! takes place in read_data.
  !!
  !! SOURCE
  subroutine gadf_add_dataset(path)
    use, intrinsic :: iso_fortran_env, only: iostat_end
    character(*), intent(in) :: path
    real(kp) :: dummy
    integer :: data_size
    logical :: file_exists
    integer :: i
    if (.not. allocated(fitfuncs)) &
         & call error(__FILE__, __LINE__, &
         & 'Number of datasets is undetermined. Call gadf_init first.')
    main: do i = 1, size(fitfuncs)
       slot_available: if (data_positions(i+1) == 1) then
          open(newunit=data_units(i), file=path, status='old', action='read', &
               & form='formatted', iostat=err_stat, iomsg=err_msg)
          call check_err(__FILE__, __LINE__)
          err_stat = 0; data_size = 0
          inquire(file=path, exist=file_exists)
          if (file_exists) then
             do while (err_stat /= iostat_end)
                read(data_units(i), *, iostat=err_stat) dummy
                if (err_stat == 0) data_size = data_size + 1
             end do
          end if
          if (data_size == 0) &
               & call error(__FILE__, __LINE__, &
               & path//' contains no valid data points.')
          data_positions(i+1) = data_positions(i) + data_size
          rewind(data_units(i))
          exit main
       end if slot_available
    end do main
  end subroutine gadf_add_dataset
  !!***

  !!****f* gadfit/gadf_set
  !!
  !! FUNCTION
  !!
  !! Defines the parameter as local, sets its value, and marks it
  !! either active or passive.
  !!
  !! INPUTS
  !!
  !! dataset_i - dataset index
  !! par - parameter index (fitfuncs(dataset_i)%pars(par))
  !! val - parameter value
  !! active (optional) - if .true., marks the variable as active
  !!
  !! SOURCE
  subroutine set_int_local_real(dataset_i, par, val, active)
    integer, intent(in) :: dataset_i, par
    real(kp), intent(in) :: val
    logical, intent(in), optional :: active
    if (.not. allocated(fitfuncs)) &
         & call error(__FILE__, __LINE__, &
         & 'Number of datasets is undetermined. Call gadf_init first.')
    if (dataset_i > size(fitfuncs)) &
         & call error(__FILE__, __LINE__, 'Invalid dataset index. &
         &Call gadf_init with the correct number of datasets.')
    is_global(par) = .false.
    call fitfuncs(dataset_i)%set(par, val)
    if (present(active) .and. active) then
       active_pars(par) = par
    else
       active_pars(par) = 0
    end if
    set_count = set_count + 1
  end subroutine set_int_local_real
  !!
  !! FUNCTION
  !!
  !! Same as set_int_local_real except val is single precision real.
  !!
  !! SOURCE
  subroutine set_int_local_real32(dataset_i, par, val, active)
    integer, intent(in) :: dataset_i, par
    real(real32), intent(in) :: val
    logical, intent(in), optional :: active
    call set_int_local_real(dataset_i, par, real(val, kp), active)
  end subroutine set_int_local_real32
  !!
  !! FUNCTION
  !!
  !! Same as set_int_local_real except defines a global parameter.
  !!
  !! SOURCE
  subroutine set_int_global_real(par, val, active)
    integer, intent(in) :: par
    real(kp), intent(in) :: val
    logical, intent(in), optional :: active
    integer :: i
    do i = 1, size(fitfuncs)
       call set_int_local_real(i, par, val, active)
    end do
    is_global(par) = .true.
  end subroutine set_int_global_real
  !!
  !! FUNCTION
  !!
  !! Same as set_int_global_real except val is single precision real.
  !! 
  !! SOURCE
  subroutine set_int_global_real32(par, val, active)
    integer, intent(in) :: par
    real(real32), intent(in) :: val
    logical, intent(in), optional :: active
    call set_int_global_real(par, real(val, kp), active)
  end subroutine set_int_global_real32
  !!
  !! FUNCTION
  !!
  !! Same as set_int_local_real except the parameter is identified by
  !! its name instead of the index.
  !!
  !! SOURCE
  subroutine set_char_local_real(dataset_i, par, val, active)
    integer, intent(in) :: dataset_i
    character(*), intent(in) :: par
    real(kp), intent(in) :: val
    logical, intent(in), optional :: active
    if (.not. allocated(fitfuncs)) &
         & call error(__FILE__, __LINE__, &
         & 'Number of datasets is undetermined. Call gadf_init first.')
    call set_int_local_real(dataset_i, fitfuncs(1)%get_index(par), val, active)
  end subroutine set_char_local_real
  !!
  !! FUNCTION
  !!
  !! Same as set_char_local_real except val is single precision real.
  !!
  !! SOURCE
  subroutine set_char_local_real32(dataset_i, par, val, active)
    integer, intent(in) :: dataset_i
    character(*), intent(in) :: par
    real(real32), intent(in) :: val
    logical, intent(in), optional :: active
    call set_char_local_real(dataset_i, par, real(val, kp), active)
  end subroutine set_char_local_real32
  !!
  !! FUNCTION
  !!
  !! Same as set_char_local_real except defines a global parameter.
  !!
  !! SOURCE
  subroutine set_char_global_real(par, val, active)
    character(*), intent(in) :: par
    real(kp), intent(in) :: val
    logical, intent(in), optional :: active
    if (.not. allocated(fitfuncs)) &
         & call error(__FILE__, __LINE__, &
         & 'Number of datasets is undetermined. Call gadf_init first.')
    call set_int_global_real(fitfuncs(1)%get_index(par), val, active)
  end subroutine set_char_global_real
  !!
  !! FUNCTION
  !!
  !! Same as set_char_global_real except val is single precision real.
  !!
  !! SOURCE
  subroutine set_char_global_real32(par, val, active)
    character(*), intent(in) :: par
    real(real32), intent(in) :: val
    logical, intent(in), optional :: active
    call set_char_global_real(par, real(val, kp), active)
  end subroutine set_char_global_real32
  !!***

  !!****f* gadfit/gadf_set_verbosity
  !!
  !! FUNCTION
  !!
  !! Gives the user some control over how the output is
  !! displayed. These variables have no effect in gadf_print.
  !!
  !! INPUTS
  !!
  !! All optional (see gadf_init for the default values)
  !!
  !! show_scope - whether to display global and/or local
  !!              parameters.
  !! show_digits - number of significant digits in the results
  !! for x in {timing, memory, workloads, delta1, delta2, cos_phi, \
  !!           grad_chi2, uphill, acc}; do
  !!   show_$x - whether to show $x
  !! done
  !! output - where the output is directed
  !!
  !! SOURCE
  subroutine gadf_set_verbosity(scope, digits, timing, memory, workloads, &
       & delta1, delta2, cos_phi, grad_chi2, uphill, acc, output)
    integer, intent(in), optional :: scope, digits
    logical, intent(in), optional :: &
         & timing, memory, workloads, delta1, delta2, cos_phi, grad_chi2, &
         & uphill, acc
    character(*), intent(in), optional :: output
    if (present(scope)) then
       if (scope /= LOCAL .and. scope /= GLOBAL .and. &
            & scope /= GLOBAL_AND_LOCAL) &
            & call error(__FILE__, __LINE__, 'Unrecognized option. &
            &Scope can be either GLOBAL, LOCAL, or GLOBAL_AND_LOCAL.')
       show_scope = scope
    end if
    if (present(digits)) show_digits = digits
    if (present(timing)) show_timing = timing
    if (present(memory)) show_memory = memory
    if (present(workloads)) show_workloads = workloads
    if (present(delta1)) show_delta1 = delta1
    if (present(delta2)) show_delta2 = delta2
    if (present(cos_phi)) show_cos_phi = cos_phi
    if (present(grad_chi2)) show_grad_chi2 = grad_chi2
    if (present(uphill)) show_uphill = uphill
    if (present(acc)) show_acc = acc
    if (present(output)) then
       open(newunit=out_unit, file=output, action='write', &
            & iostat=err_stat, iomsg=err_msg)
       call check_err(__FILE__, __LINE__)
    end if
  end subroutine gadf_set_verbosity
  !!***

  !!****f* gadfit/gadf_set_errors
  !!
  !! FUNCTION
  !!
  !! Specifies the data point errors. The weights array, built from
  !! the inverse of the error estimates, will be initialized in
  !! gadf_fit with a call to init_weights.
  !!
  !! INPUTS
  !!
  !! e - The error type. Allowed values are given by the enumerator.
  !!
  !! SOURCE
  subroutine gadf_set_errors(e)
    integer, intent(in) :: e
    data_error_type = e
  end subroutine gadf_set_errors
  !!***

  !!****f* gadfit/read_data
  !!
  !! FUNCTION
  !!
  !! Reads data from data_units, which was constructed with call(s) to
  !! gadf_add_dataset. The position of each dataset in x_data, y_data,
  !! and weights begins at data_positions(i) and ends at
  !! data_positions(i+1)-1.
  !!
  !! SOURCE
  subroutine read_data()
    integer :: i, j
    if (any(data_positions(2:) == 1)) &
         & call error(__FILE__, __LINE__, 'Some datasets are missing. &
         &gadf_add_dataset must be called '//str(size(fitfuncs))//' times.')
    allocate(x_data(data_positions(size(data_positions))-1), &
         & y_data(data_positions(size(data_positions))-1), &
         & weights(data_positions(size(data_positions))-1), &
         & stat=err_stat, errmsg=err_msg)
    call check_err(__FILE__, __LINE__)
    if (this_image() == 1 .and. num_images() > size(x_data)) &
         & call comment(__FILE__, __LINE__, &
         & 'More images than data points. This is a waste.')
    do i = 1, size(data_units)
       j = data_positions(i)
       if (data_error_type == USER) then ! User defined errors
          do while (j < data_positions(i+1))
             read(data_units(i), *, iostat=err_stat) &
                  & x_data(j), y_data(j), weights(j)
             ! At this point, weights contains the inverse of weights
             ! (see USER in init_weights).
             if (err_stat == 0) j = j + 1
          end do
       else
          do while (j < data_positions(i+1))
             read(data_units(i), *, iostat=err_stat) x_data(j), y_data(j)
             if (err_stat == 0) j = j + 1
          end do
       end if
       call safe_close(__FILE__, __LINE__, data_units(i))
    end do
  end subroutine read_data
  !!***
  
  !!****f* gadfit/init_weights
  !!
  !! SOURCE
  subroutine init_weights()
    select case(data_error_type)
    case(NONE)
       weights = 1.0_kp
    case(SQRT_Y)
       if (minval(abs(y_data)) < epsilon(1.0_kp)) call minval_error('SQRT_Y')
       weights = 1.0_kp/sqrt(y_data)
    case(PROPTO_Y)
       if (minval(abs(y_data)) < epsilon(1.0_kp)) call minval_error('PROPTO_Y')
       weights = 1.0_kp/y_data
    case(INVERSE_Y)
       weights = y_data
    case(USER)
       weights = 1.0_kp/weights ! The uncertainties were determined in
                                ! read_data.
    case default
       call error(__FILE__, __LINE__, 'Unknown weight specifier. &
            &Allowed values are NONE, SQRT_Y, PROPTO_Y, INVERSE_Y, and USER.')
    end select
  contains
    subroutine minval_error(error_type)
      character(*), intent(in) :: error_type
      write(err_msg, '(es18.10e3)') epsilon(1.0_kp)
      call error(__FILE__, __LINE__, &
           & 'The value of each datapoint must be greater than '// &
           & trim(adjustl(err_msg))//' for '//error_type//' weighting.')
    end subroutine minval_error
  end subroutine init_weights
  !!***

  !!****f* gadfit/gadf_fit
  !!
  !! FUNCTION
  !!
  !! Performs the fitting procedure. See the user guide for a full
  !! description of the arguments.
  !!
  !! INPUTS
  !!
  !! All optional
  !!
  !! lambda - damping parameter
  !! lam_up - factor by which lambda is increased
  !! lam_down - factor by which lambda is decreased
  !! accth - relative acceleration threshold
  !! grad_chi2 - tolerance for the norm of the gradient
  !! cos_phi - toleraconce for cosine of the angle between the
  !!           residual vector and the range of the Jacobian
  !! rel_error - tolerance for the relative change in any fitting
  !!             parameter
  !! rel_error_global - same but applies only to global parameters
  !! chi2_rel - tolerance for the relative change in the value of chi2
  !! chi2_abs - tolerance for the value of chi2/degrees_of_freedom
  !! DTD_min - minimum values of DTD
  !! lam_incs - number of times lambda is allowed to increase
  !!            consecutively without terminating the procedure
  !! uphill - the exponent 'b' for allowing uphill steps
  !! max_iter - iteration limit
  !! damp_max - whether the damping matrix is updated with the largest
  !!            diagonal entries of JTJ yet encounted
  !! nielsen - whether to restrict the decreasing of lambda so that
  !!           that algorithm always stays within the trust region
  !! umnigh - whether to update lambda according to Umrigar and
  !!          Nightingale
  !! ap - whether to use adaptive parallelism
  !!
  !! SOURCE
  subroutine gadf_fit(lambda, lam_up, lam_down, accth, grad_chi2, cos_phi, &
       & rel_error, rel_error_global, chi2_rel, chi2_abs, DTD_min, lam_incs, &
       & uphill, max_iter, damp_max, nielsen, umnigh, ap)
    ! INPUT
    real(real32), intent(in), optional :: lambda, lam_up, lam_down, accth, &
         & grad_chi2, cos_phi, rel_error, rel_error_global, chi2_rel, chi2_abs
    real(kp), intent(in), optional :: DTD_min(:)
    integer, intent(in), optional :: lam_incs, uphill, max_iter
    logical, intent(in), optional :: damp_max, nielsen, umnigh
    logical, value, optional :: ap
    ! MAIN WORK VARIABLES
    real(kp) :: lambda_loc, lam_up_loc, lam_down_loc
    real(kp) :: new_chi2, old_chi2, old_old_chi2
    real(kp) :: umnigh_a = 0.5d0
    real(kp), parameter :: umnigh_m = exp(-0.2d0)
    integer :: lam_incs_loc, uphill_loc, degrees_of_freedom
    real(kp), allocatable :: Jacobian(:,:), JacobianT(:,:), JTJ(:,:)[:]
    real(kp), allocatable :: res(:) ! y-f
    real(kp), allocatable :: JTres(:)[:] ! JacobianT*res
    real(kp), allocatable :: omega(:)
    real(kp), allocatable :: JTomega(:)[:] ! JacobianT*omega
    real(kp), allocatable :: old_pars(:,:)
    real(real64), allocatable :: DTD(:,:), delta1(:), delta2(:), old_delta1(:)
    ! Tells how to position the derivatives in the Jacobian with
    ! multiple datasets. First index iterates over active parameters
    ! in a dataset, second index over the datasets. For the example
    ! shown in the user guide, with alpha parameter 1 and beta
    ! parameter 2, Jacobian_indices = [[1,3],[2,3]], i.e., the
    ! derivative with respect to beta is always put into the third
    ! column in the Jacobian.
    integer, allocatable :: Jacobian_indices(:,:)
    ! PARALLELISM
    real(real64) :: img_weights(num_images()), old_img_weights(num_images())
    real(real64) :: old_Jac_cpu(num_images()), old_chi2_cpu(num_images())
    real(real64) :: old_linalg_cpu(num_images()), old_omega_cpu(num_images())
    integer :: sizes(num_images())
    ! img_bounds defines the work regions for each image in x_data,
    ! y_data, and weights. In order to go through all data points
    ! attributed to a given image, a double loop is necessary of the
    ! form
    !   do i = 1, size(fitfuncs) ! number of datasets
    !     do j = img_bounds(i), img_bounds(i+1)-1
    !        ...
    ! As an example, given 3 images and 2 datasets, one with 50 and
    ! one with 30 data points, we would have
    !   img_bounds = [ 1, 28, 28] ! Image 1
    !   img_bounds = [28, 51, 55] ! Image 2
    !   img_bounds = [55, 55, 81] ! Image 3
    integer, allocatable :: img_bounds(:)
    ! OTHER
    real(real64), allocatable :: linalg_tmp(:,:)
    real(kp) :: acc_ratio, beta
    real(kp), save :: tmp[*]
    type(advar) :: dummy
    logical :: flag
    integer :: active_pars_tmp(size(active_pars)), shift
    integer :: i, j
    if (.not. allocated(fitfuncs)) &
         & call error(__FILE__, __LINE__, &
         & 'Number of datasets is undetermined. Call gadf_init first.')
    call read_data()
    call init_weights()
    ! Initialize some of the optional arguments
    lambda_loc = 1.0_kp
    if (present(lambda)) lambda_loc = lambda
    lam_up_loc = 10.0_kp
    if (present(lam_up)) lam_up_loc = lam_up
    lam_down_loc = 10.0_kp
    if (present(lam_down)) lam_down_loc = lam_down
    lam_incs_loc = 2
    if (present(lam_incs)) then
       if (lam_incs < 1) &
            & call error(__FILE__, __LINE__, &
            & 'Input parameter lam_incs must be at least 1.')
       lam_incs_loc = lam_incs
    end if
    uphill_loc = 0
    if (present(uphill)) uphill_loc = uphill
    ! Reshape active_pars without the zeros
    if (minval(active_pars) == 0) then ! Only if has passive pars
       do i = 1, size(active_pars)-1 ! Move zeros to end
          do j = 1, size(active_pars) - 1
             if (active_pars(j) == 0) then
                active_pars(j) = active_pars(j+1)
                active_pars(j+1) = 0
             end if
          end do
       end do
       active_pars_tmp = active_pars
       call safe_deallocate(__FILE__, __LINE__, active_pars)
       allocate(active_pars(minloc(active_pars_tmp,1)-1))
       active_pars = active_pars_tmp(:size(active_pars))
    end if
#ifdef USE_AD
    ! Make all fitting parameters active if using AD
    if (size(active_pars) == 0) &
         & call error(__FILE__, __LINE__, 'There are no active parameters.')
    if (this_image() == 1 .and. &
         & set_count < size(fitfuncs)*size(fitfuncs(1)%pars)) &
         & call warning(__FILE__, __LINE__, &
         & 'Some parameters might be uninitialized.')
    do j = 1, size(fitfuncs)
#ifdef POLYM_ARRAY_SUPPORT
       fitfuncs(j)%pars(active_pars)%index = [(i, i=1, size(active_pars))]
#else
       fitfunc_tmp => fitfuncs(j)
       fitfunc_tmp%pars(active_pars)%index = [(i, i=1, size(active_pars))]
#endif
    end do
    index_count = size(active_pars) ! Index of the next AD variable
                                    ! will be index_count+1
#endif
    ! Determine the parameter indices in the Jacobian
    allocate(Jacobian_indices(size(active_pars), size(fitfuncs)), &
         & stat=err_stat, errmsg=err_msg)
    call check_err(__FILE__, __LINE__)
    shift = 0
    do i = 1, size(fitfuncs)
       do j = 1, size(active_pars)
          if (is_global(active_pars(j))) then
             Jacobian_indices(j,i) = j
             if (i > 1) shift = shift + 1
          else
             Jacobian_indices(j,i) = j + (i-1)*size(active_pars) - shift
          end if
       end do
    end do
    ! Initialize the main work variables
    ! dim = (global parameters) + (local parameters) x (number of datasets)
    associate(dim => size(fitfuncs)*size(active_pars)-shift)
      allocate(JTres(dim)[*], JTJ(dim, dim)[*], DTD(dim, dim), delta1(dim), &
           & old_delta1(dim), linalg_tmp(dim, dim), JTomega(dim)[*], &
           & delta2(dim), old_pars(size(active_pars), size(fitfuncs)), &
           & img_bounds(size(data_positions)), stat=err_stat, errmsg=err_msg)
      call check_err(__FILE__, __LINE__)
    end associate
    call re_initialize()
    delta2 = 0d0
    DTD = 0d0
    if (present(DTD_min)) forall (i=1:size(DTD,1)) DTD(i,i) = DTD_min(i)
    ! Degrees of freedom
    degrees_of_freedom = size(x_data) - size(Jacobian,2)
    if (degrees_of_freedom < 0) then
       call error(__FILE__, __LINE__, &
            & 'More independent fitting parameters than data points.')
    else if (degrees_of_freedom == 0) then
       if (this_image() == 1) &
            & call comment(__FILE__, __LINE__, &
            & 'There are no degrees of freedom - chi2/DOF has no meaning.')
       degrees_of_freedom = 1
    end if
    ! Old parameters are initialized with the current ones.
    do i = 1, size(fitfuncs)
#ifdef POLYM_ARRAY_SUPPORT
       old_pars(:,i) = fitfuncs(i)%pars(active_pars)%val
#else
       fitfunc_tmp => fitfuncs(i)
       old_pars(:,i) = fitfunc_tmp%pars(active_pars)%val
#endif
    end do
    ! Initialize timers
    call Jacobian_timer%reset()
    call chi2_timer%reset()
    call linalg_timer%reset()
    call omega_timer%reset()
    call main_loop_timer%reset()
    ! The main algorithm
    call main_loop_timer%time()
    old_chi2 = chi2()
    main: do
       if (present(ap) .and. ap .and. num_images() > 1) call re_initialize()
       ! STEP 1 - Jacobian and the residuals
       call Jacobian_timer%time()
       do j = 1, size(fitfuncs)
          ! forward_values must be re-initialized each time the
          ! fitting function changes.
#ifdef POLYM_ARRAY_SUPPORT
          forward_values(:size(active_pars)) = fitfuncs(j)%pars(active_pars)%val
#else
          fitfunc_tmp => fitfuncs(j)
          forward_values(:size(active_pars)) = fitfunc_tmp%pars(active_pars)%val
#endif
          do i = img_bounds(j), img_bounds(j+1)-1
             res(i-img_bounds(1)+1) = fitfuncs(j)%eval(x_data(i))
             res(i-img_bounds(1)+1) = &
                  & (y_data(i) - res(i-img_bounds(1)+1))*weights(i)
#ifdef USE_AD
             call ad_grad(size(active_pars))
#else
             call fitfuncs(j)%grad_finite(x_data(i), active_pars, adjoints)
#endif
             JacobianT(Jacobian_indices(:,j), i-img_bounds(1)+1) = &
                  & adjoints(:size(active_pars))*weights(i)
          end do
       end do
       call Jacobian_timer%time()
       ! STEP 2 - Intermediate linalg operations and delta1
       call linalg_timer%time()
       Jacobian = transpose(JacobianT)
       JTJ = matmul(JacobianT,Jacobian)
       JTres = matmul(JacobianT, res)
       call linalg_timer%time()
       call co_sum(JTJ)
       call co_sum(JTres)
       if (present(damp_max) .and. .not. damp_max) then
          forall (i=1:size(DTD,1)) DTD(i,i) = real(JTJ(i,i), real64)
       else
          forall (i=1:size(DTD,1)) &
               & DTD(i,i) = max(DTD(i,i), real(JTJ(i,i),real64))
       end if
       delta1 = real(JTres, real64)
       linalg_tmp = real(JTJ + lambda_loc*DTD, real64)
       call potr_f08(__FILE__, __LINE__, linalg_tmp, delta1)
       ! STEP 3 - omega and delta2
       calc_acc: if (present(accth) .and. accth > tiny(1.0)) then
          reverse_mode = .false.
          call omega_timer%time()
          do j = 1, size(fitfuncs)
#ifdef POLYM_ARRAY_SUPPORT
             fitfuncs(j)%pars(active_pars)%d = delta1(Jacobian_indices(:,j))
#else
             fitfunc_tmp => fitfuncs(j)
             fitfunc_tmp%pars(active_pars)%d = delta1(Jacobian_indices(:,j))
#endif
             do i = img_bounds(j), img_bounds(j+1)-1
#ifdef USE_AD
                dummy = fitfuncs(j)%eval(x_data(i))
                omega(i-img_bounds(1)+1) = -dummy%dd*weights(i)
#else
                omega(i-img_bounds(1)+1) = -fitfuncs(j)%dir_deriv_2nd_finite( &
                     & x_data(i), active_pars, &
                     & real(delta1(Jacobian_indices(:,j)), kp))*weights(i)
#endif
             end do
          end do
          call omega_timer%time()
          reverse_mode = .true.
          JTomega = matmul(JacobianT, omega)
          call co_sum(JTomega)
          delta2 = real(JTomega,real64)
          linalg_tmp = real(JTJ + lambda_loc*DTD, real64)
          call potr_f08(__FILE__, __LINE__, linalg_tmp, delta2)
          ! Use acc if sqrt(delta2Ddelta2/delta1Ddelta1) < alpha
          acc_ratio = sqrt(dot_product(delta2, matmul(DTD,delta2))/ &
               & dot_product(delta1, matmul(DTD,delta1)))
          if (present(accth) .and. acc_ratio > accth) delta2 = 0d0
       end if calc_acc
       ! Update the fitting parameters
       do i = 1, size(fitfuncs)
#ifdef POLYM_ARRAY_SUPPORT
          fitfuncs(i)%pars(active_pars)%val = &
               & fitfuncs(i)%pars(active_pars)%val + &
               & delta1(Jacobian_indices(:,i)) + &
               & 0.5*delta2(Jacobian_indices(:,i))
#else
          fitfunc_tmp => fitfuncs(i)
          fitfunc_tmp%pars(active_pars)%val = &
               & fitfunc_tmp%pars(active_pars)%val + &
               & delta1(Jacobian_indices(:,i)) + &
               & 0.5*delta2(Jacobian_indices(:,i))
#endif
       end do
       ! STEP 4 - Update lambda
       new_lambda: do i = 1, lam_incs_loc + 1
          new_chi2 = chi2()
          if (iterations == 0) then
             beta = 0.0_kp
          else
             beta = dot_product(delta1, matmul(DTD, old_delta1))/ &
                  sqrt(dot_product(delta1, matmul(DTD, delta1)))/ &
                  sqrt(dot_product(old_delta1, matmul(DTD, old_delta1)))
          end if
          chi2_chk: if ((1.0_kp-beta)**uphill_loc*new_chi2 < old_chi2) then
             if (present(nielsen) .and. nielsen) then
                associate(rho => (old_chi2-new_chi2)/2/ &
                     & dot_product(delta1, matmul(JTJ+lambda_loc*DTD,delta1)))
                  lambda_loc = lambda_loc*max(1/lam_down_loc, 1-(2*rho-1)**3)
                end associate
             end if
             if (present(umnigh) .and. umnigh) then
                if (new_chi2 < old_chi2 .and. beta >= 0.0_kp) then
                   umnigh_a = umnigh_a*umnigh_m + 1.0_kp-umnigh_m
                   lambda_loc = lambda_loc*min(1.0_kp, max(1e-2_kp, &
                        & (1.0_kp-abs(2.0_kp*umnigh_a-1.0_kp))**2))
                else
                   umnigh_a = umnigh_a*umnigh_m + (1.0_kp-umnigh_m)/2.0_kp
                   if (new_chi2 >= old_chi2) &
                        & lambda_loc = lambda_loc/min(10.0_kp, max(1.0_kp, &
                        & (1.0_kp-abs(2.0_kp*umnigh_a-1.0_kp))))
                end if
             end if
             if (.not. ((present(nielsen) .and. nielsen) .or. &
                  & (present(umnigh) .and. umnigh))) &
                  lambda_loc = lambda_loc/lam_down_loc
             exit new_lambda
          else
             proceed_or_quit: if (i <= lam_incs_loc) then
                if (present(umnigh) .and. umnigh) then
                   umnigh_a = umnigh_a*umnigh_m
                   if (beta < 0.0_kp) then
                      lambda_loc = lambda_loc*min(1.0_kp, max(1e-2_kp, &
                           & (1.0_kp-abs(2.0_kp*umnigh_a-1.0_kp))**2))
                   else
                      lambda_loc = lambda_loc*min(1.0_kp, max(0.1_kp, &
                           & (1.0_kp-abs(2.0_kp*umnigh_a-1.0_kp))))
                   end if
                else
                   lambda_loc = lam_up_loc*lambda_loc
                end if
                do j = 1, size(fitfuncs)
#ifdef POLYM_ARRAY_SUPPORT
                   fitfuncs(j)%pars(active_pars)%val = old_pars(:,j)
#else
                   fitfunc_tmp => fitfuncs(j)
                   fitfunc_tmp%pars(active_pars)%val = old_pars(:,j)
#endif
                end do
                delta1 = real(JTres, real64)
                linalg_tmp = real(JTJ + lambda_loc*DTD, real64)
                call potr_f08(__FILE__, __LINE__, linalg_tmp, delta1)
                do j = 1, size(fitfuncs)
#ifdef POLYM_ARRAY_SUPPORT
                   fitfuncs(j)%pars(active_pars)%val = &
                        & fitfuncs(j)%pars(active_pars)%val + &
                        & delta1(Jacobian_indices(:,j))
#else
                   fitfunc_tmp => fitfuncs(j)
                   fitfunc_tmp%pars(active_pars)%val = &
                        & fitfunc_tmp%pars(active_pars)%val + &
                        & delta1(Jacobian_indices(:,j))
#endif
                end do
             else
                do j = 1, size(fitfuncs)
#ifdef POLYM_ARRAY_SUPPORT
                   fitfuncs(j)%pars(active_pars)%val = old_pars(:,j)
#else
                   fitfunc_tmp => fitfuncs(j)
                   fitfunc_tmp%pars(active_pars)%val = old_pars(:,j)
#endif
                end do
                if (this_image() == 1) write(out_unit, '(1x, *(g0))') &
                     & 'Lambda increased '//str(lam_incs_loc+1)// &
                     & ' times in a row.'
                exit main
             end if proceed_or_quit
          end if chi2_chk
       end do new_lambda
       ! Update some variables for the next iteration
       do i = 1, size(fitfuncs)
#ifdef POLYM_ARRAY_SUPPORT
          old_pars(:,i) = fitfuncs(i)%pars(active_pars)%val
#else
          fitfunc_tmp => fitfuncs(i)
          old_pars(:,i) = fitfunc_tmp%pars(active_pars)%val
#endif
       end do
       old_delta1 = delta1
       old_old_chi2 = old_chi2
       old_chi2 = min(old_chi2, new_chi2) ! In case of uphill steps
       iterations = iterations + 1
       call print_info(out_unit, new_chi2/degrees_of_freedom, &
            & lambda_loc, iterations, delta1, delta2, Jacobian_indices)
       if (this_image() == 1 .and. show_acc) &
            & write(out_unit, '(1x, *(g0))') 'acc ratio     = ', acc_ratio
       if (this_image() == 1 .and. show_uphill) &
            & write(out_unit, '(1x, *(g0))') 'beta (uphill) = ', beta
       ! STEP 5 - Check all convergence criteria
       if (present(chi2_abs) .and. old_chi2/degrees_of_freedom < chi2_abs) then
          if (this_image() == 1) &
               & write(out_unit, '(/, 1x, *(g0))') 'Chi2/DOF is less than ', &
               & chi2_abs
          exit main
       end if
       if (present(chi2_rel) .and. &
            & (old_old_chi2 - old_chi2)/old_chi2 < chi2_rel) then
          if (this_image() == 1) &
               & write(out_unit, '(/, 1x, *(g0))') &
               & 'Relative change in Chi2 was less than ', chi2_rel
          exit main
       end if
       if (present(grad_chi2) .or. show_grad_chi2) then
          JTres = matmul(JacobianT, res)
          call co_sum(JTres)
          dummy%val = 2*norm2(JTres)
          if (this_image() == 1 .and. show_grad_chi2) &
               & write(out_unit, '(1x, *(g0))') '|nabla(chi2)| = ', dummy%val
          if (present(grad_chi2) .and. dummy%val < grad_chi2) then
             if (this_image() == 1) &
                  & write(out_unit, '(/, 1x, *(g0))') &
                  & 'The norm of the gradient is less than ', grad_chi2
             exit main
          end if
       end if
       if (present(cos_phi) .or. show_cos_phi) then
          block
            real(kp), allocatable :: Jdelta(:)
            ! With AP the size changes with each iteration
            allocate(Jdelta(size(res)), stat=err_stat, errmsg=err_msg)
            call check_err(__FILE__, __LINE__)
            Jdelta = matmul(Jacobian, delta1)
            tmp = dot_product(res, Jdelta)
            call co_sum(tmp)
            dummy%val = abs(tmp)
            tmp = dot_product(res,res)
            call co_sum(tmp)
            dummy%val = dummy%val/sqrt(tmp)
            tmp = dot_product(Jdelta, Jdelta)
            call co_sum(tmp)
            dummy%val = dummy%val/sqrt(tmp)
          end block
          if (this_image() == 1 .and. show_cos_phi) &
               & write(out_unit, '(1x, *(g0))') '|cos(phi)|    = ', dummy%val
          if (present(cos_phi) .and. dummy%val < cos_phi ) then
             if (this_image() == 1) &
                  & write(out_unit, '(/, 1x, *(g0))') 'Cosine of the angle &
                  &between y-f and Jdelta is less than ', cos_phi
             exit main
          end if
       end if
       if (present(rel_error)) then
          flag = .false.
          do j = 1, size(fitfuncs)
#ifdef POLYM_ARRAY_SUPPORT
             if (any(abs(delta1(Jacobian_indices(:,j))/ &
                  & fitfuncs(j)%pars(active_pars)%val) > rel_error)) &
                  & flag = .true.
#else
             fitfunc_tmp => fitfuncs(j)
             if (any(abs(delta1(Jacobian_indices(:,j))/ &
                  & fitfunc_tmp%pars(active_pars)%val) > rel_error)) &
                  & flag = .true.
#endif   
          end do
          if (.not. flag) then
             if (this_image() == 1) &
                  & write(out_unit, '(/, 1x, *(g0))') &
                  & 'Relative change in all parameters was less than ', &
                  & rel_error
             exit main
          end if
       end if
       if (present(rel_error_global)) then
#ifdef POLYM_ARRAY_SUPPORT
          if (.not. any(is_global(active_pars) .and. &
               & abs(delta1(Jacobian_indices(:,1))/ &
               & fitfuncs(1)%pars(active_pars)%val) > &
               & rel_error_global)) then
#else
          fitfunc_tmp => fitfuncs(1)
          if (.not. any(is_global(active_pars) .and. &
               & abs(delta1(Jacobian_indices(:,1))/ &
               & fitfunc_tmp%pars(active_pars)%val) > &
               & rel_error_global)) then
#endif
             if (this_image() == 1) &
                  & write(out_unit, '(/, 1x, *(g0))') &
                  & 'Relative change in all global parameters was less than ', &
                  & rel_error_global
             exit main
          end if
       end if
       if (present(max_iter) .and. iterations >= max_iter) then
          if (this_image() == 1) write(out_unit, '(/, 1x, g0)') &
               & 'Iteration limit was reached'
          exit main
       end if
       if (this_image() == 1) write(out_unit, '(g0)')
    end do main
    call main_loop_timer%time()
    sync all ! Wait for all timers
    if (show_memory)    call print_memory_usage(out_unit)
    if (show_workloads) call print_workloads(out_unit)
    if (show_timing)    call print_timing(out_unit)
  contains
    !!***
    
    !!****f* gadfit/gadf_fit/re_initialize
    !!
    !! FUNCTION
    !!
    !! Initializes all work variables that depend on the relative
    !! workloads. There are two main steps: determination of the image
    !! weights and the construction of img_bounds. The weights are
    !! determined by the amount of cpu time each image spent in the
    !! parallel parts of the main loop during the previous
    !! iteration. The weights determine the length of the work region
    !! (my_size) that is attributed to each image in img_bounds (see
    !! above for img_bounds). If adaptive parallelism is not used, the
    !! determination of the weights is straightforward and this
    !! procedure is called only once.
    !!
    !! SOURCE
    subroutine re_initialize()
      integer :: my_size, prev_size, cur_length, tmp
      call safe_deallocate(__FILE__, __LINE__, Jacobian)
      call safe_deallocate(__FILE__, __LINE__, JacobianT)
      call safe_deallocate(__FILE__, __LINE__, res)
      call safe_deallocate(__FILE__, __LINE__, omega)
      ! STEP 1 - determine image weights
      if (iterations == 0) then
         img_weights = 1d0
         old_img_weights = 1d0/num_images()
         old_Jac_cpu = 0d0
         old_chi2_cpu = 0d0
         old_linalg_cpu = 0d0
         old_omega_cpu = 0d0
      else
         do i = 1, num_images()
            img_weights(i) = Jacobian_timer[i]%cpu_time - old_Jac_cpu(i) + &
                 & chi2_timer[i]%cpu_time - old_chi2_cpu(i) + &
                 & linalg_timer[i]%cpu_time - old_linalg_cpu(i) + &
                 & omega_timer[i]%cpu_time - old_omega_cpu(i)
            old_Jac_cpu(i) = Jacobian_timer[i]%cpu_time
            old_chi2_cpu(i) = chi2_timer[i]%cpu_time
            old_linalg_cpu(i) = linalg_timer[i]%cpu_time
            old_omega_cpu(i) = omega_timer[i]%cpu_time
         end do
         if (this_image() == 1) write(out_unit, '(g0, *(1x, f5.4))') &
              & 'Current workloads:', img_weights/sum(img_weights)
         if (this_image() == 1) write(out_unit, '(g0)')
         if (minval(img_weights) <= epsilon(1d0)) then
            if (this_image() == 1 ) &
                 & call comment(__FILE__, __LINE__, &
                 & 'The calculation is too fast for adaptive &
                 &parallelism (AP) to be effective. AP is now switched off.')
            ap = .false.
            img_weights = 1d0
         else
            img_weights = 1d0/img_weights
         end if
      end if
      img_weights = img_weights/sum(img_weights)
      img_weights = old_img_weights - (1d0/num_images() - img_weights)
      old_img_weights = img_weights
      ! STEP 2 - build img_bounds
      sizes = int(img_weights*size(x_data))
      tmp = sum(sizes)
      do i = 1, size(sizes)
         if (i <= size(x_data)-tmp) sizes(i) = sizes(i) + 1
      end do
      prev_size = sum(sizes(1:this_image()-1))
      img_bounds = prev_size+1
      my_size = sizes(this_image())
      do i = 2, size(data_positions)
         cur_length = data_positions(i)-data_positions(i-1)
         if (prev_size >= cur_length) then
            prev_size = prev_size - cur_length
         else if (my_size > 0) then
            if (my_size+prev_size >= cur_length) then
               img_bounds(i) = img_bounds(i-1) + cur_length -  prev_size
               my_size = my_size + prev_size - cur_length
               prev_size = 0
            else
               img_bounds(i) = img_bounds(i-1) + my_size
               my_size = my_size - cur_length
            end if
         else
            img_bounds(i) = img_bounds(i-1)
         end if
      end do
      my_size = sizes(this_image())
      ! STEP 3 - Reallocate the work variables
      ! dim = (global parameters) + (local parameters) x (number of datasets)
      associate(dim => size(fitfuncs)*size(active_pars)-shift)
        allocate(Jacobian(my_size, dim), JacobianT(dim, my_size), &
             & res(my_size), omega(my_size), stat=err_stat, errmsg=err_msg)
        call check_err(__FILE__, __LINE__)
        JacobianT = 0.0_kp
      end associate
    end subroutine re_initialize
    !!***
    
    !!****f* gadfit/gadf_fit/chi2
    !!
    !! FUNCTION
    !!
    !! The sum of squares over all datasets.
    !!
    !! SOURCE
    real(kp) function chi2() result(y)
      real(kp), save :: sum[*]
      integer :: saved_indices(size(fitfuncs(1)%pars))
      integer :: i, j
      saved_indices = 0
      call chi2_timer%time()
      do j = 1, size(fitfuncs)
#ifdef POLYM_ARRAY_SUPPORT
         call swap(fitfuncs(j)%pars%index, saved_indices)
#else
         fitfunc_tmp => fitfuncs(j)
         call swap(fitfunc_tmp%pars%index, saved_indices)
#endif
         do i = img_bounds(j), img_bounds(j+1)-1
            res(i-img_bounds(1)+1) = fitfuncs(j)%eval(x_data(i))
            res(i-img_bounds(1)+1) = &
                 & (y_data(i) - res(i-img_bounds(1)+1))*weights(i)
         end do
#ifdef POLYM_ARRAY_SUPPORT
         call swap(fitfuncs(j)%pars%index, saved_indices)
#else
         call swap(fitfunc_tmp%pars%index, saved_indices)
#endif
      end do
      sum = dot_product(res,res)
      call chi2_timer%time()
      call co_sum(sum)
      y = sum
    end function chi2
    !!***
  end subroutine gadf_fit

  !!****f* gadfit/print_memory_usage
  !!
  !! SOURCE
  subroutine print_memory_usage(io_unit)
    integer, intent(in) :: io_unit
    if (this_image() /= 1) return
    write(io_unit, '(g0)')
    call numerical_integration_memory_report(io_unit)
    call ad_memory_report(io_unit)
  end subroutine print_memory_usage
  !!***

  !!****f* gadfit/print_workloads
  !!
  !! SOURCE
  subroutine print_workloads(io_unit)
    integer, intent(in) :: io_unit
    real(real64) :: img_weights(num_images())
    integer :: i
    if (this_image() /= 1) return
    do i = 1, num_images()
       img_weights(i) = Jacobian_timer[i]%cpu_time + chi2_timer[i]%cpu_time + &
            & linalg_timer[i]%cpu_time
    end do
    if (sum(img_weights) > epsilon(1d0)) then
       img_weights = img_weights/sum(img_weights)
       write(io_unit, '(/, 1x, g0)') 'Workloads'
       write(io_unit, '(1x, g0)') '========='
       do i = 1, num_images()
          write(io_unit, '(2x, "Image ", i4.4, 1x, f6.4)') i, img_weights(i)
       end do
    end if
  end subroutine print_workloads
  !!***

  !!****f* gadfit/print_timing
  !!
  !! SOURCE
  subroutine print_timing(io_unit)
    use, intrinsic :: iso_fortran_env, only: int64
    integer, intent(in) :: io_unit
    real(real64) :: Jac_ave_cpu, chi2_ave_cpu, linalg_ave_cpu, omega_ave_cpu
    real(real64) :: total_cpu_time
    integer(int64) :: count_rate
    integer :: i
    if (this_image() /= 1) return
    if (main_loop_timer%cpu_time < epsilon(1d0)) then
       write(io_unit, '(/, 1x, g0)') &
            & 'Timing is suppressed because the calculation was too fast.'
       return
    end if
    Jac_ave_cpu = 0d0; chi2_ave_cpu = 0d0; linalg_ave_cpu = 0d0
    omega_ave_cpu = 0d0; total_cpu_time = 0d0
    do i = 1, num_images()
       Jac_ave_cpu = Jac_ave_cpu + Jacobian_timer[i]%cpu_time
       chi2_ave_cpu = chi2_ave_cpu + chi2_timer[i]%cpu_time
       linalg_ave_cpu = linalg_ave_cpu + linalg_timer[i]%cpu_time
       omega_ave_cpu = omega_ave_cpu + omega_timer[i]%cpu_time
    end do
    Jac_ave_cpu = Jac_ave_cpu/num_images()
    chi2_ave_cpu = chi2_ave_cpu/num_images()
    linalg_ave_cpu = linalg_ave_cpu/num_images()
    omega_ave_cpu = omega_ave_cpu/num_images()
    call system_clock(count_rate=count_rate)
    write(io_unit, '(/, 1x, g0)') 'Average cpu time per call'
    write(io_unit, '(1x, g0)') '========================='
    write(io_unit, '(2x, g0, f8.2)') &
         & 'Jacobian: ', Jac_ave_cpu/Jacobian_timer%num_calls
    write(io_unit, '(6x, g0, f8.2)') 'Chi2: ', chi2_ave_cpu/chi2_timer%num_calls
    write(io_unit, '(4x, g0, f8.2)') &
         & 'linalg: ', linalg_ave_cpu/linalg_timer%num_calls
    if (omega_timer%num_calls == 0) omega_timer%num_calls = 1
    write(io_unit, '(5x, g0, f8.2)') &
         & 'omega: ', omega_ave_cpu/omega_timer%num_calls
    write(io_unit, '(/, 1x, g0)') 'Relative cost'
    write(io_unit, '(1x, g0)') '============='
    write(io_unit, '(2x, g0, f5.1, g0)') &
         & 'Jacobian: ', 1d2*Jac_ave_cpu/main_loop_timer%cpu_time, '%'
    write(io_unit, '(6x, g0, f5.1, g0)') &
         & 'Chi2: ', 1d2*chi2_ave_cpu/main_loop_timer%cpu_time, '%'
    write(io_unit, '(4x, g0, f5.1, g0)') &
         & 'linalg: ', 1d2*linalg_ave_cpu/main_loop_timer%cpu_time, '%'
    write(io_unit, '(5x, g0, f5.1, g0)') &
         & 'omega: ', 1d2*omega_ave_cpu/main_loop_timer%cpu_time, '%'
    write(io_unit, '(5x, g0, f5.1, g0)') &
         & 'Total: ', 1d2*(Jac_ave_cpu + chi2_ave_cpu + linalg_ave_cpu + &
         & omega_ave_cpu)/main_loop_timer%cpu_time, '%'
    write(io_unit, '(/, 1x, g0)') 'Timing            cpu     wall'
    write(io_unit, '(1x, g0)') '=============================='
    write(io_unit, '(2x, g0)') 'Jacobian'
    do i = 1, num_images()
       write(io_unit, '(3x, g0, i4.4, 2f9.1)') &
            & 'Image ', i, Jacobian_timer[i]%cpu_time, &
            & 1d0*Jacobian_timer[i]%wall_time/count_rate
    end do
    write(io_unit, '(2x, g0)') 'Chi2'
    do i = 1, num_images()
       write(io_unit, '(3x, g0, i4.4, 2f9.1)') 'Image ', i, &
            & chi2_timer[i]%cpu_time, 1d0*chi2_timer[i]%wall_time/count_rate
    end do
    write(io_unit, '(2x, g0)') 'Omega'
    do i = 1, num_images()
       write(io_unit, '(3x, g0, i4.4, 2f9.1)') 'Image ', i, &
            & omega_timer[i]%cpu_time, 1d0*omega_timer[i]%wall_time/count_rate
    end do
    write(io_unit, '(1x, g0)')'=============================='
    do i = 1, num_images()
       total_cpu_time = total_cpu_time + main_loop_timer[i]%cpu_time
    end do
    write(io_unit, '(2x, g0, f13.1, f9.1)') &
         & 'Total  ', total_cpu_time, 1d0*main_loop_timer%wall_time/count_rate
  end subroutine print_timing
  !!***
  
  !!****f* gadfit/print_info
  !!
  !! FUNCTION
  !!
  !! Prints the results of the last iteration. Arguments have the same
  !! meaning as in gadf_fit.
  !! 
  !! SOURCE
  subroutine print_info(io_unit, chi2_DOF, lambda, iterations, delta1, delta2, &
       & Jacobian_indices)
    integer, intent(in) :: io_unit
    real(kp), intent(in), optional :: chi2_DOF, lambda
    integer, intent(in), optional :: iterations
    real(real64), intent(in), optional :: delta1(:), delta2(:)
    integer, intent(in), optional :: Jacobian_indices(:,:)
    ! I/O work variables for prettier output
    character(:), allocatable, save :: fmt_name, fmt_value, par_digits
    type(string) :: tmp_names(size(active_pars))
    integer :: i, j
    if (this_image() /= 1) return
    if (.not. present(iterations) .or. (present(iterations) .and. &
         & iterations == 1)) then ! This does not have to run more than once
       tmp_names = fitfuncs(1)%get_name(active_pars)
       fmt_name = 'a'//str(maxval(len(tmp_names),1))
       fmt_value = 'es'//str(show_digits+8)//'.'//str(show_digits)//'e3'
       associate(x => 1+int(log(1.0*size(fitfuncs(1)%pars))/log(10.0)))
         par_digits = 'i'//str(x)//'.'//str(x)
       end associate
    end if
    if (present(iterations)) &
         & write(io_unit, '(g0, g0)') 'Iteration: ', iterations
    if (present(lambda)) write(io_unit, '(g0, es8.1e2)') 'Lambda:', lambda
    if (present(chi2_DOF)) &
         & write(io_unit, '(g0, '//fmt_value//')') 'Chi2/DOF:', chi2_DOF
    ! Global parameters
    if (show_scope == GLOBAL .or. show_scope == GLOBAL_AND_LOCAL .or. &
         & size(fitfuncs) == 1) then
       if (size(fitfuncs) /= 1) write(io_unit, '(1x, g0)') 'Global parameters:'
       tmp_names = fitfuncs(1)%get_name(active_pars)
       do i = 1, size(active_pars)
          if (is_global(active_pars(i))) then
#ifdef POLYM_ARRAY_SUPPORT
             associate(val => fitfuncs(1)%pars(active_pars(i))%val)
#else
             fitfunc_tmp => fitfuncs(1)
             associate(val => fitfunc_tmp%pars(active_pars(i))%val)
#endif
               if (len_aux(tmp_names(i)%name) > 0) then ! If is named
                  write(io_unit, '(2x'//fmt_name//'" ="'//fmt_value//')', &
                       & advance='no') tmp_names(i)%name, val
               else
                  write(io_unit, '(2x, "Par ", '//par_digits//', " ="'// &
                       & fmt_value//')', advance='no') active_pars(i), val
               end if
               if (present(delta1) .and. show_delta1) &
                    & write(io_unit, '(1x, "(", '//fmt_value//', ")")', &
                    & advance='no') delta1(active_pars(i))
               if (present(delta2) .and. show_delta2) &
                    & write(io_unit, '(1x, "(", '//fmt_value//', ")")', &
                    & advance='no') delta2(active_pars(i))
               write(io_unit, '(g0)')
             end associate
          end if
       end do
    end if
    ! Local parameters
    if ((show_scope == LOCAL .or. show_scope == GLOBAL_AND_LOCAL)) then
       if (size(fitfuncs) /= 1) write(io_unit, '(1x, g0)') 'Local parameters:'
       tmp_names = fitfuncs(1)%get_name(active_pars)
       do i = 1, size(active_pars)
          do j = 1, size(fitfuncs)
             if (.not. is_global(active_pars(i))) then
                associate(x => 1+int(log(1.0*size(fitfuncs))/log(10.0)))
                  write(io_unit, '(2x, g0, i'//str(x)//'.'//str(x)//', g0)', &
                       & advance='no') 'Curve ', j, ':'
                end associate
#ifdef POLYM_ARRAY_SUPPORT
                associate(val => fitfuncs(j)%pars(active_pars(i))%val)
#else
                fitfunc_tmp => fitfuncs(j)
                associate(val => fitfunc_tmp%pars(active_pars(i))%val)
#endif
                  if (len_aux(tmp_names(i)%name) > 0) then ! If is named
                     write(io_unit, '(x'//fmt_name//'" ="'//fmt_value//')', &
                          & advance='no') tmp_names(i)%name, val
                  else
                     write(io_unit, '(1x, "Par "'//par_digits//'" ="'// &
                          & fmt_value//')', advance='no') active_pars(i), val
                  end if
                  if (present(delta1) .and. show_delta1) &
                       & write(io_unit, '(1x, "(", '//fmt_value//', ")")', &
                       & advance='no') delta1(Jacobian_indices(i,j))
                  if (present(delta2) .and. show_delta2) &
                       & write(io_unit, '(1x, "(", '//fmt_value//', ")")', &
                       & advance='no') delta2(Jacobian_indices(i,j))
                  write(io_unit, '(g0)')
                end associate
             end if
          end do
       end do
    end if
  contains
    integer function len_aux(x) result(y)
      character(*), intent(in) :: x
#ifdef DEF_LEN_CHAR_COMP_SUPPORT
      y = len(x)
#else
      y = len(trim(x))
#endif
    end function len_aux
  end subroutine print_info
  !!***

  !!****f* gadfit/gadf_print
  !!
  !! FUNCTION
  !!
  !! If a fitting procedure has been performed, 3 files are created:
  !! one containing the function values that are calculated using the
  !! optimized fitting parameters, one containing the optimized
  !! parameters, and a log file containing the summary of the
  !! resources used. If a fitting procedure has not been performed,
  !! only the function values are printed with the user defined
  !! initial parameters.
  !!
  !! INPUTS
  !!
  !! All optional
  !!
  !! begin - the lower x-bound
  !! end - the upper x-bound
  !! points - number of points printed per curve
  !! output - I/O device
  !! grouped - whether to put all (x,y)-values into a single file
  !! logplot - whether to produce results suitable for a log(x),y-plot
  !! begin_kp - same as begin but with double/quad precision;
  !!            overrides 'begin'
  !! end_kp - same as end but with double/quad precision; overrides
  !!          'end'
  !!
  !! SOURCE
  subroutine gadf_print(begin, end, points, output, grouped, logplot, &
       & begin_kp, end_kp)
    real(real32), intent(in), optional :: begin, end
    integer, intent(in), optional :: points
    character(*), intent(in), optional :: output
    logical, intent(in), optional :: grouped, logplot
    real(kp), intent(in), optional :: begin_kp, end_kp
    real(kp) :: begin_loc, end_loc
    character(:), allocatable :: output_loc
    real(kp), allocatable :: buffer(:,:)[:]
    integer :: points_loc, io_unit, saved_indices(size(fitfuncs(1)%pars))
    integer :: i, j, k
    ! STEP 1 - Process arguments
    if (present(begin_kp)) then
       begin_loc = begin_kp
    else if (present(begin)) then
       begin_loc = begin
    else if (data_positions(2) == 1) then
       call error(__FILE__, __LINE__, 'Since no datasets are loaded, &
            &the lowest x-value must be explicitly given.')
    else
       if (.not. allocated(x_data)) call read_data()
       begin_loc = x_data(1)
    end if
    if (present(end_kp)) then
       end_loc = end_kp
    else if (present(end)) then
       end_loc = end
    else if (data_positions(2) == 1) then
       call error(__FILE__, __LINE__, 'Since no datasets are loaded, &
            &the highest x-value must be explicitly given.')
    else
       if (.not. allocated(x_data)) call read_data()
       end_loc = x_data(size(x_data))
    end if
    if (present(output)) then
       output_loc = output
    else
       output_loc = 'out'
    end if
    if (present(points)) then
       points_loc = points - mod(points, num_images())
       if (points_loc <= 0) points_loc = num_images()
       if (this_image() == 1 .and. mod(points, num_images()) /= 0) &
            & call comment(__FILE__, __LINE__, &
            & str(points_loc)//' points are printed.')
    else
       points_loc = 200-mod(200, num_images()) ! DEFAULT 200 points
       if (points_loc <= 0) points_loc = num_images()
    end if
    ! STEP 2 - Fill buffer
    allocate(buffer(size(fitfuncs)+1, points_loc/num_images())[*], &
         & stat=err_stat, errmsg=err_msg)
    call check_err(__FILE__, __LINE__)
    associate(start => (this_image()-1)*points_loc/num_images()+1, &
         & finish => this_image()*points_loc/num_images()) ! x-values
      if (present(logplot) .and. logplot) then
         buffer(1,:) = [(exp(log(begin_loc) + &
              & (i-1)*(log(end_loc)-log(begin_loc))/(points_loc-1)), &
              & i = start, finish)]
      else
         buffer(1,:) = [(begin_loc + &
              & (i-1)*(end_loc - begin_loc)/(points_loc-1), i=start, finish)]
      end if
    end associate
    saved_indices = 0
    do j = 1, size(fitfuncs) ! y-values
#ifdef POLYM_ARRAY_SUPPORT
       call swap(fitfuncs(j)%pars%index, saved_indices)
       do i = 1, points_loc/num_images()
          buffer(j+1,i) = fitfuncs(j)%eval(buffer(1,i))
       end do
       call swap(fitfuncs(j)%pars%index, saved_indices)
#else
       fitfunc_tmp => fitfuncs(j)
       call swap(fitfunc_tmp%pars%index, saved_indices)
       do i = 1, points_loc/num_images()
          buffer(j+1,i) = fitfuncs(j)%eval(buffer(1,i))
       end do
       call swap(fitfunc_tmp%pars%index, saved_indices)
#endif
    end do
    sync all
    ! STEP 3 - Print to I/O device(s)
    first_image: if (this_image() == 1) then
       print_results: if (size(fitfuncs) == 1 .or. .not. present(grouped) .or. &
            & (present(grouped) .and. grouped)) then ! Single output
          open(newunit=io_unit, file=output_loc, action='write', &
               & form='formatted', iostat=err_stat, iomsg=err_msg)
          call check_err(__FILE__, __LINE__)
          do j = 1, num_images()
             do i = 1, size(buffer, 2)
                write(io_unit, '(g0, '//str(size(fitfuncs))//'(1x, g0))') &
                     & buffer(:,i)[j]
             end do
          end do
          call safe_close(__FILE__, __LINE__, io_unit)
       else ! Multiple outputs
          do k = 1, size(fitfuncs)
             open(newunit=io_unit, file=output_loc//str(k), action='write', &
                  & form='formatted', iostat=err_stat, iomsg=err_msg)
             call check_err(__FILE__, __LINE__)
             do j = 1, num_images()
                do i = 1, size(buffer, 2)
                   write(io_unit, '(g0, 1x, g0)') &
                        & buffer(1,i)[j], buffer(1+k,i)[j]
                end do
             end do
             call safe_close(__FILE__, __LINE__, io_unit)
          end do
       end if print_results
       ! Print the rest only if a fitting procedure has been performed
       print_results_and_log: if (iterations > 0) then
          i = show_scope
          j = show_digits
          show_scope = GLOBAL_AND_LOCAL
          show_digits = -int(log10(epsilon(1.0_kp)), kind(1))
          open(newunit=io_unit, file=output_loc//'_parameters', &
               & action='write', form='formatted', &
               & iostat=err_stat, iomsg=err_msg)
          call check_err(__FILE__, __LINE__)
          call print_header()
          write(io_unit, '(g0)')
          call print_info(io_unit)
          show_scope = i
          show_digits = j
          close(io_unit)
          open(newunit=io_unit, file=output_loc//'_log', action='write', &
               & form='formatted', iostat=err_stat, iomsg=err_msg)
          call check_err(__FILE__, __LINE__)
          call print_header()
          call print_memory_usage(io_unit)
          call print_workloads(io_unit)
          call print_timing(io_unit)
          close(io_unit)
       end if print_results_and_log
    end if first_image
  contains
    subroutine print_header()
      character(10) :: date, time
      write(io_unit, '(*(g0))') &
           & 'GADfit version ', GADfit_VERSION_MAJOR, '.', &
           & GADfit_VERSION_MINOR, '.', GADfit_VERSION_PATCH
      write(io_unit, '(*(g0))') 'Platform: ', CMAKE_PLATFORM
      write(io_unit, '(*(g0))') 'Fortran compiler: ', CMAKE_FORTRAN_COMPILER, ''
      call date_and_time(date, time)
      write(io_unit, '(*(g0))') &
           & 'Calculation finished: ', time(1:2), ':', time(3:4), ':', &
           & time(5:6), ' ', date(7:8), '.', date(5:6), '.', date(1:4)
    end subroutine print_header
  end subroutine gadf_print
  !!***

  !!****f* gadfit/gadf_close
  !!
  !! FUNCTION
  !!
  !! Deallocates all the remaining work arrays. After this, it is safe
  !! to call gadf_init again.
  !!
  !! SOURCE
  subroutine gadf_close()
    use, intrinsic :: iso_fortran_env, only: output_unit
    if (out_unit /= output_unit) call safe_close(__FILE__, __LINE__, out_unit)
    call safe_deallocate(__FILE__, __LINE__, fitfuncs)
    call safe_deallocate(__FILE__, __LINE__, active_pars)
    call safe_deallocate(__FILE__, __LINE__, is_global)
    call safe_deallocate(__FILE__, __LINE__, x_data)
    call safe_deallocate(__FILE__, __LINE__, y_data)
    call safe_deallocate(__FILE__, __LINE__, weights)
    call safe_deallocate(__FILE__, __LINE__, data_units)
    call safe_deallocate(__FILE__, __LINE__, data_positions)
    call free_integration()
    call ad_close()
  end subroutine gadf_close
  !!***
end module gadfit

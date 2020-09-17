! This Source Code Form is subject to the terms of the GNU General
! Public License, v. 3.0. If a copy of the GPL was not distributed
! with this file, You can obtain one at
! http://gnu.org/copyleft/gpl.txt.

! AD implementation of adaptive numerical integration based on the
! Gauss-Kronrod rules.
module numerical_integration

  use ad
  use gadf_constants, only: kp
  use gauss_kronrod_parameters
  use messaging
  use misc,           only: swap

  implicit none

  private
  public :: workspace, workspace_init, workspace_destroy, GAUSS_KRONROD_15P, &
       & GAUSS_KRONROD_21P, GAUSS_KRONROD_31P, GAUSS_KRONROD_41P, &
       & GAUSS_KRONROD_51P, GAUSS_KRONROD_61P, init_integration, &
       & init_integration_dbl, set_integration_rule, INFINITY, integrate, &
       & free_integration, numerical_integration_memory_report

  enum, bind(c)
     enumerator :: &
          & GAUSS_KRONROD_15P, GAUSS_KRONROD_21P, GAUSS_KRONROD_31P, &
          & GAUSS_KRONROD_41P, GAUSS_KRONROD_51P, GAUSS_KRONROD_61P
     enumerator :: INFINITY = 521207248
  end enum

  integer, parameter :: DEFAULT_INTEGRATION_RULE = GAUSS_KRONROD_15P
  integer, parameter :: DEFAULT_WORKSPACE_SIZE = 1000

  type workspace
     ! Integration workspace. Contains work arrays for the lower and
     ! upper bounds, sums, and absolute errors. Each array element
     ! corresponds to an integration interval.
     type(advar), allocatable :: sums(:)
     real(kp), allocatable :: lower(:), upper(:), abs_error(:)
   contains
     procedure :: init => workspace_init
     procedure :: destroy => workspace_destroy
  end type workspace

  interface integrate
     module procedure integrate_real_real, integrate_real_inf, &
          & integrate_inf_real, integrate_inf_inf, integrate_advar_advar, &
          & integrate_advar_real, integrate_real_advar, integrate_advar_inf, &
          & integrate_inf_advar
  end interface integrate

  ! Default relative errors for the inner and outer integral
  real(kp) :: default_rel_error_inner = 1e+2*epsilon(1.0_kp)
  real(kp) :: default_rel_error_outer = 1e+3*epsilon(1.0_kp)
  ! The 2n+1 nodes of the Legendre polynomials, the weights
  ! corresponding to the n-point Gaussian rule, and weights
  ! corresponding to the (2n+1)-point Kronrod rule.
  real(kp), allocatable :: roots(:), weights_gauss(:), weights_kronrod(:)
  ! ws(1) and ws(2) are the workspaces for outer and inner integrals,
  ! respectively. In principle, higher order integrals could be
  ! allowed by simply increasing the size of this array.
  type(workspace), save :: ws(2)
  ! Maximum value of current_index during any integration
  ! procedure. For memory report.
  integer :: max_ws(2)[*]

contains

  ! Allocates lower(size), upper(size), sums(size), and
  ! abs_error(size).
  subroutine workspace_init(this, size)
    class(workspace), intent(in out) :: this
    integer, intent(in), optional :: size
    integer :: size_loc
    size_loc = DEFAULT_WORKSPACE_SIZE
    if (present(size)) size_loc = size
    if (allocated(this%lower)) then
       if (this_image() == 1) &
            & call comment(__FILE__, __LINE__, &
            & 'The workspace is already allocated. Reallocating ...')
       call this%destroy()
    end if
    allocate(this%lower(size_loc), this%upper(size_loc), this%sums(size_loc), &
         & this%abs_error(size_loc), stat=err_stat, errmsg=err_msg)
    call check_err(__FILE__, __LINE__)
  end subroutine workspace_init

  impure elemental subroutine workspace_destroy(this)
    class(workspace), intent(in out) :: this
    call safe_deallocate(__FILE__, __LINE__, this%lower)
    call safe_deallocate(__FILE__, __LINE__, this%upper)
    call safe_deallocate(__FILE__, __LINE__, this%sums)
    call safe_deallocate(__FILE__, __LINE__, this%abs_error)
  end subroutine workspace_destroy

  ! Sets the default relative integration error, initializes a
  ! workspace, and sets the integration rule.
  !
  ! Inputs all optional
  !
  ! rel_error - Relative error. This can be overridden by an optional
  !             argument to 'integrate'.
  ! workspace_size - workspace_init is called with this argument. The
  !                  actual workspace size is 4*workspace_size.
  ! integration_rule - set_integration_rule is called with this rule
  subroutine init_integration(rel_error, workspace_size, integration_rule)
    real(kp), intent(in), optional :: rel_error
    integer, intent(in), optional :: workspace_size, integration_rule
    if (.not. allocated(ws(2)%lower)) &
         & default_rel_error_outer = default_rel_error_inner
    if (present(rel_error)) default_rel_error_outer = rel_error
    call ws(1)%init(workspace_size)
    call set_integration_rule(integration_rule)
    max_ws = 0
  end subroutine init_integration

  ! Same as init_integration except that two workspaces are
  ! initialized and two relative errors defined.
  subroutine init_integration_dbl(rel_error_inner, rel_error_outer, &
       & ws_size_inner, ws_size_outer, integration_rule)
    real(kp), intent(in), optional :: rel_error_inner, rel_error_outer
    integer, intent(in), optional :: ws_size_inner, ws_size_outer
    integer, intent(in), optional :: integration_rule
    if (present(rel_error_inner)) default_rel_error_inner = rel_error_inner
    call ws(2)%init(ws_size_inner)
    call init_integration(rel_error_outer, ws_size_outer, integration_rule)
  end subroutine init_integration_dbl

  ! Initializes roots, gauss_weights, and kronrod_weights as specified
  ! by the integration rule.
  subroutine set_integration_rule(rule)
    integer, intent(in), optional :: rule
    integer :: rule_loc
    rule_loc = DEFAULT_INTEGRATION_RULE
    if (present(rule)) rule_loc = rule
    select case(rule_loc)
    case(GAUSS_KRONROD_15P)
       call init(roots_15p, weights_gauss_7p, weights_kronrod_15p)
    case(GAUSS_KRONROD_21P)
       call init(roots_21p, weights_gauss_10p, weights_kronrod_21p)
    case(GAUSS_KRONROD_31P)
       call init(roots_31p, weights_gauss_15p, weights_kronrod_31p)
    case(GAUSS_KRONROD_41P)
       call init(roots_41p, weights_gauss_20p, weights_kronrod_41p)
    case(GAUSS_KRONROD_51P)
       call init(roots_51p, weights_gauss_25p, weights_kronrod_51p)
    case(GAUSS_KRONROD_61P)
       call init(roots_61p, weights_gauss_30p, weights_kronrod_61p)
    case default
       call error(__FILE__, __LINE__, &
            & 'Invalid input. The following rules are available: &
            &GAUSS_KRONROD_15p, GAUSS_KRONROD_21p, GAUSS_KRONROD_31p, &
            &GAUSS_KRONROD_41p, GAUSS_KRONROD_51p, and GAUSS_KRONROD_61p.')
    end select
  contains
    subroutine init(r, wg, wk)
      real(kp), intent(in) :: r(:), wg(:), wk(:)
      allocate(roots(size(r)), weights_gauss(size(wg)), &
           & weights_kronrod(size(wk)), stat=err_stat, errmsg=err_msg)
      call check_err(__FILE__, __LINE__)
      roots = r; weights_gauss = wg; weights_kronrod = wk
    end subroutine init
  end subroutine set_integration_rule

  ! Performs adaptive integration for a user defined function. The
  ! integration interval is divided into subintervals, and on each
  ! iteration the subinterval with the largest error is
  ! processed. During the search for the optimal set of subintervals,
  ! AD is switched off by making all parameters passive. When
  ! convergence has been achieved, AD is switched on and the
  ! Gauss-Kronrod algorithm is applied once more with the final set of
  ! subintervals. integrate_real_real is the main engine under the
  ! 'integrate' interface. The other procedures contain calls to
  ! integrate_real_real.
  !
  ! Arguments of all members of the 'integrate' interface have the
  ! same meaning.
  !
  ! f - the integrand
  ! pars - parameters of f
  ! lower - lower integration bound
  ! upper - upper integration bound
  ! rel_error (optional) - relative error bound
  ! abs_error (optional) - absolute error bound
  type(advar) recursive function integrate_real_real(f, pars, lower, upper, &
       & rel_error, abs_error) result(y)
    interface
       type(advar) function f(x, pars)
         import kp, advar
         real(kp), intent(in) :: x
         type(advar), intent(in out) :: pars(:)
       end function f
    end interface
    type(advar), intent(in out) :: pars(:)
    real(kp), intent(in) :: lower, upper
    real(kp), intent(in), optional :: rel_error, abs_error
    real(kp) :: rel_error_loc, abs_error_loc
    ! Local integration bounds
    real(kp) :: aa, middle, bb
    ! Tells whether we are currently in the inner or outer integral.
    integer, save :: int_order = 0 ! ifort fails without 'save'!
    ! Flag for whether ws(2) was automatically initialized
    logical :: ws2_auto_init = .false.
    ! Iterator of the main loop
    integer :: current_size
    ! Index of the subinterval with the largest error
    integer :: max_error_index
    ! For turning on/off AD by swapping indices
    integer :: saved_indices(size(pars))
    integer :: i
    int_order = int_order + 1
    ! Auto-initialize workspaces
    if (.not. allocated(ws(1)%lower)) call init_integration()
    if (.not. allocated(ws(2)%lower) .and. int_order == 2) then
       call ws(2)%init()
       ws2_auto_init = .true.
    end if
    ! Error tolerances
    if (present(rel_error)) then
       rel_error_loc = rel_error
    else if (int_order == 1) then
       rel_error_loc = default_rel_error_outer
    else
       rel_error_loc = default_rel_error_inner
    end if
    abs_error_loc = 0.0_kp
    if (present(abs_error)) abs_error_loc = abs_error
    ! Switch off AD
    saved_indices = 0
    call swap(pars%index, saved_indices)
    ! First approximation to the integral
    ws(int_order)%lower(1) = lower
    ws(int_order)%upper(1) = upper
    ws(int_order)%sums(1) = gauss_kronrod(f, pars, lower, upper, &
         & ws(int_order)%abs_error(1))
    if (ws2_auto_init .and. int_order == 1) then
       ! If the workspace for the inner integral was auto-initialized,
       ! then reduce the error tolerance of the outer integral.
       rel_error_loc = default_rel_error_outer
       ws2_auto_init = .false.
    end if
    main: do current_size = 1, size(ws(int_order)%sums)-1
       max_error_index = maxloc(ws(int_order)%abs_error(:current_size), 1)
       aa = ws(int_order)%lower(max_error_index)
       bb = ws(int_order)%upper(max_error_index)
       middle = (aa+bb)/2
       ws(int_order)%sums(max_error_index) = &
            & gauss_kronrod(f, pars, aa, middle, &
            & ws(int_order)%abs_error(max_error_index))
       ws(int_order)%sums(current_size+1) = gauss_kronrod(f, pars, middle, bb, &
            & ws(int_order)%abs_error(current_size+1))
       ws(int_order)%upper(max_error_index) = middle
       ws(int_order)%lower(current_size+1) = middle
       ws(int_order)%upper(current_size+1) = bb
       if (sum(ws(int_order)%abs_error(:current_size+1)) < abs_error_loc .or. &
            & sum(ws(int_order)%abs_error(:current_size+1))/ &
            & sum(ws(int_order)%sums(:current_size+1)%val) < rel_error_loc) &
            & then
          ! Turn on AD for the final result
          call swap(pars%index, saved_indices)
          y = 0.0_kp
          do i = 1, current_size + 1
             y = y + gauss_kronrod(f, pars, &
                  & ws(int_order)%lower(i), ws(int_order)%upper(i), &
                  & ws(int_order)%abs_error(max_error_index))
          end do
          ! For memory report
          max_ws(int_order) = max(max_ws(int_order), current_size)
          int_order = int_order - 1
          return
       end if
    end do main
    call error(__FILE__, __LINE__, 'Number of iterations was insufficient. &
         &Increase either workspace size or the error bound(s).')
  end function integrate_real_real

  ! Same as integrate_real_real except the upper bound is -INFINITY or
  ! INFINITY. For INFINITY the integral over (a,infinity) is mapped
  ! into the semi-open interval (0,1] using the transformation
  ! x=a+(1-t)/t. For -INFINITY -integrate_inf_real is called with the
  ! appropriate bounds.
  type(advar) recursive function integrate_real_inf(f, pars, lower, upper, &
       & rel_error, abs_error) result(y)
    interface
       type(advar) function f(x, pars)
         import kp, advar
         real(kp), intent(in) :: x
         type(advar), intent(in out) :: pars(:)
       end function f
    end interface
    type(advar), intent(in out) :: pars(:)
    real(kp), intent(in) :: lower
    integer, intent(in) :: upper
    real(kp), intent(in), optional :: rel_error, abs_error
    if (upper == INFINITY) then
       y = integrate_real_real(f_transformed, pars, 0.0_kp, 1.0_kp, &
            & rel_error, abs_error)
    else if (upper == -INFINITY) then
       y = -integrate_inf_real(f, pars, -INFINITY, lower, rel_error, abs_error)
    else
       call error(__FILE__, __LINE__, 'Incorrect upper bound. &
            &Use either (+-)INFINITY or a real number.')
    end if
  contains
    type(advar) function f_transformed(x, pars) result(y)
      real(kp), intent(in) :: x
      type(advar), intent(in out) :: pars(:)
      y = f(lower - 1 + 1/x, pars)/x**2
    end function f_transformed
  end function integrate_real_inf

  ! Similar to integrate_real_inf except the lower bound is -INFINITY
  ! or INFINITY. For the case of -INFINITY, -integrate_real_inf is
  ! called with the appropriate bounds.
  type(advar) recursive function integrate_inf_real(f, pars, lower, upper, &
       & rel_error, abs_error) result(y)
    interface
       type(advar) function f(x, pars)
         import kp, advar
         real(kp), intent(in) :: x
         type(advar), intent(in out) :: pars(:)
       end function f
    end interface
    type(advar), intent(in out) :: pars(:)
    integer, intent(in) :: lower
    real(kp), intent(in) :: upper
    real(kp), intent(in), optional :: rel_error, abs_error
    if (lower == -INFINITY) then
       y = integrate_real_real(f_transformed, pars, 0.0_kp, 1.0_kp, &
            & rel_error, abs_error)
    else if (lower == INFINITY) then
       y = -integrate_real_inf(f, pars, upper, INFINITY, rel_error, abs_error)
    else
       call error(__FILE__, __LINE__, 'Incorrect lower bound. &
            &Use either (+-)INFINITY or a real number.')
    end if
  contains
    type(advar) function f_transformed(x, pars) result(y)
      real(kp), intent(in) :: x
      type(advar), intent(in out) :: pars(:)
      y = f(upper + 1 - 1/x, pars)/x**2
    end function f_transformed
  end function integrate_inf_real

  type(advar) recursive function integrate_inf_inf(f, pars, lower, upper, &
       & rel_error, abs_error) result(y)
    interface
       type(advar) function f(x, pars)
         import kp, advar
         real(kp), intent(in) :: x
         type(advar), intent(in out) :: pars(:)
       end function f
    end interface
    type(advar), intent(in out) :: pars(:)
    integer, intent(in) :: lower
    integer, intent(in) :: upper
    real(kp), intent(in), optional :: rel_error, abs_error
    y = integrate_inf_real(f, pars, lower, 0.0_kp, rel_error, abs_error) + &
         & integrate_real_inf(f, pars, 0.0_kp, upper, rel_error, abs_error)
  end function integrate_inf_inf

  ! Similar to integrate_real_real except that both bounds are of
  ! type(advar). If either the integrand or either bound is active
  ! (nonzero index), then in the reverse mode, the operation is
  ! recorded for the return sweep by appending to trace,
  ! forward_values, and ad_constants (see the AD module). In the
  ! forward mode the derivatives are calculated on the fly.
  type(advar) recursive function integrate_advar_advar(f, pars, lower, upper, &
       & rel_error, abs_error) result(y)
    interface
       type(advar) function f(x, pars)
         import kp, advar
         real(kp), intent(in) :: x
         type(advar), intent(in out) :: pars(:)
       end function f
    end interface
    type(advar), intent(in out) :: pars(:)
    type(advar), intent(in) :: lower
    type(advar), intent(in) :: upper
    real(kp), intent(in), optional :: rel_error, abs_error
    integer :: saved_indices(size(pars))
    type(advar) :: dummy
    if (lower%index /= 0 .and. upper%index == 0) then
       y = integrate_advar_real(f, pars, lower, upper%val, rel_error, abs_error)
       return
    else if (lower%index == 0 .and. upper%index /= 0) then
       y = integrate_real_advar(f, pars, lower%val, upper, rel_error, abs_error)
       return
    end if
    y = integrate_real_real(f, pars, lower%val, upper%val, rel_error, abs_error)
    if (lower%index /= 0 .and. upper%index /= 0) then
       if (reverse_mode) then
          saved_indices = 0
          if (y%index == 0) then
             index_count = index_count + 1
             forward_values(index_count) = y%val
             y%index = index_count
          end if
          if (y%index /= 0) call swap(pars%index, saved_indices)
          const_count = const_count + 1
          ad_constants(const_count) = f(lower%val, pars)
          const_count = const_count + 1
          ad_constants(const_count) = f(upper%val, pars)
          if (y%index /= 0) call swap(pars%index, saved_indices)
          trace(trace_count+1) = lower%index
          trace(trace_count+2) = upper%index
          trace(trace_count+3) = y%index
          trace(trace_count+4) = INT_BOTH_BOUNDS
          trace_count = trace_count + 4
       else
          y%d = y%d - lower%d*f(lower%val, pars) + upper%d*f(upper%val, pars)
          dummy = f(lower%val, pars)
          y%dd = y%dd - lower%dd*dummy%val - &
               & lower%d*(dir_deriv_finite(f, lower, pars)+dummy%d)
          dummy = f(upper%val, pars)
          y%dd = y%dd + upper%dd*dummy%val + &
               & upper%d*(dir_deriv_finite(f, upper, pars)+dummy%d)
          if (y%index == 0) y%index = 1
       end if
    end if
  end function integrate_advar_advar

  ! Similar to integrate_advar_advar except that the upper bound is a
  ! real number. integrate_advar_advar calls this with a passive upper
  ! bound.
  type(advar) recursive function integrate_advar_real(f, pars, lower, upper, &
       & rel_error, abs_error) result(y)
    interface
       type(advar) function f(x, pars)
         import kp, advar
         real(kp), intent(in) :: x
         type(advar), intent(in out) :: pars(:)
       end function f
    end interface
    type(advar), intent(in out) :: pars(:)
    type(advar), intent(in) :: lower
    real(kp), intent(in) :: upper
    real(kp), intent(in), optional :: rel_error, abs_error
    integer :: saved_indices(size(pars))
    type(advar) :: dummy
    y = integrate_real_real(f, pars, lower%val, upper, rel_error, abs_error)
    if (lower%index /= 0) then
       if (reverse_mode) then
          saved_indices = 0
          if (y%index == 0) then
             index_count = index_count + 1
             forward_values(index_count) = y%val
             y%index = index_count
          end if
          if (y%index /= 0) call swap(pars%index, saved_indices)
          const_count = const_count + 1
          ad_constants(const_count) = f(lower%val, pars)
          if (y%index /= 0) call swap(pars%index, saved_indices)
          trace(trace_count+1) = lower%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = INT_LOWER_BOUND
          trace_count = trace_count + 3
       else
          y%d = y%d - lower%d*f(lower%val, pars)
          dummy = f(lower%val, pars)
          y%dd = y%dd - lower%dd*dummy%val - &
               & lower%d*(dir_deriv_finite(f, lower, pars)+dummy%d)
          if (y%index == 0) y%index = 1
       end if
    end if
  end function integrate_advar_real

  ! Same as integrate_advar_real except the lower bound is real.
  type(advar) recursive function integrate_real_advar(f, pars, lower, upper, &
       & rel_error, abs_error) result(y)
    interface
       type(advar) function f(x, pars)
         import kp, advar
         real(kp), intent(in) :: x
         type(advar), intent(in out) :: pars(:)
       end function f
    end interface
    type(advar), intent(in out) :: pars(:)
    real(kp), intent(in) :: lower
    type(advar), intent(in) :: upper
    real(kp), intent(in), optional :: rel_error, abs_error
    integer :: saved_indices(size(pars))
    type(advar) :: dummy
    y = integrate_real_real(f, pars, lower, upper%val, rel_error, abs_error)
    if (upper%index /= 0) then
       if (reverse_mode) then
          saved_indices = 0
          if (y%index == 0) then
             index_count = index_count + 1
             forward_values(index_count) = y%val
             y%index = index_count
          end if
          if (y%index /= 0) call swap(pars%index, saved_indices)
          const_count = const_count + 1
          ad_constants(const_count) = f(upper%val, pars)
          if (y%index /= 0) call swap(pars%index, saved_indices)
          trace(trace_count+1) = upper%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = INT_UPPER_BOUND
          trace_count = trace_count + 3
       else
          y%d = y%d + upper%d*f(upper%val, pars)
          dummy = f(upper%val, pars)
          y%dd = y%dd + upper%dd*dummy%val + &
               & upper%d*(dir_deriv_finite(f, upper, pars)+dummy%d)
          if (y%index == 0) y%index = 1
       end if
    end if
  end function integrate_real_advar

  ! Same as integrate_real_advar except the lower bound is +-INFINITY.
  type(advar) recursive function integrate_inf_advar(f, pars, lower, upper, &
       & rel_error, abs_error) result(y)
    interface
       type(advar) function f(x, pars)
         import kp, advar
         real(kp), intent(in) :: x
         type(advar), intent(in out) :: pars(:)
       end function f
    end interface
    type(advar), intent(in out) :: pars(:)
    integer, intent(in) :: lower
    type(advar), intent(in) :: upper
    real(kp), intent(in), optional :: rel_error, abs_error
    integer :: saved_indices(size(pars))
    type(advar) :: dummy
    y = integrate_inf_real(f, pars, lower, upper%val, rel_error, abs_error)
    if (upper%index /= 0) then
       if (reverse_mode) then
          saved_indices = 0
          if (y%index /= 0) call swap(pars%index, saved_indices)
          const_count = const_count + 1
          ad_constants(const_count) = f(upper%val, pars)
          if (y%index /= 0) call swap(pars%index, saved_indices)
          if (y%index == 0) then
             index_count = index_count + 1
             forward_values(index_count) = y%val
             y%index = index_count
          end if
          trace(trace_count+1) = upper%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = INT_UPPER_BOUND
          trace_count = trace_count + 3
       else
          y%d = y%d + upper%d*f(upper%val, pars)
          dummy = f(upper%val, pars)
          y%dd = y%dd + upper%dd*dummy%val + &
               & upper%d*(dir_deriv_finite(f, upper, pars)+dummy%d)
          if (y%index == 0) y%index = 1
       end if
    end if
  end function integrate_inf_advar

  ! Same as integrate_advar_real except the upper bound is +-INFINITY.
  type(advar) recursive function integrate_advar_inf(f, pars, lower, upper, &
       & rel_error, abs_error) result(y)
    interface
       type(advar) function f(x, pars)
         import kp, advar
         real(kp), intent(in) :: x
         type(advar), intent(in out) :: pars(:)
       end function f
    end interface
    type(advar), intent(in out) :: pars(:)
    type(advar), intent(in) :: lower
    integer, intent(in) :: upper
    real(kp), intent(in), optional :: rel_error, abs_error
    integer :: saved_indices(size(pars))
    type(advar) :: dummy
    y = integrate_real_inf(f, pars, lower%val, upper, rel_error, abs_error)
    if (lower%index /= 0) then
       if (reverse_mode) then
          saved_indices = 0
          if (y%index /= 0) call swap(pars%index, saved_indices)
          const_count = const_count + 1
          ad_constants(const_count) = f(lower%val, pars)
          if (y%index /= 0) call swap(pars%index, saved_indices)
          if (y%index == 0) then
             index_count = index_count + 1
             forward_values(index_count) = y%val
             y%index = index_count
          end if
          trace(trace_count+1) = lower%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = INT_LOWER_BOUND
          trace_count = trace_count + 3
       else
          y%d = y%d - lower%d*f(lower%val, pars)
          dummy = f(lower%val, pars)
          y%dd = y%dd - lower%dd*dummy%val - &
               & lower%d*(dir_deriv_finite(f, lower, pars)+dummy%d)
          if (y%index == 0) y%index = 1
       end if
    end if
  end function integrate_advar_inf

  ! Main integration algorithm.
  !
  ! Those that are present have the same meaning as for
  ! integrate_real_real.
  type(advar) recursive function gauss_kronrod(f, pars, lower, upper, &
       & abs_error) result(y)
    interface
       type(advar) function f(x, pars)
         import kp, advar
         real(kp), intent(in) :: x
         type(advar), intent(in out) :: pars(:)
       end function f
    end interface
    type(advar), intent(in out) :: pars(:)
    real(kp), intent(in) :: lower, upper
    real(kp), intent(out) :: abs_error
    type(advar) :: f_value
    real(kp) :: scale, shift, sum_gauss
    integer :: i
    scale = (upper-lower)/2
    shift = (lower+upper)/2
    sum_gauss = 0.0_kp
    y = 0.0_kp
    do i = 1, size(roots)
       f_value = f(scale*roots(i)+shift, pars)
       if (mod(i,2) == 0) sum_gauss = sum_gauss + weights_gauss(i/2)*f_value%val
       y = y + weights_kronrod(i)*f_value
    end do
    y = scale*y
    abs_error = abs(y%val - scale*sum_gauss)
  end function gauss_kronrod

  ! The directional derivative of the integrand f(b(p),p) with respect
  ! to the parameter vector p, where b is either the lower or upper
  ! bound of integration. It is taken along the direction pars%d. It
  ! is explained in the user guide why we use finite differences here.
  !
  ! f - integrand
  ! b - lower or upper bound
  ! pars - parameters of f
  ! y - directional derivative
  real(kp) function dir_deriv_finite(f, b, pars) result(y)
    interface
       type(advar) function f(x, pars)
         import kp, advar
         real(kp), intent(in) :: x
         type(advar), intent(in out) :: pars(:)
       end function f
    end interface
    type(advar), intent(in) :: b
    type(advar), intent(in out) :: pars(:)
    real(kp) :: saved_values(size(pars))
    integer :: saved_indices(size(pars))
    saved_indices = 0
    call swap(saved_indices, pars%index)
    saved_values = pars%val
    y = f(b%val, pars)
    associate(h => sqrt(epsilon(1.0_kp)))
      pars%val = pars%val + h*pars%d
      y = (f(b%val+h*b%d, pars) - y)/h
    end associate
    pars%val = saved_values
    call swap(saved_indices, pars%index)
  end function dir_deriv_finite

  ! Prints the size(s) of the workspace(s) according to what was
  ! requested and actual usage. The results represent maxima over all
  ! images.
  subroutine numerical_integration_memory_report(io_unit)
    use, intrinsic :: iso_fortran_env, only: output_unit
    integer, intent(in), optional :: io_unit
    integer :: io_unit_loc
    integer :: i
    if (.not. allocated(ws(1)%lower) .or. this_image() /= 1) return
    io_unit_loc = output_unit
    if (present(io_unit)) io_unit_loc = io_unit
    do i = 2, num_images()
       max_ws(1) = max(max_ws(1), max_ws(1)[i])
       max_ws(2) = max(max_ws(2), max_ws(2)[i])
    end do
    write(io_unit_loc, '(1x, g0)') 'Numerical integration memory usage'
    write(io_unit_loc, '(1x, g0)') '=================================='
    if (allocated(ws(1)%sums)) then
       if_dbl: if (allocated(ws(2)%sums)) then
          write(io_unit_loc, '(2x, g0)', advance='no') &
               & 'Requested (inner integral): '
       else
          write(io_unit_loc, '(2x, g0)', advance='no') 'Requested: '
       end if if_dbl
       call print_memory(io_unit_loc, 4*kp*size(ws(1)%sums))
       write(io_unit_loc, '(1x, *(g0))') '(4x', size(ws(1)%sums), ')'
       if (allocated(ws(2)%sums)) then
          write(io_unit_loc, '(2x, g0)', advance='no') &
               & 'Requested (outer integral): '
          call print_memory(io_unit_loc, 4*kp*size(ws(2)%sums))
          write(io_unit_loc, '(1x, *(g0))') '(4x', size(ws(2)%sums), ')'
       end if
       if (allocated(ws(2)%sums)) then
          write(io_unit_loc, '(7x, g0)', advance='no') 'Used (inner integral): '
       else
          write(io_unit_loc, '(7x, g0)', advance='no') 'Used: '
       end if
       call print_memory(io_unit_loc, 4*kp*max_ws(1))
       write(io_unit_loc, '(1x, *(g0))') '(4x', max_ws(1), ')'
       if (allocated(ws(2)%sums)) then
          write(io_unit_loc, '(7x, g0)', advance='no') 'Used (outer integral): '
          call print_memory(io_unit_loc, 4*kp*max_ws(2))
          write(io_unit_loc, '(1x, *(g0))') '(4x', max_ws(2), ')'
       end if
    end if
    write(io_unit, '(g0)')
  end subroutine numerical_integration_memory_report

  subroutine free_integration()
    call safe_deallocate(__FILE__, __LINE__, roots)
    call safe_deallocate(__FILE__, __LINE__, weights_gauss)
    call safe_deallocate(__FILE__, __LINE__, weights_kronrod)
    call ws%destroy()
  end subroutine free_integration
end module numerical_integration

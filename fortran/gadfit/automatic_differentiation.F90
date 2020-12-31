! Licensed under the Apache License, Version 2.0 (the "License"); you
! may not use this file except in compliance with the License.  You
! may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
! implied.  See the License for the specific language governing
! permissions and limitations under the License.

! Procedures for finding the gradient of a function using the reverse
! mode of automatic differentiation (AD), and procedures for finding
! the first and second directional derivatives using the forward
! mode. This includes the implementation of all elemental operations,
! except numerical integration, which is given in
! numerical_integration.f90. Also provides the definition of
! type(advar) (the AD variable), which is the central variable used in
! GADfit. See the user guide for details about the AD algorithm.
module ad

  use, intrinsic :: iso_c_binding,   only: c_double
  use, intrinsic :: iso_fortran_env, only: real32
  use gadf_constants, only: dp, kp, qp, sqrtpi
  use messaging
  use misc,           only: safe_deallocate

  implicit none

  public
  private :: c_double, real32, dp, kp, qp, sqrtpi, DEFAULT_SWEEP_SIZE, &
       & max_trace_count, max_index_count, max_const_count

  ! Operation codes
  enum, bind(c)
     enumerator :: ADD_A_A = -1, ADD_SUBTRACT_A_R = -2, &
          & SUBTRACT_A_A = -3, SUBTRACT_R_A = -4, &
          & MULTIPLY_A_A = -5, MULTIPLY_DIVIDE_A_R = -6, &
          & DIVIDE_A_A = -7, DIVIDE_R_A = -8, &
          & POWER_A_A = -9, POWER_A_R = -10, POWER_R_A = -11, &
          & POWER_INTEGER = -12, ABS_A = -13
     enumerator :: EXP_A = -14, SQRT_A = -15, LOG_A = -16, &
          & SIN_A = -17, COS_A = -18, TAN_A = -19, &
          & ASIN_A = -20, ACOS_A = -21, ATAN_A = -22, &
          & SINH_A = -23, COSH_A = -24, TANH_A = -25, &
          & ASINH_A = -26, ACOSH_A = -27, ATANH_A = -28
     enumerator :: ERF_A = -29, LI2_A = -30
     enumerator :: INT_LOWER_BOUND = -31, INT_UPPER_BOUND = -32, &
          & INT_BOTH_BOUNDS = -33
     ! Notes:
     !   ADD_SUBTRACT_A_R - advar+real, real+advar, and advar-real
     !                      share the same derivative code
     !   MULTIPLY_DIVIDE_A_R - advar*const and advar/const share the
     !                         same derivative code.
     !   Negative values are for debugging purposes; all other
     !   elements in trace are positive, except possibly a negative
     !   integer exponent (see power_advar_integer).
  end enum

  ! Default size of forward_values and adjoints
  integer, parameter :: DEFAULT_SWEEP_SIZE = 10000

  ! The AD variable
  type advar
     real(kp) :: val, d = 0.0_kp, dd = 0.0_kp
     integer :: index = 0
   contains
     procedure :: assign_advar_integer
     procedure, pass(this) :: assign_integer_advar
     procedure :: assign_advar_real32
     procedure, pass(this) :: assign_real32_advar
     procedure :: assign_advar_dp
     procedure, pass(this) :: assign_dp_advar
     procedure :: assign_advar_qp
     procedure, pass(this) :: assign_qp_advar
     generic :: assignment(=) => assign_advar_integer, assign_integer_advar, &
          & assign_advar_real32, assign_real32_advar, assign_advar_dp, &
          & assign_dp_advar, assign_advar_qp, assign_qp_advar
  end type advar

  interface operator(>)
     module procedure advar_gt_advar, advar_gt_real32, real32_gt_advar, &
          & advar_gt_dp, dp_gt_advar, advar_gt_qp, qp_gt_advar
  end interface operator(>)

  interface operator(<)
     module procedure advar_lt_advar, advar_lt_real32, real32_lt_advar, &
          & advar_lt_dp, dp_lt_advar, advar_lt_qp, qp_lt_advar
  end interface operator(<)

  ! Most interfaces for safe_deallocate are in misc.f90. Adding this
  ! to misc.f90 would produce a circular dependence.
  interface safe_deallocate
     module procedure safe_deallocate_advar
  end interface safe_deallocate

  ! AD elemental operations. For most binary operations, all variants
  ! are present: (advar,advar), (advar,T), and (T,advar), where T can
  ! be a real(real32), real(dp), real(qp), or an integer. T
  ! is a real(kp) in the main implementation, while the others are
  ! wrapper procedures with calls to it. An exception is
  ! exponentiation with an integral power, where the integer is not
  ! converted to real(kp) but instead a separate implementation is
  ! provided for this.
  interface operator(+)
     module procedure add_advar_advar, add_advar_real, add_real_advar, &
          & add_advar_integer, add_integer_advar, add_real32_advar, &
          & add_advar_real32, &
#ifdef QUAD_PRECISION
          & add_advar_dp, add_dp_advar
#else
          & add_advar_qp, add_qp_advar
#endif
  end interface operator(+)

  interface operator(-)
     module procedure subtract_advar_advar, subtract_advar, &
          & subtract_advar_real, subtract_real_advar, subtract_advar_integer, &
          & subtract_integer_advar, subtract_advar_real32, &
          & subtract_real32_advar, &
#ifdef QUAD_PRECISION
          & subtract_advar_dp, subtract_dp_advar
#else
          & subtract_advar_qp, subtract_qp_advar
#endif
  end interface operator(-)

  interface operator(*)
     module procedure multiply_advar_advar, multiply_advar_real, &
          & multiply_real_advar, multiply_advar_integer, &
          & multiply_integer_advar, multiply_advar_real32, &
          & multiply_real32_advar, &
#ifdef QUAD_PRECISION
          & multiply_advar_dp, multiply_dp_advar
#else
          & multiply_advar_qp, multiply_qp_advar
#endif
  end interface operator(*)

  interface operator(/)
     module procedure divide_advar_advar, divide_advar_real, &
          & divide_real_advar, divide_advar_integer, divide_integer_advar, &
          & divide_advar_real32, divide_real32_advar, &
#ifdef QUAD_PRECISION
          & divide_advar_dp, divide_dp_advar
#else
          & divide_advar_qp, divide_qp_advar
#endif
  end interface operator(/)

  interface operator(**)
     module procedure power_advar_advar, power_advar_real, power_real_advar, &
          & power_advar_integer, power_integer_advar, power_advar_real32, &
          & power_real32_advar, &
#ifdef QUAD_PRECISION
          & power_advar_dp, power_dp_advar
#else
          & power_advar_qp, power_qp_advar
#endif
  end interface operator(**)

  interface abs ! abs(0.0) is considered positive
     module procedure abs_advar
  end interface abs

  interface exp
     module procedure exp_advar
  end interface exp

  interface sqrt
     module procedure sqrt_advar
  end interface sqrt

  interface log
     module procedure log_advar
  end interface log

  interface sin
     module procedure sin_advar
  end interface sin

  interface cos
     module procedure cos_advar
  end interface cos

  interface tan
     module procedure tan_advar
  end interface tan

  interface asin
     module procedure asin_advar
  end interface asin

  interface acos
     module procedure acos_advar
  end interface acos

  interface atan
     module procedure atan_advar
  end interface atan

  interface sinh
     module procedure sinh_advar
  end interface sinh

  interface cosh
     module procedure cosh_advar
  end interface cosh

  interface tanh
     module procedure tanh_advar
  end interface tanh

  interface asinh
     module procedure asinh_advar
  end interface asinh

  interface acosh
     module procedure acosh_advar
  end interface acosh

  interface atanh
     module procedure atanh_advar
  end interface atanh

  interface erf
     module procedure erf_advar
  end interface erf

#ifdef USE_GSL
  interface Li2
     real(c_double) function Li2_real(arg) bind(c, name='gsl_sf_dilog')
       import c_double
       real(c_double), intent(in), value :: arg
     end function Li2_real
     module procedure Li2_advar
  end interface Li2
#endif

  ! Arrays for the intermediate values, the adjoints, constants, and
  ! the execution trace.
  real(kp), allocatable :: forward_values(:), adjoints(:), ad_constants(:)
  integer,  allocatable :: trace(:)
  ! The number of nonzero elements in trace, index of the last active
  ! parameter (next one will be index_count+1), and the number of
  ! nonzero elements in ad_constants (excluding the ones that are zero
  ! by coincidence).
  integer :: trace_count, index_count, const_count
  ! Whether to use the reverse mode of AD (forward mode if .false.)
  logical :: reverse_mode = .false.
  ! Maximum number of nonzero elements encountered so far in trace,
  ! adjoints, and ad_constants. These are checked at each call to
  ! ad_grad. If the trace changes at any point, e.g. when dealing with
  ! a piecewise function, it is the maximum memory usage that is
  ! eventually reported in ad_memory_report.
  integer :: max_trace_count[*], max_index_count[*], max_const_count[*]

contains

  ! Initializes the main work variables of the reverse mode. One can
  ! specify either 'memory' or, for optimal use of memory, sweep_size,
  ! trace_size, and const_size separately. If all four are present,
  ! 'memory' takes precedence. With no arguments a call to
  ! init_ad_reverse allocates forward_values(DEFAULT_SWEEP_SIZE),
  ! adjoints(DEFAULT_SWEEP_SIZE), trace(4*DEFAULT_SWEEP_SIZE), and
  ! ad_constants(0.5*DEFAULT_SWEEP_SIZE). For a better understading of
  ! memory usage, call ad_memory_report() after the calculation. Sets
  ! the default mode to be the reverse mode.
  !
  ! All inputs optional
  !
  ! memory - allocates forward_values(x), adjoints(x), trace(4*x), and
  !          ad_constants(x/2), where x is such that the total memory
  !          reserved is equal to the amount specified by 'memory',
  !          whose format is 'integer unit', where 'unit' is 'B',
  !          'kB', 'MB', or 'GB' (also acceptable are 'b', 'kb', 'mb',
  !          or 'gb').
  ! sweep_size - size of forward_values and adjoints
  ! trace_size - size of trace
  ! const_size - size of ad_constants
  subroutine ad_init_reverse(memory, sweep_size, trace_size, const_size)
    character(*), intent(in), optional :: memory
    integer, intent(in), optional :: sweep_size, trace_size, const_size
    integer :: sweep_size_loc, trace_size_loc, const_size_loc
    real(kp) :: mem_size
    character(2) :: unit
    if (present(memory)) then
       read(memory, *, iostat=err_stat, iomsg=err_msg) mem_size, unit
       call check_err(__FILE__, __LINE__)
       if (unit == 'B') then
          sweep_size_loc = int(real(mem_size, kp)/(2.5*kp + 4*kind(1)))
       else if (unit == 'kB') then
          sweep_size_loc = int(real(mem_size, kp)/(2.5*kp + 4*kind(1))*1e3)
       else if (unit == 'MB') then
          sweep_size_loc = int(real(mem_size, kp)/(2.5*kp + 4*kind(1))*1e6)
       else if (unit == 'GB') then
          sweep_size_loc = int(real(mem_size, kp)/(2.5*kp + 4*kind(1))*1e9)
       else
          call error(__FILE__, __LINE__, 'Unrecognized unit: '//unit)
       end if
       trace_size_loc = 4*sweep_size_loc
       const_size_loc = sweep_size_loc/2
    else
       sweep_size_loc = DEFAULT_SWEEP_SIZE
       if (present(sweep_size)) sweep_size_loc = sweep_size
       ! Default is 4 entries per op in trace, i.e., most operations
       ! are expected to be binary.
       trace_size_loc = 4*DEFAULT_SWEEP_SIZE
       if (present(trace_size)) trace_size_loc = trace_size
       ! Default is to assume that 1/2 of all elemental operations
       ! produce a constant to be saved for the return sweep.
       const_size_loc = DEFAULT_SWEEP_SIZE/2
       if (present(const_size)) const_size_loc = const_size
    end if
    allocate(forward_values(sweep_size_loc), adjoints(sweep_size_loc), &
         & trace(trace_size_loc), ad_constants(const_size_loc), &
         & stat=err_stat, errmsg=err_msg)
    call check_err(__FILE__, __LINE__)
    trace_count = 0; max_trace_count = 0; adjoints = 0.0_kp
    index_count = 0; max_index_count = 0; trace = 0
    const_count = 0; max_const_count = 0; reverse_mode = .true.
  end subroutine ad_init_reverse

  elemental logical function advar_gt_advar(x1, x2) result(y)
    type(advar), intent(in) :: x1, x2
    y = x1%val > x2%val
  end function advar_gt_advar

  elemental logical function advar_gt_real32(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real32), intent(in) :: x2
    y = x1%val > x2
  end function advar_gt_real32

  elemental logical function real32_gt_advar(x1, x2) result(y)
    real(real32), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = x1 > x2%val
  end function real32_gt_advar

  elemental logical function advar_gt_dp(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(dp), intent(in) :: x2
    y = x1%val > x2
  end function advar_gt_dp

  elemental logical function dp_gt_advar(x1, x2) result(y)
    real(dp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = x1 > x2%val
  end function dp_gt_advar

  elemental logical function advar_gt_qp(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(qp), intent(in) :: x2
    y = x1%val > x2
  end function advar_gt_qp

  elemental logical function qp_gt_advar(x1, x2) result(y)
    real(qp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = x1 > x2%val
  end function qp_gt_advar

  elemental logical function advar_lt_advar(x1, x2) result(y)
    type(advar), intent(in) :: x1, x2
    y = x1%val < x2%val
  end function advar_lt_advar

  elemental logical function advar_lt_real32(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real32), intent(in) :: x2
    y = x1%val < x2
  end function advar_lt_real32

  elemental logical function real32_lt_advar(x1, x2) result(y)
    real(real32), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = x1 < x2%val
  end function real32_lt_advar

  elemental logical function advar_lt_dp(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(dp), intent(in) :: x2
    y = x1%val < x2
  end function advar_lt_dp

  elemental logical function dp_lt_advar(x1, x2) result(y)
    real(dp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = x1 < x2%val
  end function dp_lt_advar

  elemental logical function advar_lt_qp(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(qp), intent(in) :: x2
    y = x1%val < x2
  end function advar_lt_qp

  elemental logical function qp_lt_advar(x1, x2) result(y)
    real(qp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = x1 < x2%val
  end function qp_lt_advar

  ! Conversion from real to type(advar) copies the real value to the
  ! val field; the result is a passive variable
  ! (index=d=dd=0). Conversion from type(advar) to real copies the val
  ! field.
  impure elemental subroutine assign_advar_integer(this, x)
    class(advar), intent(out) :: this
    integer, intent(in) :: x
    this%val = real(x, kp)
  end subroutine assign_advar_integer

  impure elemental subroutine assign_advar_real32(this, x)
    class(advar), intent(out) :: this
    real(real32), intent(in) :: x
    this%val = real(x, kp)
  end subroutine assign_advar_real32

  impure elemental subroutine assign_advar_dp(this, x)
    class(advar), intent(out) :: this
    real(dp), intent(in) :: x
    this%val = real(x, kp)
  end subroutine assign_advar_dp

  impure elemental subroutine assign_advar_qp(this, x)
    class(advar), intent(out) :: this
    real(qp), intent(in) :: x
    this%val = real(x, kp)
  end subroutine assign_advar_qp

  elemental subroutine assign_integer_advar(x, this)
    integer, intent(out) :: x
    class(advar), intent(in) :: this
    x = int(this%val, real32)
  end subroutine assign_integer_advar

  elemental subroutine assign_real32_advar(x, this)
    real(real32), intent(out) :: x
    class(advar), intent(in) :: this
    x = real(this%val, real32)
  end subroutine assign_real32_advar

  elemental subroutine assign_dp_advar(x, this)
    real(dp), intent(out) :: x
    class(advar), intent(in) :: this
    x = real(this%val, dp)
  end subroutine assign_dp_advar

  elemental subroutine assign_qp_advar(x, this)
    real(qp), intent(out) :: x
    class(advar), intent(in) :: this
    x = real(this%val, qp)
  end subroutine assign_qp_advar

  ! AD ELEMENTAL OPERATIONS BEGIN

  ! In the reverse mode, each elemental operation appends to trace,
  ! forward_values, and (optionally) ad_constants. In the forward
  ! mode, the derivatives are calculated on the fly.
  type(advar) function add_advar_advar(x1, x2) result(y)
    type(advar), intent(in) :: x1, x2
    if (x1%index /= 0 .and. x2%index /= 0) then
       y%val = x1%val + x2%val
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x1%index
          trace(trace_count+2) = x2%index
          trace(trace_count+3) = y%index
          trace(trace_count+4) = ADD_A_A
          trace_count = trace_count + 4
       else
          y%d = x1%d + x2%d
          y%dd = x1%dd + x2%dd
          y%index = 1
       end if
    else if(x1%index /= 0) then
       y = add_advar_real(x1, x2%val)
    else if(x2%index /= 0) then
       y = add_real_advar(x1%val, x2)
    else
       y = x1%val + x2%val
    end if
  end function add_advar_advar

  type(advar) function add_advar_real(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(kp), intent(in) :: x2
    y%val = x1%val + x2
    if (x1%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x1%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = ADD_SUBTRACT_A_R
          trace_count = trace_count + 3
       else
          y%d = x1%d
          y%dd = x1%dd
          y%index = 1
       end if
    end if
  end function add_advar_real

  type(advar) function add_advar_real32(x1,x2) result(y)
    type(advar), intent(in) :: x1
    real(real32), intent(in) :: x2
    y = add_advar_real(x1, real(x2, kp))
  end function add_advar_real32

  type(advar) function add_advar_dp(x1,x2) result(y)
    type(advar), intent(in) :: x1
    real(dp), intent(in) :: x2
    y = add_advar_real(x1, real(x2, kp))
  end function add_advar_dp

  type(advar) function add_advar_qp(x1,x2) result(y)
    type(advar), intent(in) :: x1
    real(qp), intent(in) :: x2
    y = add_advar_real(x1, real(x2, kp))
  end function add_advar_qp

  type(advar) function add_advar_integer(x1,x2) result(y)
    type(advar), intent(in) :: x1
    integer, intent(in) :: x2
    y = add_advar_real(x1, real(x2, kp))
  end function add_advar_integer

  type(advar) function add_real_advar(x1, x2) result(y)
    real(kp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y%val = x1 + x2%val
    if (x2%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x2%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = ADD_SUBTRACT_A_R
          trace_count = trace_count + 3
       else
          y%d = x2%d
          y%dd = x2%dd
          y%index = 1
       end if
    end if
  end function add_real_advar

  type(advar) function add_real32_advar(x1, x2) result(y)
    real(real32), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = add_real_advar(real(x1, kp), x2)
  end function add_real32_advar

  type(advar) function add_dp_advar(x1, x2) result(y)
    real(dp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = add_real_advar(real(x1, kp), x2)
  end function add_dp_advar

  type(advar) function add_qp_advar(x1, x2) result(y)
    real(qp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = add_real_advar(real(x1, kp), x2)
  end function add_qp_advar

  type(advar) function add_integer_advar(x1, x2) result(y)
    integer, intent(in) :: x1
    type(advar), intent(in) :: x2
    y = add_real_advar(real(x1, kp), x2)
  end function add_integer_advar

  type(advar) function subtract_advar_advar(x1, x2) result(y)
    type(advar), intent(in) :: x1, x2
    if (x1%index /= 0 .and. x2%index /= 0) then
       y%val = x1%val - x2%val
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x1%index
          trace(trace_count+2) = x2%index
          trace(trace_count+3) = y%index
          trace(trace_count+4) = SUBTRACT_A_A
          trace_count = trace_count + 4
       else
          y%d = x1%d - x2%d
          y%dd = x1%dd - x2%dd
          y%index = 1
       end if
    else if (x1%index /= 0) then
       y = subtract_advar_real(x1, x2%val)
    else if (x2%index /= 0) then
       y = subtract_real_advar(x1%val, x2)
    else
       y = x1%val - x2%val
    end if
  end function subtract_advar_advar

  type(advar) function subtract_advar(x) result(y)
    type(advar), intent(in) :: x
    y = subtract_real_advar(0.0_kp, x)
  end function subtract_advar

  type(advar) function subtract_advar_real(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(kp), intent(in) :: x2
    y%val = x1%val - x2
    if (x1%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x1%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = ADD_SUBTRACT_A_R
          trace_count = trace_count + 3
       else
          y%d = x1%d
          y%dd = x1%dd
          y%index = 1
       end if
    end if
  end function subtract_advar_real

  type(advar) function subtract_advar_real32(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real32), intent(in) :: x2
    y = subtract_advar_real(x1, real(x2, kp))
  end function subtract_advar_real32

  type(advar) function subtract_advar_dp(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(dp), intent(in) :: x2
    y = subtract_advar_real(x1, real(x2, kp))
  end function subtract_advar_dp

  type(advar) function subtract_advar_qp(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(qp), intent(in) :: x2
    y = subtract_advar_real(x1, real(x2, kp))
  end function subtract_advar_qp

  type(advar) function subtract_advar_integer(x1, x2) result(y)
    type(advar), intent(in) :: x1
    integer, intent(in) :: x2
    y = subtract_advar_real(x1, real(x2, kp))
  end function subtract_advar_integer

  type(advar) function subtract_real_advar(x1, x2) result(y)
    real(kp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y%val = x1 - x2%val
    if (x2%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x2%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = SUBTRACT_R_A
          trace_count = trace_count + 3
       else
          y%d = -x2%d
          y%dd = -x2%dd
          y%index = 1
       end if
    end if
  end function subtract_real_advar

  type(advar) function subtract_real32_advar(x1, x2) result(y)
    real(real32), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = subtract_real_advar(real(x1, kp), x2)
  end function subtract_real32_advar

  type(advar) function subtract_dp_advar(x1, x2) result(y)
    real(dp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = subtract_real_advar(real(x1, kp), x2)
  end function subtract_dp_advar

  type(advar) function subtract_qp_advar(x1, x2) result(y)
    real(qp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = subtract_real_advar(real(x1, kp), x2)
  end function subtract_qp_advar

  type(advar) function subtract_integer_advar(x1, x2) result(y)
    integer, intent(in) :: x1
    type(advar), intent(in) :: x2
    y = subtract_real_advar(real(x1, kp), x2)
  end function subtract_integer_advar

  type(advar) function multiply_advar_advar(x1, x2) result(y)
    type(advar), intent(in) :: x1, x2
    if (x1%index /= 0 .and. x2%index /= 0) then
       y%val = x1%val*x2%val
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x1%index
          trace(trace_count+2) = x2%index
          trace(trace_count+3) = y%index
          trace(trace_count+4) = MULTIPLY_A_A
          trace_count = trace_count + 4
       else
          y%d = x1%val*x2%d + x1%d*x2%val
          y%dd = x1%val*x2%dd + 2*x1%d*x2%d + x1%dd*x2%val
          y%index = 1
       end if
    else if (x1%index /= 0) then
       y = multiply_advar_real(x1, x2%val)
    else if (x2%index /= 0) then
       y = multiply_real_advar(x1%val, x2)
    else
       y = x1%val*x2%val
    end if
  end function multiply_advar_advar

  type(advar) function multiply_advar_real(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(kp), intent(in) :: x2
    y%val = x1%val*x2
    if (x1%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          const_count = const_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          ad_constants(const_count) = x2
          trace(trace_count+1) = x1%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = MULTIPLY_DIVIDE_A_R
          trace_count = trace_count + 3
       else
          y%d = x1%d*x2
          y%dd = x1%dd*x2
          y%index = 1
       end if
    end if
  end function multiply_advar_real

  type(advar) function multiply_advar_real32(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real32), intent(in) :: x2
    y = multiply_advar_real(x1, real(x2, kp))
  end function multiply_advar_real32

  type(advar) function multiply_advar_dp(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(dp), intent(in) :: x2
    y = multiply_advar_real(x1, real(x2, kp))
  end function multiply_advar_dp

  type(advar) function multiply_advar_qp(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(qp), intent(in) :: x2
    y = multiply_advar_real(x1, real(x2, kp))
  end function multiply_advar_qp

  type(advar) function multiply_advar_integer(x1, x2) result(y)
    type(advar), intent(in) :: x1
    integer, intent(in) :: x2
    y = multiply_advar_real(x1, real(x2, kp))
  end function multiply_advar_integer

  type(advar) function multiply_real_advar(x1, x2) result(y)
    real(kp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y%val = x1*x2%val
    if (x2%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          const_count = const_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          ad_constants(const_count) = x1
          trace(trace_count+1) = x2%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = MULTIPLY_DIVIDE_A_R
          trace_count = trace_count + 3
       else
          y%d = x1*x2%d
          y%dd = x1*x2%dd
          y%index = 1
       end if
    end if
  end function multiply_real_advar

  type(advar) function multiply_real32_advar(x1, x2) result(y)
    real(real32), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = multiply_real_advar(real(x1, kp), x2)
  end function multiply_real32_advar

  type(advar) function multiply_dp_advar(x1, x2) result(y)
    real(dp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = multiply_real_advar(real(x1, kp), x2)
  end function multiply_dp_advar

  type(advar) function multiply_qp_advar(x1, x2) result(y)
    real(qp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = multiply_real_advar(real(x1, kp), x2)
  end function multiply_qp_advar

  type(advar) function multiply_integer_advar(x1, x2) result(y)
    integer, intent(in) :: x1
    type(advar), intent(in) :: x2
    y = multiply_real_advar(real(x1, kp), x2)
  end function multiply_integer_advar

  type(advar) function divide_advar_advar(x1, x2) result(y)
    type(advar), intent(in) :: x1, x2
    real(kp) :: inv_x2
    if (x1%index /= 0 .and. x2%index /= 0) then
       inv_x2 = 1/x2%val
       y%val = x1%val*inv_x2
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x1%index
          trace(trace_count+2) = x2%index
          trace(trace_count+3) = y%index
          trace(trace_count+4) = DIVIDE_A_A
          trace_count = trace_count + 4
       else
          y%d = (x1%d - y%val*x2%d)*inv_x2
          y%dd = (x1%dd - y%val*x2%dd - 2*y%d*x2%d)*inv_x2
          y%index = 1
       end if
    else if (x1%index /= 0) then
       y = divide_advar_real(x1, x2%val)
    else if (x2%index /= 0) then
       y = divide_real_advar(x1%val, x2)
    else
       y = x1%val/x2%val
    end if
  end function divide_advar_advar

  type(advar) function divide_advar_real(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(kp), intent(in) :: x2
    real(kp) :: inv_x2
    inv_x2 = 1/x2
    y%val = x1%val*inv_x2
    if (x1%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          const_count = const_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          ad_constants(const_count) = inv_x2
          trace(trace_count+1) = x1%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = MULTIPLY_DIVIDE_A_R
          trace_count = trace_count + 3
       else
          y%d = x1%d*inv_x2
          y%dd = x1%dd*inv_x2
          y%index = 1
       end if
    end if
  end function divide_advar_real

  type(advar) function divide_advar_real32(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real32), intent(in) :: x2
    y = divide_advar_real(x1, real(x2, kp))
  end function divide_advar_real32

  type(advar) function divide_advar_dp(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(dp), intent(in) :: x2
    y = divide_advar_real(x1, real(x2, kp))
  end function divide_advar_dp

  type(advar) function divide_advar_qp(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(qp), intent(in) :: x2
    y = divide_advar_real(x1, real(x2, kp))
  end function divide_advar_qp

  type(advar) function divide_advar_integer(x1, x2) result(y)
    type(advar), intent(in) :: x1
    integer, intent(in) :: x2
    y = divide_advar_real(x1, real(x2, kp))
  end function divide_advar_integer

  type(advar) function divide_real_advar(x1, x2) result(y)
    real(kp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y%val = x1/x2%val
    if (x2%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          const_count = const_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          ad_constants(const_count) = x1
          trace(trace_count+1) = x2%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = DIVIDE_R_A
          trace_count = trace_count + 3
       else
          y%d = -y%val*x2%d/x2%val
          y%dd = (-y%val*x2%dd - 2*y%d*x2%d)/x2%val
          y%index = 1
       end if
    end if
  end function divide_real_advar

  type(advar) function divide_real32_advar(x1, x2) result(y)
    real(real32), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = divide_real_advar(real(x1, kp), x2)
  end function divide_real32_advar

  type(advar) function divide_dp_advar(x1, x2) result(y)
    real(dp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = divide_real_advar(real(x1, kp), x2)
  end function divide_dp_advar

  type(advar) function divide_qp_advar(x1, x2) result(y)
    real(qp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = divide_real_advar(real(x1, kp), x2)
  end function divide_qp_advar

  type(advar) function divide_integer_advar(x1, x2) result(y)
    integer, intent(in) :: x1
    type(advar), intent(in) :: x2
    y = divide_real_advar(real(x1, kp), x2)
  end function divide_integer_advar

  type(advar) function abs_advar(x) result(y)
    type(advar), intent(in) :: x
    y%val = abs(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = ABS_A
          trace_count = trace_count + 3
       else
          y%d = x%d*sign(1.0_kp, x%val)
          y%dd = x%dd*sign(1.0_kp, x%val)
          y%index = 1
       end if
    end if
  end function abs_advar

  type(advar) function power_advar_advar(x1, x2) result(y)
    type(advar), intent(in) :: x1, x2
    real(kp) :: inv_x1, log_value
    if (x1%index /= 0 .and. x2%index /= 0) then
       y%val = x1%val**x2%val
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x1%index
          trace(trace_count+2) = x2%index
          trace(trace_count+3) = y%index
          trace(trace_count+4) = POWER_A_A
          trace_count = trace_count + 4
       else
          log_value = log(x1%val)
          y%d = y%val*x2%d*log_value + x1%d*x2%val*x1%val**(x2%val-1)
          inv_x1 = 1/x1%val
          y%dd = y%d**2/y%val + y%val*(x2%dd*log_value + &
               & (2*x2%d*x1%d + x2%val*(x1%dd - x1%d**2*inv_x1))*inv_x1)
          y%index = 1
       end if
    else if (x1%index /= 0) then
       y = power_advar_real(x1, x2%val)
    else if (x2%index /= 0) then
       y = power_real_advar(x1%val, x2)
    else
       y = x1%val**x2%val
    end if
  end function power_advar_advar

  type(advar) function power_advar_real(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(kp), intent(in) :: x2
    real(kp) :: inv_x1
    y%val = x1%val**x2
    if (x1%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          const_count = const_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          ad_constants(const_count) = x2
          trace(trace_count+1) = x1%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = POWER_A_R
          trace_count = trace_count + 3
       else
          y%d = x1%d*x2*x1%val**(x2-1)
          inv_x1 = 1/x1%val
          y%dd = y%d**2/y%val + y%val*x2*(x1%dd - x1%d**2*inv_x1)*inv_x1
          y%index = 1
       end if
    end if
  end function power_advar_real

  type(advar) function power_advar_real32(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real32), intent(in) :: x2
    y = power_advar_real(x1, real(x2, kp))
  end function power_advar_real32

  type(advar) function power_advar_dp(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(dp), intent(in) :: x2
    y = power_advar_real(x1, real(x2, kp))
  end function power_advar_dp

  type(advar) function power_advar_qp(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(qp), intent(in) :: x2
    y = power_advar_real(x1, real(x2, kp))
  end function power_advar_qp

  type(advar) function power_advar_integer(x1, x2) result(y)
    type(advar), intent(in) :: x1
    integer, intent(in) :: x2
    real(kp) :: inv_x1
    y%val = x1%val**x2
    if (x1%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x1%index
          ! As opposed to real constants which are put into
          ! ad_constants, here we have an integer constant which, as
          ! an exception, is put into trace without introducing a new
          ! array for integer constants.
          trace(trace_count+2) = x2
          trace(trace_count+3) = y%index
          trace(trace_count+4) = POWER_INTEGER
          trace_count = trace_count + 4
       else
          inv_x1 = 1/x1%val
          y%d = y%val*x2*x1%d*inv_x1
          y%dd = y%d**2/y%val + y%val*x2*(x1%dd - x1%d**2*inv_x1)*inv_x1
          y%index = 1
       end if
    end if
  end function power_advar_integer

  type(advar) function power_real_advar(x1, x2) result(y)
    real(kp), intent(in) :: x1
    type(advar), intent(in) :: x2
    real(kp) :: log_value
    y%val = x1**x2%val
    if (x2%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          const_count = const_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          ad_constants(const_count) = x1
          trace(trace_count+1) = x2%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = POWER_R_A
          trace_count = trace_count + 3
       else
          log_value = log(x1)
          y%d = y%val*x2%d*log_value
          y%dd = y%d**2/y%val + y%val*x2%dd*log_value
          y%index = 1
       end if
    end if
  end function power_real_advar

  type(advar) function power_real32_advar(x1, x2) result(y)
    real(real32), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = power_real_advar(real(x1, kp), x2)
  end function power_real32_advar

  type(advar) function power_dp_advar(x1, x2) result(y)
    real(dp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = power_real_advar(real(x1, kp), x2)
  end function power_dp_advar

  type(advar) function power_qp_advar(x1, x2) result(y)
    real(qp), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = power_real_advar(real(x1, kp), x2)
  end function power_qp_advar

  type(advar) function power_integer_advar(x1, x2) result(y)
    integer, intent(in) :: x1
    type(advar), intent(in) :: x2
    y = power_real_advar(real(x1, kp), x2)
  end function power_integer_advar

  type(advar) function exp_advar(x) result(y)
    type(advar), intent(in) :: x
    y%val = exp(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = EXP_A
          trace_count = trace_count + 3
       else
          y%d = x%d*y%val
          y%dd = x%dd*y%val + x%d*y%d
          y%index = 1
       end if
    end if
  end function exp_advar

  type(advar) function sqrt_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: inv_y
    y%val = sqrt(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = SQRT_A
          trace_count = trace_count + 3
       else
          inv_y = 1/y%val
          y%d = x%d/2*inv_y
          y%dd = (x%dd*inv_y - y%d*x%d/x%val)/2
          y%index = 1
       end if
    end if
  end function sqrt_advar

  type(advar) function log_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: inv_x
    y%val = log(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = LOG_A
          trace_count = trace_count + 3
       else
          inv_x = 1/x%val
          y%d = x%d*inv_x
          y%dd = (x%dd - x%d*y%d)*inv_x
          y%index = 1
       end if
    end if
  end function log_advar

  type(advar) function sin_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: cos_value
    y%val = sin(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = SIN_A
          trace_count = trace_count + 3
       else
          cos_value = cos(x%val)
          y%d = x%d*cos_value
          y%dd = x%dd*cos_value - x%d**2*y%val
          y%index = 1
       end if
    end if
  end function sin_advar

  type(advar) function cos_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: sin_value
    y%val = cos(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = COS_A
          trace_count = trace_count + 3
       else
          sin_value = -sin(x%val)
          y%d = x%d*sin_value
          y%dd = x%dd*sin_value -x%d**2*y%val
          y%index = 1
       end if
    end if
  end function cos_advar

  type(advar) function tan_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: inv_cos_value2
    y%val = tan(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = TAN_A
          trace_count = trace_count + 3
       else
          inv_cos_value2 = 1/cos(x%val)
          inv_cos_value2 = inv_cos_value2*inv_cos_value2
          y%d = x%d*inv_cos_value2
          y%dd = x%dd*inv_cos_value2 + 2*x%d*y%val*y%d
          y%index = 1
       end if
    end if
  end function tan_advar

  type(advar) function asin_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: tmp
    y%val = asin(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = ASIN_A
          trace_count = trace_count + 3
       else
          tmp = 1/sqrt(1-x%val**2)
          y%d = x%d*tmp
          y%dd = tmp*(x%dd + x%val*y%d**2)
          y%index = 1
       end if
    end if
  end function asin_advar

  type(advar) function acos_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: tmp
    y%val = acos(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = ACOS_A
          trace_count = trace_count + 3
       else
          tmp = -1/sqrt(1-x%val**2)
          y%d = x%d*tmp
          y%dd = tmp*(x%dd + x%val*y%d**2)
          y%index = 1
       end if
    end if
  end function acos_advar

  type(advar) function atan_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: tmp
    y%val = atan(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = ATAN_A
          trace_count = trace_count + 3
       else
          tmp = 1/(1+x%val**2)
          y%d = x%d*tmp
          y%dd = x%dd*tmp - 2*x%val*y%d**2
          y%index = 1
       end if
    end if
  end function atan_advar

  type(advar) function sinh_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: cosh_value
    y%val = sinh(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = SINH_A
          trace_count = trace_count + 3
       else
          cosh_value = cosh(x%val)
          y%d = x%d*cosh_value
          y%dd = x%dd*cosh_value + y%val*x%d**2
          y%index = 1
       end if
    end if
  end function sinh_advar

  type(advar) function cosh_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: sinh_value
    y%val = cosh(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = COSH_A
          trace_count = trace_count + 3
       else
          sinh_value = sinh(x%val)
          y%d = x%d*sinh_value
          y%dd = x%dd*sinh_value + y%val*x%d**2
          y%index = 1
       end if
    end if
  end function cosh_advar

  type(advar) function tanh_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: inv_cosh_value2
    y%val = tanh(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = TANH_A
          trace_count = trace_count + 3
       else
          inv_cosh_value2 = 1/cosh(x%val)**2
          y%d = x%d*inv_cosh_value2
          y%dd = x%dd*inv_cosh_value2 - 2*y%val*x%d*y%d
          y%index = 1
       end if
    end if
  end function tanh_advar

  type(advar) function asinh_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: tmp
    y%val = asinh(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = ASINH_A
          trace_count = trace_count + 3
       else
          tmp = 1/sqrt(1+x%val**2)
          y%d = x%d*tmp
          y%dd = tmp*(x%dd - x%val*y%d**2)
          y%index = 1
       end if
    end if
  end function asinh_advar

  type(advar) function acosh_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: tmp
    y%val = acosh(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = ACOSH_A
          trace_count = trace_count + 3
       else
          tmp = 1/sqrt(x%val**2-1)
          y%d = x%d*tmp
          y%dd = tmp*(x%dd - x%val*y%d**2)
          y%index = 1
       end if
    end if
  end function acosh_advar

  type(advar) function atanh_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: tmp
    y%val = atanh(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = ATANH_A
          trace_count = trace_count + 3
       else
          tmp = 1/(1-x%val**2)
          y%d = x%d*tmp
          y%dd = (x%dd + 2*x%val*x%d*y%d)*tmp
          y%index = 1
       end if
    end if
  end function atanh_advar

  type(advar) function erf_advar(x) result(y)
    type(advar), intent(in) :: x
    real(kp) :: tmp
    y%val = erf(x%val)
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = ERF_A
          trace_count = trace_count + 3
       else
          tmp = 2/sqrtpi*exp(-x%val**2)
          y%d = x%d*tmp
          y%dd = (x%dd - 2*x%d*x%d*x%val)*tmp
          y%index = 1
       end if
    end if
  end function erf_advar

#ifdef USE_GSL
  type(advar) function Li2_advar(x) result(y)
    type(advar), intent(in) :: x
    y%val = Li2(real(x%val, dp))
    if (x%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          trace(trace_count+1) = x%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = LI2_A
          trace_count = trace_count + 3
       else
          y%d = -x%d*log(abs(1-x%val))/x%val
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = x%dd*y%d/x%d - x%d*(y%d + x%d/(x%val-1))/x%val
          else
             y%dd = -x%dd*log(abs(1-x%val)/x%val)
          end if
          y%index = 1
       end if
    end if
  end function Li2_advar
#endif
  ! For numerical integration see numerical_integration.f90

  ! AD ELEMENTAL OPERATIONS END

  ! The return sweep of the reverse mode. Assumes that the forward
  ! sweep has been performed with function evaluation. The adjoints
  ! array is reinitialized to zero each time before the sweep. Note
  ! that after the sweep, forward_values need not be reinitialized
  ! since it can simply be overwritten.
  !
  ! num_parameters - Number of active parameters. After the return
  !                  sweep, index_count will be reset to this
  !                  number. If there are, e.g., 2 active parameters,
  !                  then during the next forward sweep the new
  !                  indices should start at 3.
  subroutine ad_grad(num_parameters)
    integer, intent(in) :: num_parameters
    integer :: op_id
    if (num_parameters == 0) &
         & call error(__FILE__, __LINE__, 'No active parameters')
    if (index_count > size(forward_values) .or. trace_count > size(trace) .or. &
         & const_count > size(ad_constants)) &
         & call error(__FILE__, __LINE__, &
         & 'Too many elemental operations. Request more memory.')
    ! For memory report
    max_trace_count = max(max_trace_count, trace_count)
    max_index_count = max(max_index_count, index_count)
    max_const_count = max(max_const_count, const_count)
    adjoints(:index_count-1) = 0.0_kp
    adjoints(index_count) = 1.0_kp ! df/d(v[index_count]) = df/df = 1
    do while(trace_count > 0)
       associate(i => trace_count)
         op_id = trace(i)
         select case(op_id)
            ! Elementary
         case(ADD_A_A)
            adjoints(trace(i-3)) = adjoints(trace(i-3)) + adjoints(trace(i-1))
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + adjoints(trace(i-1))
            i = i - 4
         case(ADD_SUBTRACT_A_R)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + adjoints(trace(i-1))
            i = i - 3
         case(SUBTRACT_A_A)
            adjoints(trace(i-3)) = adjoints(trace(i-3)) + adjoints(trace(i-1))
            adjoints(trace(i-2)) = adjoints(trace(i-2)) - adjoints(trace(i-1))
            i = i - 4
         case(SUBTRACT_R_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) - adjoints(trace(i-1))
            i = i - 3
         case(MULTIPLY_A_A)
            adjoints(trace(i-3)) = adjoints(trace(i-3)) + &
                 & adjoints(trace(i-1))*forward_values(trace(i-2))
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))*forward_values(trace(i-3))
            i = i - 4
         case(MULTIPLY_DIVIDE_A_R)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))*ad_constants(const_count)
            const_count = const_count - 1
            i = i - 3
         case(DIVIDE_A_A)
            adjoints(trace(i-3)) = adjoints(trace(i-3)) + &
                 & adjoints(trace(i-1))/forward_values(trace(i-2))
            adjoints(trace(i-2)) = adjoints(trace(i-2)) - &
                 & adjoints(trace(i-1))*forward_values(trace(i-1))/ &
                 & forward_values(trace(i-2))
            i = i - 4
         case(DIVIDE_R_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) - &
                 & adjoints(trace(i-1))*ad_constants(const_count)/ &
                 & forward_values(trace(i-2))/forward_values(trace(i-2))
            const_count = const_count - 1
            i = i - 3
         case(POWER_A_A)
            adjoints(trace(i-3)) = adjoints(trace(i-3)) + &
                 & adjoints(trace(i-1))*forward_values(trace(i-2))* &
                 & forward_values(trace(i-3))**(forward_values(trace(i-2))-1)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))*log(forward_values(trace(i-3)))* &
                 & forward_values(trace(i-3))**forward_values(trace(i-2))
            i = i - 4
         case(POWER_A_R)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))*ad_constants(const_count)* &
                 & forward_values(trace(i-2))**(ad_constants(const_count)-1)
            const_count = const_count - 1
            i = i - 3
         case(POWER_R_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))*log(ad_constants(const_count))* &
                 & ad_constants(const_count)**forward_values(trace(i-2))
            const_count = const_count - 1
            i = i - 3
         case(POWER_INTEGER)
            adjoints(trace(i-3)) = adjoints(trace(i-3)) + &
                 & adjoints(trace(i-1))*trace(i-2)* &
                 & forward_values(trace(i-3))**(trace(i-2)-1)
            i = i - 4
         case(ABS_A)
            if (forward_values(trace(i-2)) < 0) then
               adjoints(trace(i-2)) = adjoints(trace(i-2)) - &
                    & adjoints(trace(i-1))
            else
               adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                    & adjoints(trace(i-1))
            end if
            i = i - 3
         case(EXP_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))*forward_values(trace(i-1))
            i = i - 3
         case(SQRT_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))/2.0_kp/forward_values(trace(i-1))
            i = i - 3
         case(LOG_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))/forward_values(trace(i-2))
            i = i - 3
            ! Trigonometric
         case(SIN_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))*cos(forward_values(trace(i-2)))
            i = i - 3
         case(COS_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) - &
                 & adjoints(trace(i-1))*sin(forward_values(trace(i-2)))
            i = i - 3
         case(TAN_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))/cos(forward_values(trace(i-2)))**2
            i = i - 3
         case(ASIN_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))/sqrt(1-forward_values(trace(i-2))**2)
            i = i - 3
         case(ACOS_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) - &
                 & adjoints(trace(i-1))/sqrt(1-forward_values(trace(i-2))**2)
            i = i - 3
         case(ATAN_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))/(1+forward_values(trace(i-2))**2)
            i = i - 3
            ! Hyperbolic
         case(SINH_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))*cosh(forward_values(trace(i-2)))
            i = i - 3
         case(COSH_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))*sinh(forward_values(trace(i-2)))
            i = i - 3
         case(TANH_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))/cosh(forward_values(trace(i-2)))**2
            i = i - 3
         case(ASINH_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))/sqrt(forward_values(trace(i-2))**2+1)
            i = i - 3
         case(ACOSH_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))/sqrt(forward_values(trace(i-2))**2-1)
            i = i - 3
         case(ATANH_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))/(1-(forward_values(trace(i-2)))**2)
            i = i - 3
            ! Special
         case(ERF_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))*2.0_kp/sqrtpi* &
                 & exp(-forward_values(trace(i-2))**2)
            i = i - 3
#ifdef USE_GSL
         case(LI2_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) - &
                 & adjoints(trace(i-1))* &
                 & log(abs(1-forward_values(trace(i-2))))/ &
                 & forward_values(trace(i-2))
            i = i - 3
#endif
            ! Numerical integration
         case(INT_BOTH_BOUNDS)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))*ad_constants(const_count)
            const_count = const_count - 1
            adjoints(trace(i-3)) = adjoints(trace(i-3)) - &
                 & adjoints(trace(i-1))*ad_constants(const_count)
            const_count = const_count - 1
            i = i - 4
         case(INT_LOWER_BOUND)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) - &
                 & adjoints(trace(i-1))*ad_constants(const_count)
            const_count = const_count - 1
            i = i - 3
         case(INT_UPPER_BOUND)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))*ad_constants(const_count)
            const_count = const_count - 1
            i = i - 3
         end select
       end associate
    end do
    index_count = num_parameters
  end subroutine ad_grad

  ! Prints the size of adjoints, trace, and const in terms of both the
  ! number of elements and the amount of memory used. The latter is
  ! different for double and quad precision. The sizes, which
  ! represent the maxima over all images, are shown as requested
  ! during initialization and how much was actually used.
  subroutine ad_memory_report(io_unit)
    use, intrinsic :: iso_fortran_env, only: output_unit
    integer, intent(in), optional :: io_unit
    integer :: io_unit_loc, max_trace, max_index, max_const
    integer :: i
    if (this_image() /= 1) return
    io_unit_loc = output_unit
    if (present(io_unit)) io_unit_loc = io_unit
    max_trace = 0; max_index = 0; max_const = 0
    do i = 1, num_images()
       max_trace = max(max_trace, max_trace_count[i])
       max_index = max(max_index, max_index_count[i])
       max_const = max(max_const, max_const_count[i])
    end do
    write(io_unit_loc, '(1x, g0)') 'AD memory usage'
    write(io_unit_loc, '(1x, g0)') '==============='
    write(io_unit_loc, '(2x, g0)') 'Requested:'
    write(io_unit_loc, '(2x, g0)', advance='no') 'forward+adjoints: '
    call print_memory(io_unit, 2*kp*size(adjoints))
    write(io_unit_loc, '(*(g0))') ' (2x', size(adjoints), ')'
    write(io_unit_loc, '(13x, g0)', advance='no') 'trace: '
    call print_memory(io_unit_loc, kind(trace(1))*size(trace))
    write(io_unit_loc, '(1x, *(g0))') '(', size(trace), ')'
    write(io_unit_loc, '(13x, g0)', advance='no') 'const: '
    call print_memory(io_unit_loc, kp*size(ad_constants))
    write(io_unit_loc, '(1x, *(g0))') '(', size(ad_constants), ')'
    write(io_unit_loc, '(13x, g0)', advance='no') 'Total: '
    call print_memory(io_unit_loc, kp*(2*size(adjoints)+size(ad_constants)) + &
         & kind(1)*size(trace))
    write(io_unit_loc, '(/, 2x, g0)') 'Used:'
    write(io_unit_loc, '(2x, g0)', advance='no') 'forward+adjoints: '
    call print_memory(io_unit_loc, 2*kp*max_index)
    write(io_unit_loc, '(1x, *(g0))') '(2x', max_index, ')'
    write(io_unit_loc, '(13x, g0)', advance='no') 'trace: '
    call print_memory(io_unit_loc, kind(trace(1))*max_trace)
    write(io_unit_loc, '(1x, *(g0))') '(', max_trace, ')'
    write(io_unit_loc, '(13x, g0)', advance='no') 'const: '
    call print_memory(io_unit_loc, kp*max_const)
    write(io_unit_loc, '(1x, *(g0))') '(', max_const, ')'
    write(io_unit_loc, '(13x, g0)', advance='no') 'Total: '
    call print_memory(io_unit_loc, kp*(2*max_index+max_const) + &
         & kind(1)*max_trace)
    write(io_unit_loc, '(g0)')
  end subroutine ad_memory_report

  subroutine safe_deallocate_advar(file, line, array)
    ! 'file' and 'line' should be determined by the preprocessor.
    character(*), intent(in) :: file
    integer, intent(in) :: line
    type(advar), allocatable, intent(in out) :: array(:)
    if (allocated(array)) then
       deallocate(array, stat=err_stat, errmsg=err_msg)
       call check_err(file, line)
    end if
  end subroutine safe_deallocate_advar

  subroutine ad_close()
    call safe_deallocate(__FILE__, __LINE__, forward_values)
    call safe_deallocate(__FILE__, __LINE__, adjoints)
    call safe_deallocate(__FILE__, __LINE__, trace)
    call safe_deallocate(__FILE__, __LINE__, ad_constants)
    reverse_mode = .false.
  end subroutine ad_close
end module ad

!!****m* GADfit/automatic_differentiation
!! 
!! COPYRIGHT
!! 
!! Copyright (C) 2014-2016 Raul Laasner
!! This file is distributed under the terms of the GNU General Public
!! License, see LICENSE in the root directory of the present
!! distribution or http://gnu.org/copyleft/gpl.txt .
!! 
!! FUNCTION
!!
!! Procedures for finding the gradient of a function using the reverse
!! mode of automatic differentiation (AD), and procedures for finding
!! the first and second directional derivatives using the forward
!! mode. This includes the implementation of all elemental operations,
!! except numerical integration, which is given in
!! numerical_integration.f90. Also provides the definition of
!! type(advar) (the AD variable), which is the central variable used
!! in GADfit. See the user guide for details about the AD algorithm.
!! 
!! SOURCE
#include <config.h>

module ad

  use, intrinsic :: iso_c_binding,   only: c_double
  use, intrinsic :: iso_fortran_env, only: real32, real64, real128
  use gadf_constants, only: kp, sqrtpi
  use messaging
  use misc,           only: safe_deallocate

  implicit none
  
  public
  private :: c_double, real32, real64, real128, kp, sqrtpi, &
       & DEFAULT_SWEEP_SIZE, max_trace_count, max_index_count, max_const_count

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
     procedure :: assign_advar_real64
     procedure, pass(this) :: assign_real64_advar
     procedure :: assign_advar_real128
     procedure, pass(this) :: assign_real128_advar
     generic :: assignment(=) => assign_advar_integer, assign_integer_advar, &
          & assign_advar_real32, assign_real32_advar, assign_advar_real64, &
          & assign_real64_advar, assign_advar_real128, assign_real128_advar
  end type advar

  interface operator(>)
     module procedure advar_gt_advar, advar_gt_real32, real32_gt_advar, &
          & advar_gt_real64, real64_gt_advar, advar_gt_real128, real128_gt_advar
  end interface operator(>)

  interface operator(<)
     module procedure advar_lt_advar, advar_lt_real32, real32_lt_advar, &
          & advar_lt_real64, real64_lt_advar, advar_lt_real128, real128_lt_advar
  end interface operator(<)

  ! Most interfaces for safe_deallocate are in misc.f90. Adding this
  ! to misc.f90 would produce a circular dependence.
  interface safe_deallocate
     module procedure safe_deallocate_advar
  end interface safe_deallocate
  
  ! AD elemental operations. For most binary operations, all variants
  ! are present: (advar,advar), (advar,T), and (T,advar), where T can
  ! be a real(real32), real(real64), real(real128), or an integer. T
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
          & add_advar_real64, add_real64_advar
#else
          & add_advar_real128, add_real128_advar
#endif
  end interface operator(+)

  interface operator(-)
     module procedure subtract_advar_advar, subtract_advar, &
          & subtract_advar_real, subtract_real_advar, subtract_advar_integer, &
          & subtract_integer_advar, subtract_advar_real32, &
          & subtract_real32_advar, &
#ifdef QUAD_PRECISION
          & subtract_advar_real64, subtract_real64_advar
#else
          & subtract_advar_real128, subtract_real128_advar
#endif
  end interface operator(-)

  interface operator(*)
     module procedure multiply_advar_advar, multiply_advar_real, &
          & multiply_real_advar, multiply_advar_integer, &
          & multiply_integer_advar, multiply_advar_real32, &
          & multiply_real32_advar, &
#ifdef QUAD_PRECISION
          & multiply_advar_real64, multiply_real64_advar
#else
          & multiply_advar_real128, multiply_real128_advar
#endif
  end interface operator(*)

  interface operator(/)
     module procedure divide_advar_advar, divide_advar_real, &
          & divide_real_advar, divide_advar_integer, divide_integer_advar, &
          & divide_advar_real32, divide_real32_advar, &
#ifdef QUAD_PRECISION
          & divide_advar_real64, divide_real64_advar
#else
          & divide_advar_real128, divide_real128_advar
#endif
  end interface operator(/)

  interface operator(**)
     module procedure power_advar_advar, power_advar_real, power_real_advar, &
          & power_advar_integer, power_integer_advar, power_advar_real32, &
          & power_real32_advar, &
#ifdef QUAD_PRECISION
          & power_advar_real64, power_real64_advar
#else
          & power_advar_real128, power_real128_advar
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

#ifdef GSL_DIR
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
  !!***

  !!****f* automatic_differentiation/ad_init_reverse
  !!
  !! FUNCTION
  !!
  !! Initializes the main work variables of the reverse mode. One can
  !! specify either 'memory' or, for optimal use of memory,
  !! sweep_size, trace_size, and const_size separately. If all four
  !! are present, 'memory' takes precedence. With no arguments a call
  !! to init_ad_reverse allocates forward_values(DEFAULT_SWEEP_SIZE),
  !! adjoints(DEFAULT_SWEEP_SIZE), trace(4*DEFAULT_SWEEP_SIZE), and
  !! ad_constants(0.5*DEFAULT_SWEEP_SIZE). For a better understading
  !! of memory usage call ad_memory_report() after the
  !! calculation. Sets the default mode to be the reverse mode.
  !!
  !! INPUTS
  !!
  !! All optional
  !! 
  !! memory - allocates forward_values(x), adjoints(x), trace(4*x),
  !!          and ad_constants(x/2), where x is such that the total
  !!          memory reserved is equal to the amount specified by
  !!          'memory', whose format is 'integer unit', where 'unit'
  !!          is 'B', 'kB', 'MB', or 'GB' (also acceptable are 'b',
  !!          'kb', 'mb', or 'gb').
  !! sweep_size - size of forward_values and adjoints
  !! trace_size - size of trace
  !! const_size - size of ad_constants
  !!
  !! SOURCE
  subroutine ad_init_reverse(memory, sweep_size, trace_size, const_size)
    character(*), intent(in), optional :: memory
    integer, intent(in), optional :: sweep_size, trace_size, const_size
    integer :: sweep_size_loc, trace_size_loc, const_size_loc
    real(kp) :: mem_size
    character(2) :: unit
    if (present(memory)) then
       read(memory, *, iostat=err_stat, iomsg=err_msg) mem_size, unit
       call check_err(__FILE__, __LINE__)
       if (unit == 'B' .or. unit == 'b') then
          sweep_size_loc = int(real(mem_size, kp)/(2.5*kp + 4*kind(1)))
       else if (unit == 'kB' .or. unit == 'kb') then
          sweep_size_loc = int(real(mem_size, kp)/(2.5*kp + 4*kind(1))*1e3)
       else if (unit == 'MB' .or. unit == 'mb') then
          sweep_size_loc = int(real(mem_size, kp)/(2.5*kp + 4*kind(1))*1e6)
       else if (unit == 'GB' .or. unit == 'gb') then
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
  !!***

  !!****f* automatic_differentiation/operator(>)(<)
  !!
  !! SOURCE
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

  elemental logical function advar_gt_real64(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real64), intent(in) :: x2
    y = x1%val > x2
  end function advar_gt_real64

  elemental logical function real64_gt_advar(x1, x2) result(y)
    real(real64), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = x1 > x2%val
  end function real64_gt_advar

  elemental logical function advar_gt_real128(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real128), intent(in) :: x2
    y = x1%val > x2
  end function advar_gt_real128

  elemental logical function real128_gt_advar(x1, x2) result(y)
    real(real128), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = x1 > x2%val
  end function real128_gt_advar

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

  elemental logical function advar_lt_real64(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real64), intent(in) :: x2
    y = x1%val < x2
  end function advar_lt_real64

  elemental logical function real64_lt_advar(x1, x2) result(y)
    real(real64), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = x1 < x2%val
  end function real64_lt_advar

  elemental logical function advar_lt_real128(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real128), intent(in) :: x2
    y = x1%val < x2
  end function advar_lt_real128

  elemental logical function real128_lt_advar(x1, x2) result(y)
    real(real128), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = x1 < x2%val
  end function real128_lt_advar
  !!***

  !!****f* automatic_differentiation/advar%assignment(=)
  !!
  !! FUNCTION
  !!
  !! Conversion from real to type(advar) copies the real value to the
  !! val field; the result is a passive variable
  !! (index=d=dd=0). Conversion from type(advar) to real copies the
  !! val field.
  !!
  !! SOURCE
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

  impure elemental subroutine assign_advar_real64(this, x)
    class(advar), intent(out) :: this
    real(real64), intent(in) :: x
    this%val = real(x, kp)
  end subroutine assign_advar_real64

  impure elemental subroutine assign_advar_real128(this, x)
    class(advar), intent(out) :: this
    real(real128), intent(in) :: x
    this%val = real(x, kp)
  end subroutine assign_advar_real128

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

  elemental subroutine assign_real64_advar(x, this)
    real(real64), intent(out) :: x
    class(advar), intent(in) :: this
    x = real(this%val, real64)
  end subroutine assign_real64_advar

  elemental subroutine assign_real128_advar(x, this)
    real(real128), intent(out) :: x
    class(advar), intent(in) :: this
    x = real(this%val, real128)
  end subroutine assign_real128_advar
  !!***

  ! AD ELEMENTAL OPERATIONS BEGIN

  !!****f* automatic_differentiation/elemental_operations
  !!
  !! FUNCTION
  !!
  !! In the reverse mode, each elemental operation appends to trace,
  !! forward_values, and (optionally) ad_constants. In the forward
  !! mode, the derivatives are calculated on the fly.
  !!
  !! SOURCE
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
  
  type(advar) function add_advar_real64(x1,x2) result(y)
    type(advar), intent(in) :: x1
    real(real64), intent(in) :: x2
    y = add_advar_real(x1, real(x2, kp))
  end function add_advar_real64

  type(advar) function add_advar_real128(x1,x2) result(y)
    type(advar), intent(in) :: x1
    real(real128), intent(in) :: x2
    y = add_advar_real(x1, real(x2, kp))
  end function add_advar_real128

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

  type(advar) function add_real64_advar(x1, x2) result(y)
    real(real64), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = add_real_advar(real(x1, kp), x2)
  end function add_real64_advar

  type(advar) function add_real128_advar(x1, x2) result(y)
    real(real128), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = add_real_advar(real(x1, kp), x2)
  end function add_real128_advar

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

  type(advar) function subtract_advar_real64(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real64), intent(in) :: x2
    y = subtract_advar_real(x1, real(x2, kp))
  end function subtract_advar_real64

  type(advar) function subtract_advar_real128(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real128), intent(in) :: x2
    y = subtract_advar_real(x1, real(x2, kp))
  end function subtract_advar_real128

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

  type(advar) function subtract_real64_advar(x1, x2) result(y)
    real(real64), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = subtract_real_advar(real(x1, kp), x2)
  end function subtract_real64_advar

  type(advar) function subtract_real128_advar(x1, x2) result(y)
    real(real128), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = subtract_real_advar(real(x1, kp), x2)
  end function subtract_real128_advar

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

  type(advar) function multiply_advar_real64(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real64), intent(in) :: x2
    y = multiply_advar_real(x1, real(x2, kp))
  end function multiply_advar_real64

  type(advar) function multiply_advar_real128(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real128), intent(in) :: x2
    y = multiply_advar_real(x1, real(x2, kp))
  end function multiply_advar_real128

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

  type(advar) function multiply_real64_advar(x1, x2) result(y)
    real(real64), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = multiply_real_advar(real(x1, kp), x2)
  end function multiply_real64_advar

  type(advar) function multiply_real128_advar(x1, x2) result(y)
    real(real128), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = multiply_real_advar(real(x1, kp), x2)
  end function multiply_real128_advar

  type(advar) function multiply_integer_advar(x1, x2) result(y)
    integer, intent(in) :: x1
    type(advar), intent(in) :: x2
    y = multiply_real_advar(real(x1, kp), x2)
  end function multiply_integer_advar

  type(advar) function divide_advar_advar(x1, x2) result(y)
    type(advar), intent(in) :: x1, x2
    if (x1%index /= 0 .and. x2%index /= 0) then
       y%val = x1%val/x2%val
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
          y%d = (x1%d - y%val*x2%d)/x2%val
          y%dd = (x1%dd - y%val*x2%dd - 2*y%d*x2%d)/x2%val
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
    y%val = x1%val/x2
    if (x1%index /= 0) then
       if (reverse_mode) then
          index_count = index_count + 1
          const_count = const_count + 1
          y%index = index_count
          forward_values(index_count) = y%val
          ad_constants(const_count) = 1/x2
          trace(trace_count+1) = x1%index
          trace(trace_count+2) = y%index
          trace(trace_count+3) = MULTIPLY_DIVIDE_A_R
          trace_count = trace_count + 3
       else
          y%d = x1%d/x2
          y%dd = x1%dd/x2
          y%index = 1
       end if
    end if
  end function divide_advar_real

  type(advar) function divide_advar_real32(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real32), intent(in) :: x2
    y = divide_advar_real(x1, real(x2, kp))
  end function divide_advar_real32

  type(advar) function divide_advar_real64(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real64), intent(in) :: x2
    y = divide_advar_real(x1, real(x2, kp))
  end function divide_advar_real64

  type(advar) function divide_advar_real128(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real128), intent(in) :: x2
    y = divide_advar_real(x1, real(x2, kp))
  end function divide_advar_real128

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

  type(advar) function divide_real64_advar(x1, x2) result(y)
    real(real64), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = divide_real_advar(real(x1, kp), x2)
  end function divide_real64_advar

  type(advar) function divide_real128_advar(x1, x2) result(y)
    real(real128), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = divide_real_advar(real(x1, kp), x2)
  end function divide_real128_advar

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
          y%d = y%val*x2%d*log(x1%val) + x1%d*x2%val*x1%val**(x2%val-1)
          y%dd = y%d**2/y%val + y%val*(x2%dd*log(x1%val) + &
               & (2*x2%d*x1%d + x2%val*(x1%dd - x1%d**2/x1%val))/x1%val)
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
          y%dd = y%d**2/y%val + y%val*x2*(x1%dd - x1%d**2/x1%val)/x1%val
          y%index = 1
       end if
    end if
  end function power_advar_real

  type(advar) function power_advar_real32(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real32), intent(in) :: x2
    y = power_advar_real(x1, real(x2, kp))
  end function power_advar_real32

  type(advar) function power_advar_real64(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real64), intent(in) :: x2
    y = power_advar_real(x1, real(x2, kp))
  end function power_advar_real64

  type(advar) function power_advar_real128(x1, x2) result(y)
    type(advar), intent(in) :: x1
    real(real128), intent(in) :: x2
    y = power_advar_real(x1, real(x2, kp))
  end function power_advar_real128

  type(advar) function power_advar_integer(x1, x2) result(y)
    type(advar), intent(in) :: x1
    integer, intent(in) :: x2
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
          y%d = y%val*x2*x1%d/x1%val
          y%dd = y%d**2/y%val + y%val*x2*(x1%dd - x1%d**2/x1%val)/x1%val
          y%index = 1          
       end if
    end if
  end function power_advar_integer

  type(advar) function power_real_advar(x1, x2) result(y)
    real(kp), intent(in) :: x1
    type(advar), intent(in) :: x2
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
          y%d = y%val*x2%d*log(x1)
          y%dd = y%d**2/y%val + y%val*x2%dd*log(x1)
          y%index = 1
       end if
    end if
  end function power_real_advar

  type(advar) function power_real32_advar(x1, x2) result(y)
    real(real32), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = power_real_advar(real(x1, kp), x2)
  end function power_real32_advar

  type(advar) function power_real64_advar(x1, x2) result(y)
    real(real64), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = power_real_advar(real(x1, kp), x2)
  end function power_real64_advar

  type(advar) function power_real128_advar(x1, x2) result(y)
    real(real128), intent(in) :: x1
    type(advar), intent(in) :: x2
    y = power_real_advar(real(x1, kp), x2)
  end function power_real128_advar

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
          y%d = x%d/2/y%val
          y%dd = (x%dd/y%val - y%d*x%d/x%val)/2
          y%index = 1
       end if
    end if
  end function sqrt_advar

  type(advar) function log_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = x%d/x%val
          y%dd = (x%dd - x%d*y%d)/x%val
          y%index = 1
       end if
    end if
  end function log_advar

  type(advar) function sin_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = x%d*cos(x%val)
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = x%dd*y%d/x%d - x%d**2*y%val
          else
             y%dd = x%dd*cos(x%val)
          end if
          y%index = 1
       end if
    end if
  end function sin_advar

  type(advar) function cos_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = -x%d*sin(x%val)
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = x%dd*y%d/x%d -x%d**2*y%val
          else
             y%dd = -x%dd*sin(x%val)
          end if
          y%index = 1
       end if
    end if
  end function cos_advar

  type(advar) function tan_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = x%d/cos(x%val)**2
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = (x%dd/x%d + 2*x%d*y%val/cos(x%val))*y%d
          else
             y%dd = x%dd/cos(x%val)**2
          end if
          y%index = 1
       end if
    end if
  end function tan_advar

  type(advar) function asin_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = x%d/sqrt(1-x%val**2)
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = y%d*(x%dd + x%val*y%d**2)/x%d
          else
             y%dd = x%dd/sqrt(1-x%val**2)
          end if
          y%index = 1
       end if
    end if
  end function asin_advar

  type(advar) function acos_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = -x%d/sqrt(1-x%val**2)
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = y%d*(x%dd + x%val*y%d**2)/x%d
          else
             y%dd = -x%dd/sqrt(1-x%val**2)
          end if
          y%index = 1
       end if
    end if
  end function acos_advar

  type(advar) function atan_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = x%d/(1+x%val**2)
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = y%d*(x%dd/x%d - 2*x%val*y%d)
          else
             y%dd = x%dd/(1+x%val**2)
          end if
          y%index = 1
       end if
    end if
  end function atan_advar


  type(advar) function sinh_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = x%d*cosh(x%val)
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = x%dd*y%d/x%d + y%val*x%d**2
          else
             y%dd = x%dd*cosh(x%val)
          end if
          y%index = 1
       end if
    end if
  end function sinh_advar

  type(advar) function cosh_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = x%d*sinh(x%val)
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = x%dd*y%d/x%d + y%val*x%d**2
          else
             y%dd = x%dd*sinh(x%val)
          end if
          y%index = 1
       end if
    end if
  end function cosh_advar

  type(advar) function tanh_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = x%d*(1-tanh(x%val)**2)
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = y%d*(x%dd/x%d - 2*y%val*x%d)
          else
             y%dd = x%dd*(1-tanh(x%val)**2)
          end if
          y%index = 1
       end if
    end if
  end function tanh_advar

  type(advar) function asinh_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = x%d/sqrt(1+x%val**2)
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = y%d*(x%dd - x%val*y%d**2)/x%d
          else
             y%dd = x%dd/sqrt(1+x%val**2)
          end if
          y%index = 1
       end if
    end if
  end function asinh_advar

  type(advar) function acosh_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = x%d/sqrt(x%val**2-1)
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = y%d*(x%dd - x%val*y%d**2)/x%d
          else
             y%dd = x%dd/sqrt(x%val**2-1)
          end if
          y%index = 1
       end if
    end if
  end function acosh_advar

  type(advar) function atanh_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = x%d/(1-x%val**2)
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = y%d*(x%dd/x%d + 2*x%val*y%d)
          else
             y%dd = x%dd/(1-x%val**2)
          end if
          y%index = 1
       end if
    end if
  end function atanh_advar

  type(advar) function erf_advar(x) result(y)
    type(advar), intent(in) :: x
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
          y%d = x%d*2/sqrtpi*exp(-x%val**2)
          if (abs(x%d) > tiny(1.0_kp)) then
             y%dd = y%d*(x%dd/x%d - 2*x%val*x%d)
          else
             y%dd = x%dd*2/sqrtpi*exp(-x%val**2)
          end if
          y%index = 1
       end if
    end if
  end function erf_advar

#ifdef GSL_DIR
  type(advar) function Li2_advar(x) result(y)
    type(advar), intent(in) :: x
    y%val = Li2(real(x%val, real64))
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
  !!***

  ! AD ELEMENTAL OPERATIONS END

  !!****f* automatic_differentiation/ad_grad
  !!
  !! FUNCTION
  !!
  !! The return sweep of the reverse mode. Assumes that the forward
  !! sweep has been performed with function evaluation. The adjoints
  !! array is reinitialized to zero each time before the sweep. Note
  !! that after the sweep, forward_values need not be reinitialized
  !! since it can simply be overwritten.
  !!
  !! INPUTS
  !!
  !! num_parameters - Number of active parameters. After the return
  !!                  sweep, index_count will be reset to this
  !!                  number. If there are, e.g., 2 active parameters,
  !!                  then during the next forward sweep the new
  !!                  indices should start at 3.
  !!
  !! SOURCE
  subroutine ad_grad(num_parameters)
    integer, intent(in) :: num_parameters
    integer :: op_id
    if (num_parameters == 0) &
         & call error(__FILE__, __LINE__, 'No active parameters')
    if (index_count > size(forward_values) .or. trace_count > size(trace) .or. &
         & const_count > size(ad_constants)) &
         & call error(__FILE__, __LINE__, &
         & 'Too many elemental operations. Request more memory.', this_image())
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
                 & adjoints(trace(i-1))/sqrt((forward_values(trace(i-2)))**2+1)
            i = i - 3
         case(ACOSH_A)
            adjoints(trace(i-2)) = adjoints(trace(i-2)) + &
                 & adjoints(trace(i-1))/sqrt((forward_values(trace(i-2)))**2-1)
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
#ifdef GSL_DIR
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
  !!***
  
  !!****f* automatic_differentiation/ad_memory_report
  !!
  !! FUNCTION
  !!
  !! Prints the size of adjoints, trace, and const in terms of both
  !! the number of elements and the amount of memory used. The latter
  !! is different for double and quad precision. The sizes, which
  !! represent the maxima over all images, are shown as requested
  !! during initialization and how much was actually used.
  !!
  !! SOURCE
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
  !!***

  !!****f* automatic_differentiation/safe_deallocate_advar
  !!
  !! SOURCE
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
  !!***

  !!****f* automatic_differentiation/ad_close
  !!
  !! SOURCE
  subroutine ad_close()
    call safe_deallocate(__FILE__, __LINE__, forward_values)
    call safe_deallocate(__FILE__, __LINE__, adjoints)
    call safe_deallocate(__FILE__, __LINE__, trace)
    call safe_deallocate(__FILE__, __LINE__, ad_constants)
  end subroutine ad_close
  !!***
end module ad

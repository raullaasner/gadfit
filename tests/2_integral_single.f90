#include <config.h>

module integral_single_m
  ! See the user guide for information about this fitting function.

  use ad
  use fitfunction
  use gadf_constants
  use numerical_integration
  
  implicit none
  
  type, extends(fitfunc) :: integral_single
   contains
     procedure :: init => init_integral_single
     procedure :: eval => eval_integral_single
  end type integral_single

contains
  subroutine init_integral_single(this)
    class(integral_single), intent(out) :: this
    allocate(this%pars(2))
    call this%set(1, 'a')
    call this%set(2, 'b')
  end subroutine init_integral_single

  type(advar) function eval_integral_single(this, x) result(y)
    class(integral_single), intent(in) :: this
    real(kp), intent(in) :: x
    ! In order to pass variables to the integral, we declare a new
    ! array which must be of type(advar).
    type(advar) :: pars(2)
    pars(1) = this%pars(1) ! a
    pars(2) = this%pars(2) ! b
    y = pi*integrate(integral_single_int, pars, 0.0_kp, x)
  end function eval_integral_single

  type(advar) function integral_single_int(x, pars) result(y)
    ! If no extra parameters need to be passed to the integrand,
    ! 'pars' is empty.
    real(kp), intent(in) :: x
    type(advar), intent(in out) :: pars(:)
    type(advar) :: a, b ! This is how we define new work variables
    a = pars(1); b = pars(2)
    y = x**a*exp(-b*x**2)
  end function integral_single_int
end module integral_single_m

  use integral_single_m
  use gadfit
  
  implicit none

  type(integral_single) :: f

  ! rel_error is the error tolerance for numerical integration.
  call gadf_init(f, rel_error=1e-12_kp)

  call gadf_add_dataset(TESTS_BLD//'/2_integral_single_data')

  call gadf_set('a', 10.0, .true.)
  call gadf_set('b', 1.0, .true.)

  call gadf_set_errors(NONE)
  call gadf_set_verbosity(output='/dev/null')

  call gadf_fit(10.0, rel_error=1e-6)

  call gadf_print(output=TESTS_BLD//'/2_integral_single_results')

  ! The following is for CTest and can be ignored
#define TEST_TARGET 1.923267927408180_kp
  if (this_image() == 1 .and. &
       & abs(fitfuncs(1)%pars(1) - TEST_TARGET) > 5e-7_kp) then
     write(*,*)
     write(*,'(g0)') 'Error at 2_integral_single:'
     write(*,'(2(g0))') '  "a" at the end of the fitting procedure: ', &
          & fitfuncs(1)%pars(1)%val
     write(*,'(27(" "), 2(g0))') 'Expected value: ', TEST_TARGET
     write(*,*)
  end if
  
  call gadf_close()
end program

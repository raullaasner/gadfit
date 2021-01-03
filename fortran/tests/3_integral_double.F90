module integral_double_m
  ! See the user guide for information about this fitting function.

  use integral_double_data, only: x_data, y_data, weights

  use ad
  use fitfunction
  use gadf_constants
  use numerical_integration

  implicit none

  type, extends(fitfunc) :: integral_double
   contains
     procedure :: init => init_integral_double
     procedure :: eval => eval_integral_double
  end type integral_double

contains
  subroutine init_integral_double(this)
    class(integral_double), intent(out) :: this
    allocate(this%pars(2))
    call this%set(1, 'a')
    call this%set(2, 'b')
  end subroutine init_integral_double

  type(advar) function eval_integral_double(this, x) result(y)
    class(integral_double), intent(in) :: this
    real(kp), intent(in) :: x
    type(advar) :: pars(3)
    ! We pass both parameters and the argument to the integrand.
    pars(1) = this%pars(1)
    pars(2) = this%pars(2)
    pars(3) = x
    y = integrate(outer_integrand, pars, 0.0_kp, INFINITY)/x
  end function eval_integral_double

  type(advar) function outer_integrand(x, pars) result(y)
    ! The variable t, as seen in the user guide, is played by x here.
    ! real(kp), intent(in) :: x
    type(advar), intent(in) :: x
    type(advar), intent(in out) :: pars(:)
    ! Local variables are of type(advar). It rarely makes sense to
    ! declare them anything else; type(advar) is always safe.
    type(advar) :: a, b, tmp, pars2(1)
    a   = pars(1)
    b   = pars(2)
    tmp = pars(3)
    pars2(1) = 1 + b*a*erf(x)
    y = integrate(inner_integrand, pars2, 0.0_kp, tmp/b)
    y = exp(-x)*y
  end function outer_integrand

  type(advar) function inner_integrand(x, pars) result(y)
    ! real(kp), intent(in) :: x
    type(advar), intent(in) :: x
    type(advar), intent(in out) :: pars(:)
    type(advar) :: tmp
    tmp = pars(1)
    y = log((exp(x)-1.0_kp)*tmp+1.0_kp)/x
  end function inner_integrand
end module integral_double_m

  use integral_double_m
  use gadfit

  implicit none

  type(integral_double) :: f

  ! Here the precision of numerical integration is reduced in the
  ! interest of time. Reducing the relative error increases the number
  ! of AD elementary operations, in which case '10 MB' might not be
  ! enough.
  call gadf_init(f, ad_memory='10 MB', &
       & rel_error_inner=1e-6_kp, rel_error=1e-5_kp)

  call gadf_add_dataset(x_data, y_data, weights)

  call gadf_set('a', 1.0, .true.)
  call gadf_set('b', 1.0, .true.)

  ! The data point errors are provided by the user.
  call gadf_set_errors(USER)
  ! To see memory usage, set 'memory=.true.' and remove the output
  ! argument.
  call gadf_set_verbosity(output='/dev/null')

  call gadf_fit(0.1, accth=0.9, max_iter=3)

  call gadf_print(output='3_integral_double_results')

  ! The following is for CTest and can be ignored
  if (this_image() == 1) then
     block
       real(kp), parameter :: reference_value = 8.5799477799920343_kp
       if (abs(fitfuncs(1)%pars(1)%val - reference_value) > 1e-9_kp) then
          write(*,*)
          write(*,'(g0)') 'Error at 3_integral_double:'
          write(*,'(2(g0))') '  "a" at the end of the fitting procedure: ', &
               & fitfuncs(1)%pars(1)%val
          write(*,'(27x, 2(g0))') 'Expected value: ', reference_value
          write(*,*)
          error stop
       end if
     end block
  end if

  call gadf_close()
end program

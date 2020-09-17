module gaussian_m
  ! See the user guide for information about this fitting function.

  use ad
  use fitfunction
  use gadf_constants

  implicit none

  type, extends(fitfunc) :: gaussian
   contains
     procedure :: init => init_gaussian
     procedure :: eval => eval_gaussian
  end type gaussian

contains
  subroutine init_gaussian(this)
    class(gaussian), intent(out) :: this
    allocate(this%pars(4))
    ! For convenience, each parameter can also be given a name.
    call this%set(1, 'fmax')
    call this%set(2, 'x0')
    call this%set(3, 'a')
    call this%set(4, 'bgr')
  end subroutine init_gaussian

  type(advar) function eval_gaussian(this, x) result(y)
    class(gaussian), intent(in) :: this
    real(kp), intent(in) :: x
    y = this%pars(1)*exp(-((x-this%pars(2))/this%pars(3))**2) + this%pars(4)
  end function eval_gaussian
end module gaussian_m

  use gaussian_m
  use gadfit

  implicit none

  type(gaussian) :: f

  call gadf_init(f) ! Number of curves defaults to 1.

  call gadf_add_dataset(TESTS_BLD//'/1_gaussian_data')

  ! Since we have named the parameters we can use either the name or
  ! the parameter number.
  call gadf_set('fmax', 1.0, .true.)
  ! We impose the constraint that the data is symmetric around zero.
  call gadf_set('x0', 0.0, .false.)
  call gadf_set('a', 1.0, .true.)
  call gadf_set('bgr', 1.0, .true.)

  call gadf_set_errors(NONE)
  call gadf_set_verbosity(output='/dev/null')

  call gadf_fit(0.1)

  call gadf_print(output=TESTS_BLD//'/1_gaussian_results')

  ! The following is for CTest and can be ignored
#define TEST_TARGET 37.21457246216242_kp
  if (this_image() == 1 .and. &
       & abs(fitfuncs(1)%pars(3)%val - TEST_TARGET) > 5e-7_kp) then
     write(*,*)
     write(*,'(g0)') 'Error at 1_gaussian:'
     write(*,'(2(g0))') '  "a" at the end of the fitting procedure: ', &
          & fitfuncs(1)%pars(3)%val
     write(*,'(27(" "), 2(g0))') 'Expected value: ', TEST_TARGET
     write(*,*)
     error stop
  end if

  call gadf_close()
end program

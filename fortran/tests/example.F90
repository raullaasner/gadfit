module test_f
  ! This module defines the function I(x) = I0*exp(-t/tau)+bgr.
  ! In Fortran, comments begin with an "!".

  use ad ! Imports the module containing the procedures for
         ! AD. Similar to the #include directive in C.
  use fitfunction ! Imports the abstract class for the fitting
                  ! function.
  use gadf_constants ! Imports the floating point precision specifier
                     ! (kp).

  implicit none ! For historical reasons this should be present in all
                ! modules.

  ! We have chosen to call the type of the new fitting function
  ! 'exponential'. init and eval are the constructor and the function
  ! body, which must be renamed and implemented below.
  type, extends(fitfunc) :: exponential
   contains
     procedure :: init => init_exponential
     procedure :: eval => eval_exponential
  end type exponential

contains
  ! The interfaces of the following procedures, i.e., the type of the
  ! arguments and of the return value, are fixed.
  subroutine init_exponential(this)
    class(exponential), intent(out) :: this
    ! Allocate memory to the fitting parameter array.
    allocate(this%pars(3))
    ! In C++ this would be something like this->pars = new advar[3]
  end subroutine init_exponential

  type(advar) function eval_exponential(this, x) result(y)
    class(exponential), intent(in) :: this
    real(kp), intent(in) :: x
    y = this%pars(1)*exp(-x/this%pars(2)) + this%pars(3)
    ! In this context, I0 = this%pars(1), tau = this%pars(2), and
    ! bgr = this%pars(3).
  end function eval_exponential
end module test_f

  use test_f ! Include the above-defined fitting function
  use gadfit ! and the main library.

  implicit none

  type(exponential) :: f ! An instance of the fitting function

  ! Initialize GADfit with the fitting function and the number of
  ! datasets.
  call gadf_init(f, 2)

  ! Include both decay curves. The argument must be full or relative
  ! path to the data.
  call gadf_add_dataset(DATA_DIR//'/example_data1')
  call gadf_add_dataset(DATA_DIR//'/example_data2')

  ! The initial guess for all fitting parameters is 1.0. The first
  ! argument denotes the dataset, the second argument the parameter,
  ! third argument its value, and the fourth whether the parameter is
  ! allowed to vary or is kept fixed.
  call gadf_set(1, 1, 1.0, .true.) ! I01
  call gadf_set(2, 1, 1.0, .true.) ! I02
  call gadf_set(1, 3, 1.0, .true.) ! bgr1
  call gadf_set(2, 3, 1.0, .true.) ! bgr2
  ! Global parameters don't have the dataset argument.
  call gadf_set(2, 1.0, .true.)    ! tau

  ! The uncertainties of the data points determine their weighting in
  ! the fitting procedure. Here we are assuming shot noise, i.e., the
  ! error of each data point is proportional to the square root of its
  ! value. Default is no weighting.
  call gadf_set_errors(SQRT_Y)

  ! Perform the fitting procedure starting with lambda=10. If the
  ! procedure doesn't converge, we should restart with a higher value
  ! or modify any of the other arguments to gadf_fit. All the
  ! arguments are optional with reasonable default values.
  call gadf_fit(lambda=10.0)

  ! The results are saved into ~gadfit/tests
  call gadf_print(output='example_results')
  call gadf_close() ! Free memory
end program

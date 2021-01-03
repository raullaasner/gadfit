module multiple_curves_m
  ! This is the same as the example except the data is not read in
  ! from files.

  use multiple_curves_data, only: x_data_1, x_data_2, y_data_1, y_data_2

  use ad
  use fitfunction
  use gadf_constants

  implicit none

  type, extends(fitfunc) :: exponential
   contains
     procedure :: init => init_exponential
     procedure :: eval => eval_exponential
  end type exponential

contains
  subroutine init_exponential(this)
    class(exponential), intent(out) :: this
    allocate(this%pars(3))
  end subroutine init_exponential

  type(advar) function eval_exponential(this, x) result(y)
    class(exponential), intent(in) :: this
    real(kp), intent(in) :: x
    y = this%pars(1)*exp(-x/this%pars(2)) + this%pars(3)
  end function eval_exponential
end module multiple_curves_m

  use multiple_curves_m
  use gadfit

  implicit none

  type(exponential) :: f

  call gadf_init(f, 2)

  call gadf_add_dataset(x_data_1, y_data_1)
  call gadf_add_dataset(x_data_2, y_data_2)

  call gadf_set(1, 1, 1.0, .true.)
  call gadf_set(2, 1, 1.0, .true.)
  call gadf_set(1, 3, 1.0, .true.)
  call gadf_set(2, 3, 1.0, .true.)
  call gadf_set(2, 1.0, .true.)
  call gadf_set_errors(SQRT_Y)
  call gadf_set_verbosity(output='/dev/null')
  call gadf_fit(lambda=10.0, accth=0.9, max_iter=4)

  if (this_image() == 1) then
     block
       real(kp), parameter :: reference_values(3,2) = reshape([ &
            & 46.980695087179093_kp, &
            & 21.367028663570494_kp, &
            & 8.9528433588272360_kp, &
            & 150.03361724451275_kp, &
            & 21.367028663570494_kp, &
            & 4.3777353718042322_kp], shape(reference_values))
       integer :: i_dataset, i_par
       do i_dataset = 1, 2
          do i_par = 1, 3
             if (abs(fitfuncs(i_dataset)%pars(i_par)%val - &
                  & reference_values(i_par, i_dataset)) > 1e-13_kp) then
                write(*,*)
                write(*,'(2(g0))') 'Computed value: ', &
                     & fitfuncs(i_dataset)%pars(i_par)%val
                write(*,'(2(g0))') 'Expected value: ', &
                     & reference_values(i_par, i_dataset)
                write(*,*)
                error stop
             end if
          end do
       end do
     end block
  end if

  call gadf_close()
end program

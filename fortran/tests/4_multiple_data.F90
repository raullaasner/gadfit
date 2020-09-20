module multiple_data_m
  ! This is the same as the example except the data is not read in
  ! from files.

  use multiple_data_data, only: x_data_1, x_data_2, y_data_1, y_data_2

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
end module multiple_data_m

  use multiple_data_m
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
  call gadf_fit(lambda=10.0)

  if (this_image() == 1) then
     block
       real(kp), parameter :: reference_values(3,2) = reshape([ &
            & 47.434440947488504_kp, &
            & 20.521551299489758_kp, &
            & 9.1936229247057728_kp, &
            & 152.71446174360392_kp, &
            & 20.521551299489758_kp, &
            & 4.8709977251577286_kp], shape(reference_values))
       integer :: i_dataset, i_par
       do i_dataset = 1, 2
          do i_par = 1, 3
             if (abs(fitfuncs(i_dataset)%pars(i_par)%val - &
                  & reference_values(i_par, i_dataset)) > 5e-7_kp) then
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

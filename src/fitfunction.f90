! This Source Code Form is subject to the terms of the GNU General
! Public License, v. 3.0. If a copy of the GPL was not distributed
! with this file, You can obtain one at
! http://gnu.org/copyleft/gpl.txt.

! Defines and provides basic functionality for an abstract type for
! the fitting function, from which all user defined types are
! derived. Initialization and the value function are deferred to user
! defined types. Provides procedures for the gradient and second
! directional derivative using finite differences. For the calculation
! of derivatives using automatic differentiation, see
! automatic_differentiation.f90.
module fitfunction

  use ad
  use gadf_constants, only: kp
  use messaging
  use misc,           only: string, len, safe_deallocate

  implicit none

  private
  public :: fitfunc, safe_deallocate

  type, abstract :: fitfunc
     type(advar),  allocatable :: pars(:)
     type(string), allocatable :: parnames(:)
   contains
     ! init must allocate this%pars and optionally set the parameter
     ! names.
     procedure (init), deferred :: init
     procedure, private :: init_parnames
     procedure, private :: set_par_value_int_real
     procedure, private :: set_par_value_char_real
     procedure, private :: set_par_name
     generic :: set => set_par_value_int_real, set_par_value_char_real, &
          & set_par_name
     procedure :: get_index
     procedure :: get_name
     procedure (eval), deferred :: eval ! Function body
     procedure :: grad_finite ! Gradient using finite differences
     procedure :: dir_deriv_2nd_finite ! Directional derivative
     procedure :: info
     procedure :: destroy ! FINAL doesn't work with polymorphism
  end type fitfunc

  abstract interface
     subroutine init(this)
       import :: fitfunc
       class(fitfunc), intent(out) :: this
     end subroutine init
     type(advar) function eval(this, x)
       import :: advar, fitfunc, kp
       class(fitfunc), intent(in) :: this
       real(kp), intent(in) :: x
     end function eval
  end interface

  ! Most interfaces for safe_deallocate are in misc.f90. Adding this
  ! to misc.f90 would produce a circular dependence
  interface safe_deallocate
     module procedure safe_deallocate_fitfunc
  end interface safe_deallocate

contains

  subroutine init_parnames(this)
    class(fitfunc), intent(in out) :: this
    allocate(this%parnames(size(this%pars)), stat=err_stat, errmsg=err_msg)
    call check_err(__FILE__, __LINE__)
    this%parnames = ''
  end subroutine init_parnames

  ! Sets the value of the parameter par_i to val.
  subroutine set_par_value_int_real(this, par_i, val)
    class(fitfunc), intent(in out) :: this
    integer, intent(in) :: par_i
    real(kp), intent(in) :: val
    if (.not. allocated(this%pars)) &
         & call error(__FILE__, __LINE__, &
         & 'pars is uninitialized. Calling init() could solve the problem.')
    if (par_i < 1 .or. par_i > size(this%pars)) &
         & call error(__FILE__, __LINE__, 'Index out of bounds.')
    this%pars(par_i)%val = val
    ! Initialize parnames even if unused. This is important for some
    ! print procedures.
    if (.not. allocated(this%parnames)) call this%init_parnames()
  end subroutine set_par_value_int_real

  ! Sets the value of the parameter named 'name' to val.
  subroutine set_par_value_char_real(this, name, val)
    class(fitfunc), intent(in out) :: this
    character(*), intent(in) :: name
    real(kp), intent(in) :: val
    call set_par_value_int_real(this, this%get_index(name), val)
  end subroutine set_par_value_char_real

  subroutine set_par_name(this, par_i, name)
    class(fitfunc), intent(in out) :: this
    integer, intent(in) :: par_i
    character(*), intent(in) :: name
    integer :: i
    if (.not. allocated(this%parnames)) call this%init_parnames()
    do i = 1, size(this%parnames)
       if (this_image() == 1 .and. this%parnames(i) == name) &
            & call warning(__FILE__, __LINE__, &
            & 'The name "'//this%parnames(i)%name//'" is already in use.')
    end do
    this%parnames(par_i) = name
  end subroutine set_par_name

  integer function get_index(this, name) result(y)
    class(fitfunc), intent(in out) :: this
    character(*), intent(in) :: name
    character(:), allocatable :: name_list ! In case of error
    integer :: i
    if (.not. allocated(this%parnames)) call this%init_parnames()
    do i = 1, size(this%parnames)
       if (this%parnames(i) == name) then
          y = i
          return
       end if
    end do
    name_list = ' '//this%parnames(1)%name
    do i = 2, size(this%parnames)
       name_list = name_list//', '//this%parnames(i)%name
    end do
    call error(__FILE__, __LINE__, 'There is no parameter called "'//name// &
         & '". Allowed names are'//name_list//'.')
  end function get_index

  elemental type(string) function get_name(this, par_i) result(y)
    class(fitfunc), intent(in) :: this
    integer, intent(in) :: par_i
    y = this%parnames(par_i)
  end function get_name

  ! Calculates the gradient of the function with respect to the active
  ! parameters given by active_pars using finite differences. The
  ! result is put into grad(:n), where n is the number of active
  ! parameters. It is the caller's responsibility to ensure that
  ! size(grad) >= n.
  !
  ! this
  ! x - argument at which to evaluate the gradient
  ! active_pars - indices of the active parameters
  ! grad - the gradient vector
  subroutine grad_finite(this, x, active_pars, grad)
    class(fitfunc), intent(in out) :: this
    real(kp), intent(in) :: x
    integer, intent(in) :: active_pars(:)
    real(kp), intent(in out) :: grad(:)
    real(kp) :: saved_value, step
    integer :: i
    do i = 1, size(active_pars)
       saved_value = this%pars(active_pars(i))%val
       step = sqrt(epsilon(1.0_kp))*saved_value
       if (.not. abs(step) > tiny(0.0_kp)) &
            & call error(__FILE__, __LINE__, 'Absolute value of parameter '// &
            & str(active_pars(i))//' is too small.')
       this%pars(active_pars(i))%val = this%pars(active_pars(i))%val + step
       step = this%pars(active_pars(i))%val - saved_value
       grad(i) = this%eval(x)
       this%pars(active_pars(i)) = saved_value
       grad(i) = (grad(i) - this%eval(x))/step
    end do
  end subroutine grad_finite

  ! Calculates the 2nd directional derivative of the function with
  ! respect to the active parameters given by active_pars using finite
  ! differences. y = \partial_[m]\partial_[n] f b^[m]b^[n] (Einstein
  ! summation), where b is the vector along which to evaluate the
  ! derivative.
  !
  ! this
  ! x - argument at which to evaluate the derivative
  ! active_pars - indices of the active parameters
  ! dir - vector along which the derivative is calculated
  !
  ! y - the directional derivative
  real(kp) function dir_deriv_2nd_finite(this, x, active_pars, dir) result(y)
    class(fitfunc), intent(in out) :: this
    real(kp), intent(in) :: x, dir(:)
    integer, intent(in) :: active_pars(:)
    real(kp) :: saved_values(size(active_pars))
    saved_values = this%pars(active_pars)%val
    associate(h => sqrt(sqrt(epsilon(1.0_kp))))
      this%pars(active_pars)%val = this%pars(active_pars)%val + h*dir
      y = this%eval(x)
      this%pars(active_pars)%val = saved_values - h*dir
    end associate
    y = y + this%eval(x)
    this%pars(active_pars)%val = saved_values
    y = y - 2*this%eval(x)
    y = y/sqrt(epsilon(1.0_kp))
  end function dir_deriv_2nd_finite

  ! Prints the names and values of all parameters and whether they are
  ! active or passive.
  subroutine info(this)
    use, intrinsic :: iso_fortran_env, only: output_unit
    class(fitfunc), intent(in out) :: this
    integer :: i
    if (.not. allocated(this%pars)) call this%init()
    if (.not. allocated(this%parnames)) call this%init_parnames()
    if (size(this%pars) > 0) then
       do i = 1, size(this%pars)
          if (this%pars(i)%index == 0) then
             write(output_unit, '(g0)', advance='no') 'Passive'
          else
             write(output_unit, '(1x, g0)', advance='no') 'Active'
          end if
          write(output_unit, '(2x, a'//str(maxval(len(this%parnames)))// &
               & ', 2x, g0)') this%parnames(i)%name, this%pars(i)%val
       end do
    end if
  end subroutine info

  impure elemental subroutine destroy(this)
    class(fitfunc), intent(in out) :: this
    call safe_deallocate(__FILE__, __LINE__, this%pars)
    call safe_deallocate(__FILE__, __LINE__, this%parnames)
  end subroutine destroy
  subroutine safe_deallocate_fitfunc(file, line, array)
    ! 'file' and 'line' should be determined by the preprocessor.
    character(*), intent(in) :: file
    integer, intent(in) :: line
    class(fitfunc), allocatable, intent(in out) :: array(:)
    if (allocated(array)) then
       call array%destroy()
       deallocate(array, stat=err_stat, errmsg=err_msg)
       call check_err(file, line)
    end if
  end subroutine safe_deallocate_fitfunc
end module fitfunction

! This Source Code Form is subject to the terms of the GNU General
! Public License, v. 3.0. If a copy of the GPL was not distributed
! with this file, You can obtain one at
! http://gnu.org/copyleft/gpl.txt.

#include <config.h>

! Provides some commonly used types and helper procedures.
module misc

  use, intrinsic :: iso_fortran_env, only: int64
  use gadf_constants, only: dp, kp, qp
  use messaging

  implicit none

  private
  public :: string, len, swap, timer, safe_deallocate, safe_close
#if !defined HAS_CO_SUM || defined QUAD_PRECISION
  public :: co_sum
#endif

  type :: string
     character(:), allocatable :: name
   contains
     procedure :: assign_character
     generic :: assignment(=) => assign_character
     procedure :: equals_character
     generic :: operator(==) => equals_character
  end type string

  type :: timer
     ! Total cpu and wall times
     real(dp) :: cpu_time
     integer(int64) :: wall_time
     ! Number of times a code segment has been called
     integer :: num_calls
     ! whether timing is currently in progress
     logical, private :: timing
     real(dp), private :: cpu_time_tmp
     integer(int64), private :: wall_time_tmp
   contains
     procedure :: reset
     procedure :: time
  end type timer

#if !defined HAS_CO_SUM || defined QUAD_PRECISION
  interface co_sum
     module procedure co_sum_0d, co_sum_1d, co_sum_2d
  end interface co_sum
#endif

  interface len
     module procedure len_string
  end interface len

  interface safe_deallocate
     module procedure safe_deallocate_dp, safe_deallocate_qp, &
          & safe_deallocate_dp_2d, safe_deallocate_qp_2d, &
          & safe_deallocate_integer, safe_deallocate_string, &
          & safe_deallocate_logical
  end interface safe_deallocate

contains

  impure elemental subroutine assign_character(this, x)
    class(string), intent(out) :: this
    character(*), intent(in) :: x
    this%name = x
  end subroutine assign_character

  logical function equals_character(this, x) result(y)
    class(string), intent(in) :: this
    character(*), intent(in) :: x
    if (this%name == x) then
       y = .true.
    else
       y = .false.
    end if
  end function equals_character

  elemental integer function len_string(x) result(y)
    type(string), intent(in) :: x
    y = len(x%name)
  end function len_string

  elemental subroutine swap(a, b)
    integer, intent(in out) :: a, b
    integer :: tmp
    tmp = a; a = b; b = tmp
  end subroutine swap

  subroutine reset(this)
    class(timer), intent(out) :: this
    this%cpu_time = 0d0
    this%wall_time = 0
    this%num_calls = 0
    this%timing = .false.
  end subroutine reset

  ! Find the cpu and wall times of a code segment. The code segment
  ! should be wrapped between two calls to time.
  subroutine time(this)
    class(timer), intent(in out) :: this
    real(dp) :: cpu_tmp
    integer(int64) :: wall_tmp
    if (.not. this%timing) then
       call cpu_time(this%cpu_time_tmp)
       call system_clock(this%wall_time_tmp)
       this%timing = .true.
    else
       call cpu_time(cpu_tmp)
       this%cpu_time = this%cpu_time + cpu_tmp - this%cpu_time_tmp
       call system_clock(wall_tmp)
       this%wall_time = this%wall_time + wall_tmp - this%wall_time_tmp
       this%num_calls = this%num_calls + 1
       this%timing = .false.
    end if
  end subroutine time

  ! Simple implementations of some of the "co_" procedures in case the
  ! compiler lacks intrinsic support.
#if !defined HAS_CO_SUM || defined QUAD_PRECISION
  subroutine co_sum_0d(a)
    real(kp), intent(in out) :: a[*]
    real(kp) :: tmp
    integer i
    sync all
    tmp = a[1]
    do i = 2, num_images()
       tmp = tmp + a[i]
    end do
    sync all
    a = tmp
  end subroutine co_sum_0d

  subroutine co_sum_1d(a)
    real(kp), intent(in out) :: a(:)[*]
    real(kp) :: tmp (size(a))
    integer i
    sync all
    tmp = a(:)[1]
    do i = 2, num_images()
       tmp = tmp + a(:)[i]
    end do
    sync all
    a = tmp
  end subroutine co_sum_1d

  subroutine co_sum_2d(a)
    real(kp), intent(in out) :: a(:,:)[*]
    real(kp) :: tmp(size(a,1),size(a,2))
    integer i
    sync all
    tmp = a(:,:)[1]
    do i = 2, num_images()
       tmp = tmp + a(:,:)[i]
    end do
    sync all
    a = tmp
  end subroutine co_sum_2d
#endif

  ! Deallocates an array. It is always safe to call these
  ! prcedures. 'file' and 'line' should be determined by the
  ! preprocessor.
  subroutine safe_deallocate_dp(file, line, array)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    real(dp), allocatable, intent(in out) :: array(:)
    if (allocated(array)) then
       deallocate(array, stat=err_stat, errmsg=err_msg)
       call check_err(file, line)
    end if
  end subroutine safe_deallocate_dp

  subroutine safe_deallocate_qp(file, line, array)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    real(qp), allocatable, intent(in out) :: array(:)
    if (allocated(array)) then
       deallocate(array, stat=err_stat, errmsg=err_msg)
       call check_err(file, line)
    end if
  end subroutine safe_deallocate_qp

  subroutine safe_deallocate_dp_2d(file, line, array)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    real(dp), allocatable, intent(in out) :: array(:,:)
    if (allocated(array)) then
       deallocate(array, stat=err_stat, errmsg=err_msg)
       call check_err(file, line)
    end if
  end subroutine safe_deallocate_dp_2d

  subroutine safe_deallocate_qp_2d(file, line, array)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    real(qp), allocatable, intent(in out) :: array(:,:)
    if (allocated(array)) then
       deallocate(array, stat=err_stat, errmsg=err_msg)
       call check_err(file, line)
    end if
  end subroutine safe_deallocate_qp_2d

  subroutine safe_deallocate_integer(file, line, array)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    integer, allocatable, intent(in out) :: array(:)
    if (allocated(array)) then
       deallocate(array, stat=err_stat, errmsg=err_msg)
       call check_err(file, line)
    end if
  end subroutine safe_deallocate_integer

  subroutine safe_deallocate_string(file, line, array)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    type(string), allocatable, intent(in out) :: array(:)
    if (allocated(array)) then
       deallocate(array, stat=err_stat, errmsg=err_msg)
       call check_err(file, line)
    end if
  end subroutine safe_deallocate_string

  subroutine safe_deallocate_logical(file, line, array)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    logical, allocatable, intent(in out) :: array(:)
    if (allocated(array)) then
       deallocate(array, stat=err_stat, errmsg=err_msg)
       call check_err(file, line)
    end if
  end subroutine safe_deallocate_logical

  ! Closes an I/O device and checks the success of that
  ! operation. 'file' and 'line' should be determined by the
  ! preprocessor.
  subroutine safe_close(file, line, io_unit)
    character(*), intent(in) :: file
    integer, intent(in) :: line, io_unit
    close(io_unit, iostat=err_stat, iomsg=err_msg)
    call check_err(file, line)
  end subroutine safe_close
end module misc

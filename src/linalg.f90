!!****m* GADfit/linalg
!!
!! COPYRIGHT
!!
!! This Source Code Form is subject to the terms of the GNU General
!! Public License, v. 3.0. If a copy of the GPL was not distributed
!! with this file, You can obtain one at
!! http://gnu.org/copyleft/gpl.txt.
!!
!! FUNCTION
!!
!! Wrappers for linear algebra routines.
!!
!! SOURCE
module linalg

  use, intrinsic :: iso_fortran_env, only: real64
  use messaging, only: str, error, err_stat

  implicit none

  private
  public :: potr_f08

contains
  !!***

  !!****f* gadfit/potr_f08
  !!
  !! FUNCTION
  !!
  !! Solves A*X=B with A a real symmetric positive definite matrix in
  !! two steps: first by computing the Cholesky factorization of the
  !! form U^T*U or L*L^T of A, and then solving the linear system of
  !! equations U*X=B or L*X=B.
  !!
  !! INPUTS
  !!
  !! file, line - caller's location
  !! a - a real symmetric positive definite matrix
  !! b - on entry this is B, on exit contains the solution X
  !! ul (optional) - whether to store the upper ('u') or the lower
  !!                 ('l') triangle of 'a'
  !!
  !! SOURCE
  subroutine potr_f08(file, line, a, b, ul)
    ! 'file' and 'line' should be determined by the preprocessor.
    character(*), intent(in) :: file
    integer, intent(in) :: line
    real(real64), intent(in out) :: a(:,:), b(:)
    character(*), intent(in), optional :: ul
    character(1) :: ul_loc
    ul_loc = 'u'
    if (present(ul)) ul_loc = ul
    if (size(a,1) /= size(a,2)) &
         & call error(file, line, &
         & 'Not a square matrix ('//str(size(a,1))//'/'//str(size(b))//').')
    if (size(a,1) /= size(b)) &
         & call error(file, line, 'Incompatible array dimensions ('// &
         str(size(a,1))//'/'//str(size(b))//').')
    call dpotrf(ul_loc, size(b), a, size(b), err_stat)
    if (err_stat /= 0) &
         & call error(file, line, 'Cholesky factorization failed (dpotrf).')
    call dpotrs(ul_loc, size(b), 1, a, size(b), b, size(b), err_stat)
    if (err_stat /= 0) &
         & call error(file, line, &
         & 'Could not solve the linear system of equations (dpotrs).')
  end subroutine potr_f08
  !!***
end module linalg

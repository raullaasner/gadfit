! Licensed under the Apache License, Version 2.0 (the "License"); you
! may not use this file except in compliance with the License.  You
! may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
! implied.  See the License for the specific language governing
! permissions and limitations under the License.

! Wrappers for linear algebra routines.
module gadfit_linalg

  use gadf_constants, only: dp
  use messaging,      only: str, error, err_stat

  implicit none

  private
  public :: potr_f08

contains

  ! Solves A*X=B with A a real symmetric positive definite matrix in
  ! two steps: first by computing the Cholesky factorization of the
  ! form U^T*U or L*L^T of A, and then solving the linear system of
  ! equations U*X=B or L*X=B.
  !
  ! file, line - caller's location
  ! a - a real symmetric positive definite matrix
  ! b - on entry this is B, on exit contains the solution X
  ! ul (optional) - whether to store the upper ('u') or the lower
  !                 ('l') triangle of 'a'
  subroutine potr_f08(file, line, a, b, ul)
    ! 'file' and 'line' should be determined by the preprocessor.
    character(*), intent(in) :: file
    integer, intent(in) :: line
    real(dp), intent(in out) :: a(:,:), b(:)
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
end module gadfit_linalg

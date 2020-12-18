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

module gadf_constants

  implicit none

  public

  integer, parameter :: dp = selected_real_kind(15, 307)
  integer, parameter :: qp = selected_real_kind(33, 4931)

#ifdef QUAD_PRECISION
  integer, parameter :: kp = qp
#else
  integer, parameter :: kp = dp
#endif
  ! Zwillinger, D., Standard Mathematical Tables and Formulae, 31st
  ! Ed. Boca Raton, FL: Chapman and Hall/CRC Press LLC, 2003
  real(kp), parameter :: pi = 3.14159265358979323846264338327950_kp
  real(kp), parameter :: pi_2 = 6.28318530717958647692528676655901_kp
  real(kp), parameter :: pi2 = 9.86960440108935861883449099987615_kp
  real(kp), parameter :: sqrtpi = 1.77245385090551602729816748334115_kp
  real(kp), parameter :: euler = 2.71828182845904523536028747135266_kp
end module gadf_constants

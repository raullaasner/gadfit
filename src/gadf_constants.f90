! This Source Code Form is subject to the terms of the GNU General
! Public License, v. 3.0. If a copy of the GPL was not distributed
! with this file, You can obtain one at
! http://gnu.org/copyleft/gpl.txt.

#include <config.h>

module gadf_constants

  implicit none

  public

#ifdef QUAD_PRECISION
  integer, parameter :: kp = selected_real_kind(33, 4931)
#else
  integer, parameter :: kp = selected_real_kind(15, 307)
#endif
  ! Zwillinger, D., Standard Mathematical Tables and Formulae, 31st
  ! Ed. Boca Raton, FL: Chapman and Hall/CRC Press LLC, 2003
  real(kp), parameter :: pi = 3.14159265358979323846264338327950_kp
  real(kp), parameter :: pi_2 = 6.28318530717958647692528676655901_kp
  real(kp), parameter :: pi2 = 9.86960440108935861883449099987615_kp
  real(kp), parameter :: sqrtpi = 1.77245385090551602729816748334115_kp
  real(kp), parameter :: euler = 2.71828182845904523536028747135266_kp
end module gadf_constants

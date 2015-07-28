!!****m* GADfit/gadf_constants
!! 
!! COPYRIGHT
!! 
!! Copyright (C) 2014-2015 Raul Laasner
!! This file is distributed under the terms of the GNU General Public
!! License, see LICENSE in the root directory of the present
!! distribution or http://gnu.org/copyleft/gpl.txt .
!!
!! SOURCE
#include <config.h>

module gadf_constants

#ifdef QUAD_PRECISION
  use, intrinsic :: iso_fortran_env, only: kp => real128
#else
  use, intrinsic :: iso_fortran_env, only: kp => real64
#endif

  implicit none
  
  public

  ! Zwillinger, D., Standard Mathematical Tables and Formulae, 31st
  ! Ed. Boca Raton, FL: Chapman and Hall/CRC Press LLC, 2003
  real(kp), parameter :: pi = 3.14159265358979323846264338327950_kp
  real(kp), parameter :: pi_2 = 6.28318530717958647692528676655901_kp
  real(kp), parameter :: pi2 = 9.86960440108935861883449099987615_kp
  real(kp), parameter :: sqrtpi = 1.77245385090551602729816748334115_kp
  real(kp), parameter :: euler = 2.71828182845904523536028747135266_kp
end module gadf_constants
!!***

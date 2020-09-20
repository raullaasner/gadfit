! Fixtures and other helper functions for the tests

module testing

  use, intrinsic :: iso_fortran_env, only: error_unit, real32

  use ad
  use gadf_constants, only: kp, qp

  implicit none

  public

  interface test
     module procedure :: test_advar_real_values, test_bool, test_real_value, &
          & test_real32_value, test_real128_value, test_integer_value, &
          & test_advar_real32_values
  end interface test

  real(kp), parameter :: error_tolerance = 1d1*epsilon(1d0)

  ! Random values in the range -10...10
  real(kp), parameter :: fix_d(*) = [ &
       &  6.1360420701563498d0, &
       &  2.9606444748278875d0, &
       &  9.9253736972586246d0, &
       &  0.5356792380861322d0, &
       & -1.4727205961479033d0, &
       &  4.2512661200877879d0, &
       & -2.9410316453781444d0, &
       &  3.4797551257539538d0, &
       &  2.8312317178378699d0, &
       & -1.4900798993157309d0, &
       & -9.7526376845644123d0, &
       & -8.8824179995985126d0, &
       &  6.7638244484618752d0, &
       & -7.1130268493509963d0, &
       & -4.5835128246417494d0, &
       & -8.9059115759599745d0, &
       &  3.2898784649467867d0, &
       &  2.1875264606693996d0, &
       &  7.5767671483267520d0, &
       &  9.7405995203640394d0]

  real(qp), parameter :: fix_q(*) = [ &
       &  6.1360420701563498_qp, &
       &  2.9606444748278875_qp, &
       &  9.9253736972586246_qp, &
       &  0.5356792380861322_qp, &
       & -1.4727205961479033_qp, &
       &  4.2512661200877879_qp, &
       & -2.9410316453781444_qp, &
       &  3.4797551257539538_qp, &
       &  2.8312317178378699_qp, &
       & -1.4900798993157309_qp, &
       & -9.7526376845644123_qp, &
       & -8.8824179995985126_qp, &
       &  6.7638244484618752_qp, &
       & -7.1130268493509963_qp, &
       & -4.5835128246417494_qp, &
       & -8.9059115759599745_qp, &
       &  3.2898784649467867_qp, &
       &  2.1875264606693996_qp, &
       &  7.5767671483267520_qp, &
       &  9.7405995203640394_qp]

  real(real32), parameter :: fix_f(*) = [ &
       &  6.1360420701563498, &
       &  2.9606444748278875, &
       &  9.9253736972586246, &
       &  0.5356792380861322, &
       & -1.4727205961479033, &
       &  4.2512661200877879, &
       & -2.9410316453781444, &
       &  3.4797551257539538, &
       &  2.8312317178378699, &
       & -1.4900798993157309, &
       & -9.7526376845644123, &
       & -8.8824179995985126, &
       &  6.7638244484618752, &
       & -7.1130268493509963, &
       & -4.5835128246417494, &
       & -8.9059115759599745, &
       &  3.2898784649467867, &
       &  2.1875264606693996, &
       &  7.5767671483267520, &
       &  9.7405995203640394]

  integer, parameter :: fix_i(*) = &
       & [ 6, 2, 9, 0, -1, 4, -2, 3, 2, -1, -9, -8, 6, -7, -4, -8, 3, 2, 7, 9]

contains
  impure elemental subroutine test_bool(file, line, condition)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    logical, intent(in) :: condition
    if (.not. condition) then
       write(error_unit, '(a, a, ":", i0)') 'Error at ', file, line
       write(error_unit, '(2x, a)') 'A conditional failed'
       error stop
    end if
  end subroutine test_bool

  impure elemental subroutine test_real_value(file, line, computed_value, &
       & reference_value)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    real(kp), intent(in) :: computed_value, reference_value
    if (abs(computed_value - reference_value) > error_tolerance) then
       write(error_unit, '(a, a, ":", i0)') 'Error at ', file, line
       write(error_unit, '(2x, a, g0)') 'Computed value: ', computed_value
       write(error_unit, '(2x, a, g0)') 'Expected value: ', reference_value
       error stop
    end if
  end subroutine test_real_value

  impure elemental subroutine test_real32_value(file, line, computed_value, &
       & reference_value)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    real(real32), intent(in) :: computed_value, reference_value
    if (abs(computed_value - reference_value) > sqrt(error_tolerance)) then
       write(error_unit, '(a, a, ":", i0)') 'Error at ', file, line
       write(error_unit, '(2x, a, g0)') 'Computed value: ', computed_value
       write(error_unit, '(2x, a, g0)') 'Expected value: ', reference_value
       error stop
    end if
  end subroutine test_real32_value

  impure elemental subroutine test_real128_value(file, line, computed_value, &
       & reference_value)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    real(qp), intent(in) :: computed_value, reference_value
    if (abs(computed_value - reference_value) > error_tolerance) then
       write(error_unit, '(a, a, ":", i0)') 'Error at ', file, line
       write(error_unit, '(2x, a, g0)') 'Computed value: ', computed_value
       write(error_unit, '(2x, a, g0)') 'Expected value: ', reference_value
       error stop
    end if
  end subroutine test_real128_value

  impure elemental subroutine test_integer_value(file, line, computed_value, &
       & reference_value)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    integer, intent(in) :: computed_value, reference_value
    if (computed_value /= reference_value) then
       write(error_unit, '(a, a, ":", i0)') 'Error at ', file, line
       write(error_unit, '(2x, a, g0)') 'Computed value: ', computed_value
       write(error_unit, '(2x, a, g0)') 'Expected value: ', reference_value
       error stop
    end if
  end subroutine test_integer_value

  subroutine test_advar_real_values(file, line, computed_value, &
       & reference_values)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    type(advar), intent(in) :: computed_value
    real(kp), intent(in) :: reference_values(3)
    call test_real_value(file, line, &
         & [computed_value%val, computed_value%d, computed_value%dd], &
         & reference_values)
  end subroutine test_advar_real_values

  subroutine test_advar_real32_values(file, line, computed_value, &
       & reference_values)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    type(advar), intent(in) :: computed_value
    real(real32), intent(in) :: reference_values(3)
    call test_real32_value(file, line, &
         & real([computed_value%val, computed_value%d, computed_value%dd], &
         & real32), &
         & reference_values)
  end subroutine test_advar_real32_values
end module testing

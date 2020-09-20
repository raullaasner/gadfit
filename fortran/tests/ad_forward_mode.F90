#define POS __FILE__, __LINE__

  use testing

  implicit none

  ! Run these tests only in serial
  if (this_image() /= 1) return

  ! Comparisons
  block
    type(advar) :: a, b
    a%val = test_parameters(1)
    b%val = test_parameters(2)
    call test(POS, a > b)
    call test(POS, b < a)
    call test(POS, a > real(test_parameters(2), real32))
    call test(POS, real(test_parameters(2), real32) < a)
    call test(POS, a < real(test_parameters(3), real32))
    call test(POS, real(test_parameters(3), real32) > a)
    call test(POS, a > test_parameters(2))
    call test(POS, test_parameters(2) < a)
    call test(POS, a < test_parameters(3))
    call test(POS, test_parameters(3) > a)
    call test(POS, a > real(test_parameters(2), 16))
    call test(POS, real(test_parameters(2), 16) < a)
    call test(POS, a < real(test_parameters(3), 16))
    call test(POS, real(test_parameters(3), 16) > a)
  end block

  ! Assignments
  block
    type(advar) :: a
    real(real32) :: b
    real(kp) :: c
    real(16) :: d
    integer :: e
    a = nint(test_parameters(1))
    call test(POS, a, [6d0, 0d0, 0d0])
    a = real(test_parameters(1), real32)
    call test(POS, a, [real(test_parameters(1), real32), 0.0, 0.0])
    a = test_parameters(1)
    call test(POS, a, [test_parameters(1), 0d0, 0d0])
    a = real(test_parameters(1), 16)
    call test(POS, a, [test_parameters(1), 0d0, 0d0])
    b = a
    call test(POS, b, real(test_parameters(1), real32))
    c = a
    call test(POS, c, test_parameters(1))
    d = a
    call test(POS, d, real(test_parameters(1), 16))
    e = a
    call test(POS, e, 6)
  end block

  ! Memory allocation
  block
    type(advar), allocatable :: a(:)
    allocate(a(2))
    a = test_parameters(1)
    call safe_deallocate(POS, a)
  end block

  ! Basic arithmetic
  block
    real(kp), parameter :: ref_value = -2.6269013312046944d-3
    real(kp), parameter :: references(3,8) = reshape([ &
         & ref_value,  2.9290627900623777d-004,   1.2300413970479135d-004, &
         & ref_value,  5.6302216437492463d-004,   3.3250580791940149d-004, &
         & ref_value, -8.4884222866375510d-005,  -7.3979110754473495d-005, &
         & ref_value,  1.8523166250231145d-004,   1.6993708497399016d-004, &
         & ref_value,  1.0767461650392624d-004,  -2.2125820793826775d-005, &
         & ref_value,  3.7779050187261316d-004,   2.6912584697314114d-004, &
         & ref_value, -2.7011588536868697d-004,  -3.2566619528082142d-004, &
         & ref_value,  0d0, 0d0], shape(references))
    real(kp), parameter :: par_kp = test_parameters(1)
    real(real32), parameter :: par_32 = test_parameters(2)
    real(16), parameter :: par_128 = test_parameters(3)
    integer, parameter :: par_int = nint(test_parameters(4))
    type(advar) :: a, b, c, expression
    integer :: i1, i2, i3, test_counter
    a = test_parameters(5)
    b = test_parameters(6)
    c = test_parameters(7)
    test_counter = 1
    do i1 = -1, 0
       do i2 = -1, 0
          do i3 = -1, 0
             a%index = i1
             a%d = -i1*1d0
             ! This initial value for dd probably doesn't make sense
             ! in practice. We're only testing the algorithm here.
             a%dd = -i1*1d0
             b%index = i2
             b%d = -i2*1d0
             b%dd = -i2*1d0
             c%index = i3
             c%d = -i3*1d0
             c%dd = -i3*1d0
             expression = test_parameters(8)*(a + par_kp) + &
                  & b*(par_kp - c) - (c - par_kp)/(par_kp + a) + &
                  & (-b)*test_parameters(9)
             expression = add_dp_advar(test_parameters(9), expression)
             expression = add_advar_dp(expression, test_parameters(9))
             expression = &
                  & subtract_advar_dp(expression, test_parameters(8))
             expression = &
                  & subtract_dp_advar(test_parameters(8), expression)
             expression = &
                  & multiply_advar_dp(expression, test_parameters(8))
             expression = &
                  & divide_advar_dp(expression, test_parameters(8))
             expression = &
                  & multiply_dp_advar(test_parameters(8), expression)
             expression = &
                  & divide_dp_advar(test_parameters(8), expression)
             expression = par_32*expression/par_32
             expression = par_32/expression*par_32
             expression = par_128*expression/par_128
             expression = par_int*expression/par_int
             expression = par_int/expression*par_int
             call test(POS, expression, references(:,test_counter))
             test_counter = test_counter + 1
          end do
       end do
    end do
  end block

  ! Exponentiation, logarithms
  block
    real(kp), parameter :: ref_value = 13996805112922.668d0
    real(kp), parameter :: references(3,4) = reshape([ &
         & ref_value, 203154626568451.53d0, 3198942658368528.0d0, &
         & ref_value, 160843845261.89139d0, 162390992709.71570d0, &
         & ref_value, 202993782723189.59d0, 3193984279548728.0d0, &
         & ref_value, 0d0, 0d0], shape(references))
    real(kp), parameter :: par_kp = test_parameters(1)
    real(real32), parameter :: par_32 = test_parameters(2)
    real(16), parameter :: par_128 = test_parameters(3)
    integer, parameter :: par_int = nint(test_parameters(4))
    type(advar) :: a, b, expression
    integer :: i1, i2, test_counter
    a%val = test_parameters(5)
    b%val = test_parameters(6)
    test_counter = 1
    do i1 = -1, 0
       do i2 = -1, 0
          a%index = i1
          a%d = -i1*1d0
          a%dd = -i1*1d0
          b%index = i2
          b%d = -i2*1d0
          b%dd = -i2*1d0
          expression = &
               & b**a + &
               & b**par_kp/b**par_32*b**par_128/b**par_int - &
               & par_kp**b/par_32**b*par_128**b/par_int**b* &
               & (abs(a)/abs(b))
          expression = power_advar_dp(expression, 1/par_kp)
          expression = power_dp_advar(par_kp, expression)
          expression = expression/exp(sqrt(log(b)))
          call test(POS, expression, references(:,test_counter))
          test_counter = test_counter + 1
       end do
    end do
  end block

  ! Trigonometric
  block
    real(kp), parameter :: ref_value = -0.84756731470205926d0
    real(kp), parameter :: references(3,4) = reshape([ &
         & ref_value, -0.95581756099443504d0, -15.651120019290964d0, &
         & ref_value, -1.4470151214729803d0, -19.528161631929262d0, &
         & ref_value, 0.49119756047854513d0, 1.6911147680304270d0, &
         & ref_value, 0d0, 0d0], shape(references))
    type(advar) :: a, b, expression
    integer :: i1, i2, test_counter
    a%val = test_parameters(5)
    b%val = test_parameters(6)
    test_counter = 1
    do i1 = -1, 0
       do i2 = -1, 0
          a%index = i1
          a%d = -i1*1d0
          a%dd = -i1*1d0
          b%index = i2
          b%d = -i2*1d0
          b%dd = -i2*1d0
          expression = &
               & sin(a*b)*cos(a)/cos(b) + &
               & tan(cos(a))/atan(b*asin(1/a)/acos(a/b)) + &
               & sinh(a/b)*cosh(a/b)**tanh(b/a) + &
               & asinh(a/b)*acosh(abs(b/a))**atanh(abs(a/b))
          call test(POS, expression, references(:,test_counter))
          test_counter = test_counter + 1
       end do
    end do
  end block

  ! Special
  block
    real(kp), parameter :: ref_value = 0.55128846666540832d0
    real(kp), parameter :: references(3,2) = reshape([ &
         & ref_value, 0d0, 0d0, &
         & ref_value, 0.84690224138588510d0, -0.90733589479805699d0 &
         & ], shape(references))
    type(advar) :: a, expression
    a%val = test_parameters(4)
    expression = erf(a)
    call test(POS, expression, references(:,1))
    a%index = -1
    a%d = 1d0
    a%d = 1d0
    a%val = test_parameters(4)
    expression = erf(a)
    call test(POS, expression, references(:,2))
  end block
end program

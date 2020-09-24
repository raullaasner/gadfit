#define POS __FILE__, __LINE__

  use testing

  implicit none

  ! Run these tests only in serial
  if (this_image() /= 1) return

  ! Comparisons
  block
    type(advar) :: a, b
    a%val = fix_d(1)
    b%val = fix_d(2)
    call test(POS, a > b)
    call test(POS, b < a)
    call test(POS, a > fix_f(2))
    call test(POS, fix_f(2) < a)
    call test(POS, a < fix_f(3))
    call test(POS, fix_f(3) > a)
    call test(POS, a > fix_d(2))
    call test(POS, fix_d(2) < a)
    call test(POS, a < fix_d(3))
    call test(POS, fix_d(3) > a)
    call test(POS, a > fix_q(2))
    call test(POS, fix_q(2) < a)
    call test(POS, a < fix_q(3))
    call test(POS, fix_q(3) > a)
  end block

  ! Assignments
  block
    type(advar) :: a
    real(real32) :: b
    real(kp) :: c
    real(qp) :: d
    integer :: e
    a = fix_i(1)
    call test(POS, a, [6d0, 0d0, 0d0])
    a = fix_f(1)
    call test(POS, a, [fix_f(1), 0.0, 0.0])
    a = fix_d(1)
    call test(POS, a, [fix_d(1), 0d0, 0d0])
    a = fix_q(1)
    call test(POS, a, [fix_d(1), 0d0, 0d0])
    b = a
    call test(POS, b, fix_f(1))
    c = a
    call test(POS, c, fix_d(1))
    d = a
    call test(POS, d, fix_q(1))
    e = a
    call test(POS, e, 6)
  end block

  ! Memory allocation
  block
    type(advar), allocatable :: a(:)
    allocate(a(2))
    a = fix_d(1)
    call safe_deallocate(POS, a)
  end block

  ! Basic arithmetic
  block
    real(kp), parameter :: ref_value = 17.070019074561593d0
    real(kp), parameter :: references(3,8) = reshape([ &
         & ref_value,   1.9151681173140911d0,   1.2073525195984129d0, &
         & ref_value,   3.4496862204095673d0,   3.4511016163800172d0, &
         & ref_value, -0.30778227321370000d0, -0.30378018762980596d0, &
         & ref_value,   1.2267358298817765d0,   1.2281512258522265d0, &
         & ref_value,  0.68843228743231477d0,  -2.3385395867257380d-2, &
         & ref_value,   2.2229503905277919d0,   2.2229503905277905d0, &
         & ref_value,  -1.5345181030954760d0,  -1.5345181030954760d0, &
         & ref_value,  0d0, 0d0], shape(references))
    real(kp), parameter :: par_kp = fix_d(1)
    real(real32), parameter :: par_f = fix_d(2)
    real(qp), parameter :: par_qp = fix_d(3)
    integer, parameter :: par_int = fix_i(9)
    type(advar) :: a, b, c, expression
    integer :: i1, i2, i3, test_counter
    a = fix_d(5)
    b = fix_d(6)
    c = fix_d(7)
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
             expression = fix_d(8)*(a + par_kp + par_f) + b*(par_kp - c) - &
                  & (c - par_kp)/(par_kp + a + par_qp + par_int) + (-b)*fix_d(9)
             expression = par_f + (par_qp + expression) - par_f - par_qp
             expression = par_qp - (par_f - (par_int - expression)) - par_int
             expression = add_dp_advar(fix_d(9), expression)
             expression = add_advar_dp(expression, fix_d(9))
             expression = subtract_advar_dp(expression, fix_d(8))
             expression = subtract_dp_advar(fix_d(8), expression)
             expression = multiply_advar_dp(expression, fix_d(8))
             expression = divide_advar_dp(expression, fix_d(8))
             expression = multiply_dp_advar(fix_d(8), expression)
             expression = divide_dp_advar(fix_d(8), expression)
             expression = par_f*expression/par_f
             expression = par_f/expression*par_f
             expression = par_qp*(par_qp/expression)/par_qp*par_qp
             expression = par_int*expression/par_int
             expression = par_int/expression*par_int
             call test(POS, expression, references(:,test_counter))
             test_counter = test_counter + 1
          end do
       end do
    end do
  end block

  ! Exponentiation, logarithm
  block
    real(kp), parameter :: ref_value = 18998439975.537479d0
    real(kp), parameter :: references(3,4) = reshape([ &
         & ref_value, 199282009046.68716d0, 2328449500178.3394d0, &
         & ref_value, 38415548.376606211d0, 38479204.243286937d0, &
         & ref_value, 199243593498.31061d0, 2327612220349.5225d0, &
         & ref_value, 0d0, 0d0], shape(references))
    real(kp), parameter :: par_kp = fix_d(1)
    real(real32), parameter :: par_f = fix_d(2)
    real(qp), parameter :: par_qp = fix_d(3)
    integer, parameter :: par_int = fix_i(9)
    type(advar) :: a, b, expression
    integer :: i1, i2, test_counter
    a%val = fix_d(5)
    b%val = fix_d(6)
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
               & b**par_kp/b**par_f*b**par_qp/b**par_int - &
               & par_kp**b/par_f**b*par_qp**b/par_int**b* &
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
         & ref_value, -0.95581756099443504d0, -15.650198027974600d0, &
         & ref_value, -1.4470151214729803d0, -19.527239640612894d0, &
         & ref_value, 0.49119756047854513d0, 1.6911147680304270d0, &
         & ref_value, 0d0, 0d0], shape(references))
    type(advar) :: a, b, expression
    integer :: i1, i2, test_counter
    a%val = fix_d(5)
    b%val = fix_d(6)
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
         & ref_value, 0.84690224138588510d0, -0.060433653412171925d0 &
         & ], shape(references))
    type(advar) :: a, expression
    a%val = fix_d(4)
    expression = erf(a)
    call test(POS, expression, references(:,1))
    a%index = -1
    a%d = 1d0
    a%dd = 1d0
    a%val = fix_d(4)
    expression = erf(a)
    call test(POS, expression, references(:,2))
  end block
end program

#define POS __FILE__, __LINE__

  use testing

  implicit none

  ! Initialization
  block
    call ad_init_reverse('64 B')
    call test(POS, size(trace), 4)
    call ad_close()
    call ad_init_reverse('1 kB')
    call test(POS, size(trace), 108)
    call ad_close()
    call ad_init_reverse('0.001 MB')
    call test(POS, size(trace), 108)
    call ad_close()
    call ad_init_reverse('0.00001 GB')
    call test(POS, size(trace), 1108)
    call ad_close()
    call ad_init_reverse(sweep_size=1000, trace_size=1000, const_size=100)
  end block

  ! Basic arithmetic
  block
    real(kp), parameter :: references(3,7) = reshape([ &
         & -1.0804635414747479d-3, 1.0285805895789901d-3, &
         & 2.4194688371043452d-4, 1.5111620074904533d-3, &
         & 2.4194688371043452d-4, 2.4194688371043452d-4, &
         & 1.5111620074904533d-3, -1.0804635414747479d-3, &
         & 1.0285805895789901d-3, 7.4092665000924591d-4, &
         & 8.4191590875158033d-4, 2.4194688371043452d-4, &
         & 7.4092665000924591d-4, -1.0804635414747479d-3, &
         & 8.4191590875158033d-4, 7.4092665000924591d-4, &
         & 1.5111620074904533d-3, 8.4191590875158033d-4, &
         & 7.4092665000924591d-4, 1.5111620074904533d-3, &
         & -1.0804635414747479d-3], shape(references))
    real(kp), parameter :: par_kp = fix_d(1)
    real(real32), parameter :: par_f = fix_d(2)
    real(qp), parameter :: par_qp = fix_d(3)
    integer, parameter :: par_int = fix_i(9)
    type(advar) :: a, b, c, expression
    integer :: i1, i2, i3, test_counter, n_active
    a = fix_d(5)
    b = fix_d(6)
    c = fix_d(7)
    test_counter = 1
    do i1 = 0, 1
       do i2 = 0, 1
          do i3 = 0, 1
             a%index = i1
             b%index = i2*(i1 + i2)
             c%index = i3*(i1 + i2 + i3)
             n_active = 0
             if (i1 > 0) then
                n_active = n_active + 1
                forward_values(n_active) = a%val
             end if
             if (i2 > 0) then
                n_active = n_active + 1
                forward_values(n_active) = b%val
             end if
             if (i3 > 0) then
                n_active = n_active + 1
                forward_values(n_active) = c%val
             end if
             index_count = n_active
             expression = fix_d(8)*(a + par_kp) + b*(par_kp - c) - &
                  & (c - par_kp)/(par_kp + a) + (-b)*fix_d(9)
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
             expression = par_qp*expression/par_qp
             expression = par_int*expression/par_int
             expression = par_int/expression*par_int
             if (n_active > 0) then
                call ad_grad(n_active)
                call test(POS, adjoints(:3), references(:,test_counter))
                test_counter = test_counter + 1
             end if
          end do
       end do
    end do
  end block

  ! Exponentiation, logarithms
  block
    real(kp), parameter :: references(2,3) = reshape([ &
         & 199243593498.31058d0, 8124.5154209683469d0, &
         & 38415548.376606211d0, 8124.5154209683460d0, &
         & 38415548.376606219d0, 199243593498.31058d0 &
         & ], shape(references))
    real(kp), parameter :: par_kp = fix_d(1)
    real(real32), parameter :: par_f = fix_d(2)
    real(qp), parameter :: par_qp = fix_d(3)
    integer, parameter :: par_int = fix_i(9)
    type(advar) :: a, b, expression
    integer :: i1, i2, n_active, test_counter
    a%val = fix_d(5)
    b%val = fix_d(6)
    test_counter = 1
    do i1 = 0, 1
       do i2 = 0, 1
          a%index = i1
          b%index = i2*(i1 + i2)
          n_active = 0
          if (i1 > 0) then
             n_active = n_active + 1
             forward_values(n_active) = a%val
          end if
          if (i2 > 0) then
             n_active = n_active + 1
             forward_values(n_active) = b%val
          end if
          index_count = n_active
          expression = &
               & b**a + &
               & b**par_kp/b**par_f*b**par_qp/b**par_int - &
               & par_kp**b/par_f**b*par_qp**b/par_int**b* &
               & (abs(a)/abs(b))
          expression = power_advar_dp(expression, 1/par_kp)
          expression = power_dp_advar(par_kp, expression)
          expression = expression/exp(sqrt(log(b)))
          if (n_active > 0) then
             call ad_grad(n_active)
             call test(POS, adjoints(:2), references(:,test_counter))
             test_counter = test_counter + 1
          end if
       end do
    end do
  end block

  ! Trigonometric
  block
    real(kp), parameter :: references(2,3) = reshape([ &
         & 0.49119756047854524d0, -0.22000999933155330d0, &
         & -1.4470151214729805d0, -0.22000999933155330d0, &
         & -1.4470151214729805d0,  0.49119756047854524d0 &
         & ], shape(references))
    type(advar) :: a, b, expression
    integer :: i1, i2, n_active, test_counter
    a%val = fix_d(5)
    b%val = fix_d(6)
    test_counter = 1
    do i1 = 0, 1
       do i2 = 0, 1
          a%index = i1
          b%index = i2*(i1 + i2)
          n_active = 0
          if (i1 > 0) then
             n_active = n_active + 1
             forward_values(n_active) = a%val
          end if
          if (i2 > 0) then
             n_active = n_active + 1
             forward_values(n_active) = b%val
          end if
          index_count = n_active
          expression = &
               & sin(a*b)*cos(a)/cos(b) + &
               & tan(cos(a))/atan(b*asin(1/a)/acos(a/b)) + &
               & sinh(a/b)*cosh(a/b)**tanh(b/a) + &
               & asinh(a/b)*acosh(abs(b/a))**atanh(abs(a/b))
          if (n_active > 0) then
             call ad_grad(n_active)
             call test(POS, adjoints(:2), references(:,test_counter))
             test_counter = test_counter + 1
          end if
       end do
    end do
  end block

  ! Special
  block
    real(kp), parameter :: ref_value = 0.84690224138588510d0
    type(advar) :: a, expression
    a%val = fix_d(4)
    a%index = 1
    forward_values(1) = a%val
    index_count = 1
    expression = erf(a)
    call ad_grad(1)
    call test(POS, adjoints(1), ref_value)
  end block

  ! Memory report
  block
    integer :: out_unit
    open(newunit=out_unit, file='/dev/null', action='write')
    call ad_memory_report(out_unit)
  end block
end program

! Test if a polymorphic array with allocatable components has the
! correct behaviour.
module m
  type, abstract :: base
     integer, allocatable :: n(:)
   contains
     procedure :: get
  end type base
  
  type, extends(base) :: t
  end type t

contains
  integer function get(this)
    class(base) :: this
    get = this%n(2)
  end function get
end module m

  use m
  
  class(base), allocatable :: array(:)
  type(t) :: f
  
  allocate(array(1), mold=f)
  allocate(array(1)%n(2))
  array(1)%n(2) = -654121
  if (array(1)%get() /= -654121) error stop
end program

! Accessing a co-scalar inside the associate construct of a
! sub-sub-program
module m
contains
  subroutine f()
  contains
    subroutine bla()
      integer, save :: s[*]
      associate(i=>1)
        s=1
      end associate
    end subroutine bla
  end subroutine f
end module m
end program

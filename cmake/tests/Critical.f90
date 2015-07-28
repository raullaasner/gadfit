! Critical inside a module procedure
module m
contains
  subroutine f()
    critical
    end critical
  end subroutine f
end module m
end program

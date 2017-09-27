module <module_name>

  use ad
  use fitfunction
  use gadf_constants

  implicit none

  type, extends(fitfunc) :: <function_name>
   contains
     procedure :: init => init_<function_name>
     procedure :: eval => eval_<function_name>
  end type <function_name>

contains
  subroutine init_<function_name>(this)
    class(<function_name>), intent(out) :: this
    allocate(this%pars(<number_of_fitting_parameters>))
  end subroutine init_<function_name>

  type(advar) function eval_<function_name>(this, x) result(y)
    class(<function_name>), intent(in) :: this
    real(kp), intent(in) :: x
    <function_body>
  end function eval_<function_name>
end module <module_name>

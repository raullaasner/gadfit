!!****m* GADfit/messaging
!!
!! COPYRIGHT
!!
!! This Source Code Form is subject to the terms of the GNU General
!! Public License, v. 3.0. If a copy of the GPL was not distributed
!! with this file, You can obtain one at
!! http://gnu.org/copyleft/gpl.txt.
!!
!! FUNCTION
!!
!! Contains the error, warning, and comment procedures, the status
!! check of err_stat, and some I/O helper procedures.
!!
!! SOURCE
module messaging

  implicit none

  private
  public :: error, warning, comment, check_err, err_stat, err_msg, &
       & str, print_memory

  integer, parameter :: MAX_ERROR_LENGTH = 200
  integer :: err_stat = 0
  character(MAX_ERROR_LENGTH) :: err_msg

contains
  !!***

  !!****f* messaging/error
  !!
  !! FUNCTION
  !!
  !! Prints an error message. Stops execution on all images. Output is
  !! sent to stderr. 'file' and 'line' should be determined by the
  !! preprocessor.
  !!
  !! SOURCE
  subroutine error(file, line, msg)
    use, intrinsic :: iso_fortran_env, only: error_unit
    character(*), intent(in) :: file, msg
    integer, intent(in) :: line
    critical ! Often all images encounter an error at the same time.
      write(error_unit, '(/, *(g0))') file, ':', line, ':'
      call print_msg('ERROR', msg, error_unit)
      error stop
    end critical
  end subroutine error
  !!***

  !!****f* messaging/warning
  !!
  !! FUNCTION
  !!
  !! Same as error but the execution continues.
  !!
  !! SOURCE
  subroutine warning(file, line, msg)
    use, intrinsic :: iso_fortran_env, only: error_unit
    character(*), intent(in) :: file, msg
    integer, intent(in) :: line
    critical
      write(error_unit, '(/, *(g0))') file, ':', line, ':'
      call print_msg('WARNING', msg, error_unit)
    end critical
  end subroutine warning
  !!***

  !!****f* messaging/comment
  !!
  !! FUNCTION
  !!
  !! Same as warning but output is sent to stdout.
  !!
  !! SOURCE
  subroutine comment(file, line, msg)
    use, intrinsic :: iso_fortran_env, only: output_unit
    character(*), intent(in) :: file, msg
    integer, intent(in) :: line
    critical
      write(output_unit, '(/, *(g0))') file, ':', line, ':'
      call print_msg('COMMENT', msg, output_unit)
    end critical
  end subroutine comment
  !!***

  !!****f* messaging/print_msg
  !!
  !! FUNCTION
  !!
  !! Prints a message with the format
  !!   TYPE: <line with a max length of MAX_LINE_LEN>
  !!         <line with a max length of MAX_LINE_LEN>
  !!         ...
  !! The message can be split only at spaces or at the position
  !! MAX_LINE_LEN if there are no spaces. The indent is correctly kept
  !! for all lines.
  !!
  !! INPUTS
  !!
  !! typ - type of the message that will be displayed as TYPE
  !! msg - contents of the message
  !! io_unit - I/O device
  !!
  !! SOURCE
  recursive subroutine print_msg(typ, msg, io_unit)
    character(*), intent(in) :: typ, msg
    integer, intent(in) :: io_unit
    integer, parameter :: MAX_LINE_LEN = 50
    logical :: first_line = .true.
    integer :: i
    if (len(msg) <= MAX_LINE_LEN) then
       i = len(trim(msg))
    else if (index(msg(:MAX_LINE_LEN), ' ') > 0) then
       i = index(msg(:MAX_LINE_LEN), ' ', back=.true.)
    else
       i = MAX_LINE_LEN
    end if
    if (first_line) then
       write(io_unit, '(3x, *(g0))') typ, ': ', msg(:i)
       first_line = .false.
    else
       write(io_unit, '('//str(5+len(typ))//'x, g0)') msg(:i)
    end if
    if (len(msg) > MAX_LINE_LEN) then
       call print_msg(typ, msg(i+1:), io_unit)
    else
       first_line = .true. ! Reset for the next caller
    end if
  end subroutine print_msg
  !!***

  !!****f* misc/str
  !!
  !! FUNCTION
  !!
  !! Integer to string.
  !!
  !! SOURCE
  !!***
  pure function str(x) result(y)
    integer, intent(in) :: x
    character(:), allocatable :: y
    character(128) :: tmp
    write(tmp, '(g0)') x
    y = trim(tmp)
  end function str
  !!***

  !!****f* messaging/check_err
  !!
  !! FUNCTION
  !!
  !! Prints an error message if err_stat /= 0. It can be used with any
  !! procedure that has the status and error message specifiers, which
  !! have to point to err_stat and err_msg respectively. 'file' and
  !! 'line' should be determined by the preprocessor.
  !!
  !! SOURCE
  subroutine check_err(file, line)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    if (err_stat /= 0) call error(file, line, trim(err_msg))
  end subroutine check_err
  !!***

  !!****f* messaging/print_memory
  !!
  !! FUNCTION
  !!
  !! Prints memory in units of B, kB, MB, or GB depending on the size.
  !!
  !! INPUTS
  !!
  !! io_unit - I/O device
  !! x - amount of memory in bytes
  !!
  !! SOURCE
  subroutine print_memory(io_unit, x)
    integer, intent(in) :: io_unit, x
    character(9) :: memory
    if (x < 1e3) then
       write(memory, '(i4)') x
       write(io_unit, '(*(g0))', advance='no') trim(adjustl(memory)), ' B'
    else if (x < 1e6) then
       write(memory, '(f6.1)') x/1e3
       write(io_unit, '(*(g0))', advance='no') trim(adjustl(memory)), ' kB'
    else if (x < 1e9) then
       write(memory, '(f6.1)') x/1e6
       write(io_unit, '(*(g0))', advance='no') trim(adjustl(memory)), ' MB'
    else
       write(memory, '(f6.1)') x/1e9
       write(io_unit, '(*(g0))', advance='no') trim(adjustl(memory)), ' GB'
    end if
  end subroutine print_memory
  !!***
end module messaging

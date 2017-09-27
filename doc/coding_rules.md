Coding rules
============

Foreword
--------

This document contains a set of rules that a developer should follow when contributing to GADfit. Any suggestions about modifying the rules are welcome.


Programming principles
----------------------

* **Standard-conforming code.** This project is committed to following the Fortran 2008 standard (F2008), which is a minor improvement over F2003, but a major improvement over the commonly used F90, making coding much more efficient. It is not possible to list here all the rules that the developer is expected to follow when writing modern Fortran. There is not always a unique answer anyway. Many rules are expected to be self-evident. Here we concentrate on rules specific to this project that are not standard in general for modern Fortran. A good place to start is to examine an existing module in addition to reading this document.

* **Exception handling.** Use `check_err` after allocate and open statements. `safe_close` is optional and depends on factors such as the amount of data written. For deallocation use `safe_deallocate` and, if necessary, define a new subroutine for the `safe_deallocate` interface. For reporting errors, warnings, and comments, use the appropriate procedures from the messaging module.

* **Compilation stage.** There should be no compilation warnings when using the default debug flags except when due to a compiler bug. This requirement currently holds for the following compilers: GFortran.

* **Preprocessor.** Use preprocessor macros only for actual preprocessing and not for something that is a short-hand notation for a Fortran statement. Excessive use of macros makes it difficult for source code indexing tools to navigate through the source code.


Format
------

* **Statement order.** A module should have the following format:

  ```
  module <module_name>
  <'use' statements in alphabetical order>
  implicit none
  private
  protected :: <list>
  public :: <list>
  <enumerators>
  <constants>
  <derived type definitions>
  <interfaces>
  <variables>
  contains
  <procedures>
  end module <module_name>
  ```
  
  Protected variables should not be listed after the `public` statement, but declared public elsewhere. Also, the default `private` attribute is preferred, but it might not always make sense.

* **Order of arguments.** The order of arguments to a subprogram is preferably the following: the `pass` argument, `intent(in)` arguments, `intent(in out)` arguments, `intent(out)` arguments, `optional` arguments.

* **Case.** Use lower-case everywhere except for named constants, enumerators, and preprocessor keywords. Known mathematical constants and variables can be mixed-case if more readable this way.

* **Names.** Use self-documenting variable names. The larger the scope the more informative the name should be. Do not use the Hungarian notation. Unless there is a good reason to do otherwise, the result variable of a function should be `y` with the type specified in the `function` line.

* **Spaces.** Use spaces where allowed (`in out`, `end do`, etc.). Use one space around most binary operators.

* **Kind.** Do not use syntax such as `real*8` or `real(8)`. Make use of system constants such as `real64`.

* **Line length.** Try to limit the line length to 80 symbols. Not only is it good for readability, but the compiler can be more informative about the line number where the error occured. For continuation use ampersand on both lines.

* **Indentation.** 3 spaces for `do` and `if` constructs.


Documentation
-------------

* **In-code comments.** In general, comments should explain what the code does, not how it does it. In-code comments can be used to comment data or to explain the purpose of a code segment if not immediately clear from the context. Excessive in-code commenting suggests that the procedure probably needs to be redesigned.

* **Note.** When modifying the source code, don't forget to update the user guide accordingly!


Git
---

Use the following rules for creating Git commit messages:
* Separate subject from body with a blank line
* Limit the subject line to 50 characters
* Capitalize the subject line
* Do not end the subject line with a period
* Use the imperative mood in the subject line
* Wrap the body at 72 characters
* Use the body to explain what and why vs. how
Read the detailed instructions here: https://chris.beams.io/posts/git-commit


TODO
----

Some rule about procedure ordering in the contains section.

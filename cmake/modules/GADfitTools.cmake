# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License.  You
# may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.

# Convert a string into a list. If the variable was already given as
# list, do nothing.
macro(convert_to_list var)
  list(LENGTH ${var} _length)
  if (_length EQUAL 1)
    string(REPLACE " " ";" ${var} ${${var}})
  endif()
endmacro()

# Determine current flags
macro(get_all_flags language flags)
  set(${flags} ${CMAKE_${language}_FLAGS})
  if (CMAKE_BUILD_TYPE)
    string(TOUPPER ${CMAKE_BUILD_TYPE} buildtype)
    set(${flags} "${${flags}} ${CMAKE_${language}_FLAGS_${buildtype}}")
    string(STRIP "${${flags}}" ${flags})
  endif()
endmacro()

# Without this macro, the following statements would need to be
# duplicated in the Fortran/gadfit and C++/gadfit folders. It depends
# on the test target, which is defined separately fort Fortran and
# C++, but otherwise the statements are identical for both.
macro(include_coverage)
  if (INCLUDE_COVERAGE)
    find_program(_lcov lcov)
    if (_lcov)
      add_custom_target(coverage
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMAND lcov -c -d CMakeFiles -o tracefile.info
        COMMAND
        perl -ne 'print if s|^SF:\(?!${PROJECT_SOURCE_DIR}\)|\\1|'
        tracefile.info | xargs lcov --ignore-errors unused,unused -r
        tracefile.info -o tracefile_filtered.info
        COMMAND genhtml tracefile_filtered.info -o out
        COMMAND cmake -E echo
        "Open ${PROJECT_BINARY_DIR}/out/index.html to see the detailed report"
        DEPENDS test)
    else()
      add_custom_target(coverage
        COMMAND cmake -E echo "Error: Cannot run coverage. Install lcov first.")
    endif()
  else()
    add_custom_target(coverage
      COMMAND cmake -E echo
      "Error: Cannot run coverage. Enable INCLUDE_COVERAGE first.")
  endif()
endmacro()

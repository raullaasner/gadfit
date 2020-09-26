# This Source Code Form is subject to the terms of the GNU General
# Public License, v. 3.0. If a copy of the GPL was not distributed
# with this file, You can obtain one at
# http://gnu.org/copyleft/gpl.txt.

set(tests
  ad_basic
  ad_forward_mode
  ad_reverse_mode
  )

enable_testing()

find_package(Catch2 QUIET)
if (Catch2_FOUND)
  add_library(catch2_init OBJECT EXCLUDE_FROM_ALL catch2_init.cpp)
  target_link_libraries(catch2_init PUBLIC ${LIBRARIES} gadfit Catch2::Catch2)
  foreach(_test ${tests})
    add_executable(${_test} EXCLUDE_FROM_ALL ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE catch2_init)
    add_test(${_test} ${_test})
  endforeach()
  add_custom_target(test COMMAND ctest -V DEPENDS ${tests})
else()
  add_custom_target(test
    COMMAND cmake -E echo
    "Error: Testing is disabled because could not find Catch2."
    COMMAND cmake -E false)
endif()

include_coverage()

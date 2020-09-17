# This Source Code Form is subject to the terms of the GNU General
# Public License, v. 3.0. If a copy of the GPL was not distributed
# with this file, You can obtain one at
# http://gnu.org/copyleft/gpl.txt.

enable_testing()

add_executable(tests EXCLUDE_FROM_ALL
  catch2_init.cpp
  )

find_package(Catch2 QUIET)

if (Catch2_FOUND)
  target_link_libraries(tests PRIVATE ${LIBS} gadfit Catch2::Catch2)
  add_test(tests tests)
  add_custom_target(test COMMAND ctest -V DEPENDS tests)
else()
  add_custom_target(test
    COMMAND cmake -E echo
    "Error: Testing is disabled because could not find Catch2."
    COMMAND cmake -E false)
endif()

if (INCLUDE_COVERAGE)
  find_program(_lcov lcov)
  if (_lcov)
    set(lcov_excluded_directory /usr)
    add_custom_target(coverage
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      COMMAND lcov -c -d CMakeFiles -o cov.info
      COMMAND
      sed -nr 's|SF:${lcov_excluded_directory}|${lcov_excluded_directory}|p'
      cov.info | xargs lcov -r cov.info -o cov_filtered.info
      COMMAND genhtml cov_filtered.info -o out
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

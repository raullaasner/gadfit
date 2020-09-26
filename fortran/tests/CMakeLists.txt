# This Source Code Form is subject to the terms of the GNU General
# Public License, v. 3.0. If a copy of the GPL was not distributed
# with this file, You can obtain one at
# http://gnu.org/copyleft/gpl.txt.

set(tests
  ad_forward_mode
  ad_reverse_mode
  1_gaussian
  2_integral_single
  3_integral_double
  4_multiple_data
  )

enable_testing()

# Linear algebra libraries
if (LINALG_FLAVOR MATCHES "lapack")
  find_package(LAPACK REQUIRED)
  list(APPEND LIBRARIES ${LAPACK_LIBRARIES})
elseif (LINALG_FLAVOR MATCHES "atlas")
  find_package(Atlas REQUIRED)
  list(APPEND LIBRARIES ${Atlas_LIBRARIES})
elseif (LINALG_FLAVOR MATCHES "mkl")
  find_package(MKL REQUIRED)
  list(APPEND LIBRARIES ${MKL_LIBRARIES})
endif()
set(_test_linalg FALSE)
foreach(_var LINALG_FLAVOR LIBRARY_PATHS LIBRARIES)
  if (DEFINED ${_var} AND NOT ${_var}_prev STREQUAL ${_var})
    set(${_var}_prev "${${_var}}" CACHE INTERNAL "")
    set(_test_linalg TRUE)
  endif()
endforeach()
if (_test_linalg)
  try_compile(LINALG_TEST ${CMAKE_CURRENT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/cmake/tests/Linalg.f90
    CMAKE_FLAGS
    "-DLINK_LIBRARIES=${LIBRARIES}"
    "-DLINK_DIRECTORIES=${LIBRARY_PATHS}")
  set(linalg_ok ${LINALG_TEST} CACHE INTERNAL "")
  if (LINALG_TEST)
    message(STATUS "Linear algebra library is OK")
  else()
    message(STATUS "Fallback linear algebra library is used for tests.")
  endif()
endif()
if (NOT linalg_ok)
  add_library(lapack-fallback SHARED EXCLUDE_FROM_ALL lapack-fallback.f)
  list(APPEND LIBRARIES lapack-fallback)
endif()

# OpenCoarrays
if (NOT OPEN_COARRAYS AND CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  include(ExternalProject)
  ExternalProject_Add(
    OpenCoarrays-fallback
    EXCLUDE_FROM_ALL 1
    URL
    https://github.com/sourceryinstitute/opencoarrays/archive/2.8.0.tar.gz
    CMAKE_CACHE_ARGS
    -DCMAKE_Fortran_COMPILER:STRING=${CMAKE_Fortran_COMPILER}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_BUILD_TYPE:STRING=Release
    CMAKE_GENERATOR "Unix Makefiles"
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --target caf_mpi_static
    INSTALL_COMMAND ${CMAKE_COMMAND} -E
    copy lib/libcaf_mpi.a ${CMAKE_CURRENT_BINARY_DIR}
    BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/libcaf_mpi.a
    )
  list(APPEND LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/libcaf_mpi.a)
endif()

# GADfit
list(APPEND LIBRARIES gadfit)

# Example input
add_executable(example0 EXCLUDE_FROM_ALL example.F90)
target_compile_definitions(example0 PRIVATE
  DATA_DIR='${CMAKE_CURRENT_SOURCE_DIR}')
add_dependencies(example0 gadfit)
if (TARGET lapack-fallback)
  add_dependencies(example0 lapack-fallback)
endif()
if (TARGET OpenCoarrays-fallback)
  add_dependencies(example0 OpenCoarrays-fallback)
endif()
target_include_directories(example0 PRIVATE ${PROJECT_BINARY_DIR})
target_link_libraries(example0 ${LIBRARIES})
add_custom_target(example
  COMMAND ${TESTS_RUN_PREFIX} ${CMAKE_CURRENT_BINARY_DIR}/example0
  DEPENDS example0)

# Tests
add_library(fortran_testing EXCLUDE_FROM_ALL testing.F90)
target_link_libraries(fortran_testing PUBLIC ${LIBRARIES})
foreach(f ${tests})
  add_executable(${f} EXCLUDE_FROM_ALL ${f}.F90)
  # If a test requires input data, include that in a separate module.
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${f}_data.F90)
    target_sources(${f} PRIVATE ${f}_data.F90)
  endif()
  if (TARGET lapack-fallback)
    add_dependencies(${f} lapack-fallback)
  endif()
  if (TARGET OpenCoarrays-fallback)
    add_dependencies(${f} OpenCoarrays-fallback)
  endif()
  add_test(NAME ${f} COMMAND ${TESTS_RUN_PREFIX}
    ${CMAKE_CURRENT_BINARY_DIR}/${f})
  target_include_directories(${f}
    PRIVATE ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(${f} fortran_testing)
endforeach()
add_custom_target(test
  COMMAND ctest -V DEPENDS ${tests})

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

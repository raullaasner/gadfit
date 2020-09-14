# This Source Code Form is subject to the terms of the GNU General
# Public License, v. 3.0. If a copy of the GPL was not distributed
# with this file, You can obtain one at
# http://gnu.org/copyleft/gpl.txt.

set(TESTS
  1_gaussian
  2_integral_single
  3_integral_double)

enable_testing()

# Linear algebra libraries
if (LINALG_FLAVOR MATCHES "lapack")
  find_package(LAPACK REQUIRED)
  list(APPEND LIBS ${LAPACK_LIBRARIES})
elseif (LINALG_FLAVOR MATCHES "atlas")
  find_package(Atlas REQUIRED)
  list(APPEND LIBS ${Atlas_LIBRARIES})
elseif (LINALG_FLAVOR MATCHES "mkl")
  find_package(MKL REQUIRED)
  list(APPEND LIBS ${MKL_LIBRARIES})
endif()
set(_test_linalg FALSE)
foreach(_var LINALG_FLAVOR LIB_PATHS LIBS)
  if (DEFINED ${_var} AND NOT ${_var}_prev STREQUAL ${_var})
    set(${_var}_prev "${${_var}}" CACHE INTERNAL "")
    set(_test_linalg TRUE)
  endif()
endforeach()
if (_test_linalg)
  try_compile(LINALG_TEST ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/tests/Linalg.f90
    CMAKE_FLAGS
    "-DLINK_LIBRARIES=${LIBS}"
    "-DLINK_DIRECTORIES=${LIB_PATHS}")
  set(linalg_ok ${LINALG_TEST} CACHE INTERNAL "")
  if (LINALG_TEST)
    message(STATUS "Linear algebra library is OK")
  else()
    message(STATUS "Fallback linear algebra library is used for tests.")
  endif()
endif()
if (NOT linalg_ok)
  add_library(lapack-fallback SHARED EXCLUDE_FROM_ALL lapack-fallback.f)
  list(APPEND LIBS lapack-fallback)
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
  list(APPEND LIBS ${CMAKE_CURRENT_BINARY_DIR}/libcaf_mpi.a)
endif()

# GADfit
list(APPEND LIBS gadfit)

# Example input
add_executable(example0 EXCLUDE_FROM_ALL example.F90)
add_dependencies(example0 gadfit)
if (TARGET lapack-fallback)
  add_dependencies(example0 lapack-fallback)
endif()
if (TARGET OpenCoarrays-fallback)
  add_dependencies(example0 OpenCoarrays-fallback)
endif()
target_include_directories(example0 PRIVATE ${PROJECT_BINARY_DIR})
target_link_libraries(example0 ${LIBS})
set(SRC ${CMAKE_CURRENT_SOURCE_DIR})
set(BLD ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(example
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${SRC}/example_data1 ${BLD}/example_data1
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${SRC}/example_data2 ${BLD}/example_data2
  COMMAND ${TESTS_RUN_PREFIX} ${BLD}/example0 DEPENDS example0)

# Tests
foreach(f ${TESTS})
  add_executable(${f} EXCLUDE_FROM_ALL ${f}.F90)
  if (TARGET lapack-fallback)
    add_dependencies(${f} lapack-fallback)
  endif()
  if (TARGET OpenCoarrays-fallback)
    add_dependencies(${f} OpenCoarrays-fallback)
  endif()
  add_test(NAME ${f} COMMAND ${TESTS_RUN_PREFIX}
    ${CMAKE_CURRENT_BINARY_DIR}/${f})
  target_include_directories(${f} PRIVATE ${PROJECT_BINARY_DIR})
  target_link_libraries(${f} ${LIBS})
  set_tests_properties(${f} PROPERTIES FAIL_REGULAR_EXPRESSION
    "(Error|BAD TERMINATION)")
endforeach()

add_custom_target(test
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${SRC}/1_gaussian_data ${BLD}/1_gaussian_data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${SRC}/2_integral_single_data ${BLD}/2_integral_single_data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${SRC}/3_integral_double_data ${BLD}/3_integral_double_data
  COMMAND ctest -V DEPENDS ${TESTS})

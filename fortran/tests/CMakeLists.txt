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

set(tests
  ad_forward_mode
  ad_reverse_mode
  1_gaussian
  2_integral_single
  3_integral_double
  4_multiple_curves
  )

enable_testing()

# Linear algebra libraries
try_compile(LINALG_TEST ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/cmake/tests/Linalg.f90
  CMAKE_FLAGS
  "-DLINK_LIBRARIES=${LIBRARIES}"
  "-DLINK_DIRECTORIES=${LIBRARY_PATHS}")
if (LINALG_TEST)
  message(STATUS "Linear algebra library is OK")
else()
  message(STATUS "Fallback linear algebra library is used for tests.")
  add_library(lapack-fallback SHARED EXCLUDE_FROM_ALL lapack-fallback.f)
  list(APPEND LIBRARIES lapack-fallback)
endif()

# OpenCoarrays
if (NOT OPEN_COARRAYS AND CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  include(ExternalProject)
  if (NOT CMAKE_VERSION VERSION_LESS 3.16.9)
    set(_caf_lib_dir lib)
  else()
    set(_caf_lib_dir lib64)
  endif()
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
    copy ${_caf_lib_dir}/libcaf_mpi.a ${CMAKE_CURRENT_BINARY_DIR}
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
if (TARGET OpenCoarrays-fallback)
  add_dependencies(fortran_testing OpenCoarrays-fallback)
endif()
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
  add_test(NAME ${f} COMMAND ${TESTS_RUN_PREFIX}
    ${CMAKE_CURRENT_BINARY_DIR}/${f})
  target_include_directories(${f}
    PRIVATE ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(${f} fortran_testing)
endforeach()
add_custom_target(test
  COMMAND ctest -V DEPENDS ${tests})

include_coverage()

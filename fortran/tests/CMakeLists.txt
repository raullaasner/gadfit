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

# Linear algebra library for tests
if (FALLBACK_LINALG)
  add_library(lapack-fallback SHARED EXCLUDE_FROM_ALL lapack-fallback.f)
  list(APPEND LINK_LIBRARIES lapack-fallback)
  message("Using fallback linear algebra routines")
else()
  message("Using optimized linear algebra library")
endif()

# Coarray runtime library
if (NOT CAF_LIBRARY AND CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  # GFortran ships libcaf_single (a single image) and, since GCC 15,
  # libcaf_shmem (multiple images using shared memory). Default to the
  # former, which is available with all GFortran versions.
  list(APPEND LINK_LIBRARIES caf_single)
endif()

# GADfit
list(APPEND LINK_LIBRARIES gadfit)

# Example input
add_executable(example0 EXCLUDE_FROM_ALL example.F90)
target_compile_definitions(example0 PRIVATE
  DATA_DIR='${CMAKE_CURRENT_SOURCE_DIR}')
add_dependencies(example0 gadfit)
if (TARGET lapack-fallback)
  add_dependencies(example0 lapack-fallback)
endif()
target_include_directories(example0 PRIVATE ${PROJECT_BINARY_DIR})
target_link_libraries(example0 ${LINK_LIBRARIES})
add_custom_target(example
  COMMAND ${TESTS_RUN_PREFIX} ${CMAKE_CURRENT_BINARY_DIR}/example0
  DEPENDS example0)

# Tests
add_library(fortran_testing EXCLUDE_FROM_ALL testing.F90)
target_link_libraries(fortran_testing PUBLIC ${LINK_LIBRARIES})
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

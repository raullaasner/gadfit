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
  ad_basic
  ad_forward_mode
  ad_reverse_mode
  lm_solver
  numerical_integration
  )

if (USE_MPI)
  list(APPEND tests mpi_communicators)
endif()

if (USE_SCALAPACK)
  list(APPEND tests scalapack)
endif()

enable_testing()

add_library(catch2_init OBJECT EXCLUDE_FROM_ALL catch2_init.cpp)

find_package(Catch2 3.0.1 QUIET)
if (NOT Catch2_FOUND)
  if (first_run)
    message("Catch2 >= 3.0.1 not found, downloading")
  endif()
  FetchContent_Declare(catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2
    GIT_TAG 97c48e0c343d26d50764fafdc90b1e630fbd10ce)
  FetchContent_MakeAvailable(catch2)
  target_link_libraries(catch2_init PUBLIC ${LIBRARIES} gadfit Catch2)
else()
  target_link_libraries(catch2_init PUBLIC ${LIBRARIES} gadfit Catch2::Catch2)
endif()

foreach(_test ${tests})
  add_executable(${_test} EXCLUDE_FROM_ALL ${_test}.cpp)
  target_link_libraries(${_test} PRIVATE catch2_init)
  if (USE_MPI)
    add_test(NAME ${_test} COMMAND
      ${TESTS_RUN_PREFIX}
      ${CMAKE_CURRENT_BINARY_DIR}/${_test})
  else()
    add_test(${_test} ${_test})
  endif()
  find_package(spdlog QUIET)
  if (NOT TARGET spdlog::spdlog)
    target_include_directories(${_test} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}\
/../../c++/gadfit/spdlog-prefix/src/spdlog/include")
  endif()
endforeach()
add_custom_target(test COMMAND ctest --output-on-failure DEPENDS ${tests})

# Example input
add_executable(example0 EXCLUDE_FROM_ALL example.cpp)
add_dependencies(example0 gadfit)
target_link_libraries(example0 PRIVATE gadfit)
add_custom_target(example
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/example0
  DEPENDS example0)

include_coverage()

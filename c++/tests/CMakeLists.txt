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

enable_testing()

find_package(Catch2 QUIET)
if (Catch2_FOUND)
  add_library(catch2_init OBJECT EXCLUDE_FROM_ALL catch2_init.cpp)
  target_link_libraries(catch2_init PUBLIC ${LIBRARIES} gadfit Catch2::Catch2)
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
else()
  add_custom_target(test
    COMMAND cmake -E echo
    "Error: Testing is disabled because could not find Catch2."
    COMMAND cmake -E false)
endif()

# Example input
add_executable(example0 EXCLUDE_FROM_ALL example.cpp)
add_dependencies(example0 gadfit)
target_link_libraries(example0 PRIVATE gadfit)
add_custom_target(example
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/example0
  DEPENDS example0)

include_coverage()

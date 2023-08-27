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

# List test source files here without the extension
set(tests
  ad_basic
  ad_forward_mode
  ad_reverse_mode
  lm_solver
  numerical_integration
)

enable_testing()

set(CATCH2_PATH "" CACHE STRING "Path to Catch2 installation directory")
find_package(Catch2 3.3.2 QUIET HINTS ${CATCH2_PATH})
if (Catch2_FOUND)
  get_target_property(_loc Catch2::Catch2 INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "Found ${_loc}")
else()
  if (first_run)
    message("Catch2 >= 3.3.2 not found, downloading")
  endif()
  FetchContent_Declare(catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2 GIT_TAG v3.3.2)
  FetchContent_MakeAvailable(catch2)
endif()

foreach(_test ${tests})
  add_executable(${_test} EXCLUDE_FROM_ALL ${_test}.cpp)
  target_link_libraries(${_test} PRIVATE gadfit Catch2::Catch2WithMain)
  add_test(${_test} ${_test})
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

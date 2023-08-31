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

# CMake
cmake_minimum_required(VERSION 3.13.5)

# Project
project(GADfit VERSION 2.0 LANGUAGES CXX)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# Options and variables
option(BUILD_CXX_VERSION
  "If enabled, build the C++ version of GADFit, else the Fortran version" ON)
if (NOT BUILD_CXX_VERSION)
  enable_language(Fortran)
endif()
option(BUILD_SHARED_LIBS "If true, build a shared library, else static" ON)

set(LINK_LIBRARIES "" CACHE STRING
  "Libraries to be linked against when running tests")
option(FALLBACK_LINALG
  "Whether to compile with fallback linear algebra library" ON)
option(INCLUDE_COVERAGE "Define a new target called 'coverage'" OFF)
set(COVERAGE_FLAGS "--coverage" CACHE STRING
  "Additional compiler flags used when running code coverage")
if (INCLUDE_COVERAGE)
  string(APPEND CMAKE_CXX_FLAGS " ${COVERAGE_FLAGS}")
  string(APPEND CMAKE_Fortran_FLAGS " ${COVERAGE_FLAGS}")
endif()

# The first_run variable is useful when attempting to detect the
# default value of a variable, but only during the first configuring
# and only if not already specified by the user (e.g., in the initial
# cache file).
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  set(first_run TRUE)
else()
  set(first_run FALSE)
endif()

# Define main target
add_library(gadfit "")
target_compile_definitions(gadfit PRIVATE
  PROJECT_VERSION="${PROJECT_VERSION}"
  CMAKE_PLATFORM="${CMAKE_HOST_SYSTEM}")

# Print info
include(GADfitTools)
if (BUILD_CXX_VERSION)
  message(
    "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  get_all_flags(CXX all_flags)
else()
  message(
    "Fortran compiler: ${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}")
  get_all_flags(Fortran all_flags)
endif()
message("Compiler flags: ${all_flags}")
message("Build configuration: ${CMAKE_BUILD_TYPE}")
if (BUILD_SHARED)
  message("Building a shared library")
else()
  message("Building a static library")
endif()
message("Install location: ${CMAKE_INSTALL_PREFIX}")

# Sources
if (BUILD_CXX_VERSION)
  add_subdirectory(c++/gadfit)
  if (NOT TARGET test)
    add_subdirectory(c++/tests)
  endif()
else()
  add_subdirectory(fortran/gadfit)
  add_subdirectory(fortran/tests)
endif()

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

target_sources(gadfit PRIVATE
  automatic_differentiation.cpp
  fit_function.cpp
  lm_solver.cpp
  exceptions.cpp
  parallel.cpp
  )

# First run
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  # Automatically detect USE_MPI
  try_compile(MPI_OK ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/cmake/tests/MPI.cpp)
  option(USE_MPI "Whether to compile with MPI support" ${MPI_OK})
endif()

# Options and variables (C++ only)
set(SPDLOG_PATH "" CACHE STRING "Path to spdlog installation directory")

# Linear algebra library
target_sources(gadfit PRIVATE cblas_wrapper_fallback.cpp)

# MPI
if (USE_MPI)
  target_compile_definitions(gadfit PUBLIC USE_MPI)
endif()

# Logging
find_package(spdlog HINTS ${SPDLOG_PATH})
if (TARGET spdlog::spdlog)
  target_link_libraries(gadfit PUBLIC spdlog::spdlog)
else()
  message(WARNING "spdlog not found. Consider installing spdlog using your "
    "system's package manager. Using a fallback version for now.")
  include(ExternalProject)
  ExternalProject_Add(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.8.2
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND "")
  add_dependencies(gadfit spdlog)
  install(DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/spdlog-prefix/src/spdlog/include/spdlog
    DESTINATION
    include/gadfit)
  set(_inc_dir ${CMAKE_CURRENT_BINARY_DIR}/spdlog-prefix/src/spdlog/include)
  target_include_directories(gadfit
    PRIVATE $<BUILD_INTERFACE:${_inc_dir}>
    PUBLIC $<INSTALL_INTERFACE:include/gadfit>)
endif()

# C++20 features
target_compile_features(gadfit PUBLIC cxx_std_20)

# Exporting
target_include_directories(gadfit INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
set(_headers
  lm_solver.h automatic_differentiation.h exceptions.h fit_function.h
  parallel.h)
set_target_properties(gadfit PROPERTIES PUBLIC_HEADER "${_headers}")
install(TARGETS gadfit
  EXPORT gadfitConfig
  PUBLIC_HEADER DESTINATION include/gadfit)
install(EXPORT gadfitConfig NAMESPACE gadfit:: DESTINATION lib/cmake/gadfit)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/gadfitConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES ${PROJECT_BINARY_DIR}/gadfitConfigVersion.cmake
  DESTINATION lib/cmake/gadfit)
export(EXPORT gadfitConfig NAMESPACE gadfit::
  FILE ${PROJECT_BINARY_DIR}/gadfitConfig.cmake)

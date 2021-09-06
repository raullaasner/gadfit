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
  numerical_integration.cpp
  timer.cpp
  )

# C++20 features
target_compile_features(gadfit PUBLIC cxx_std_20)

# First run
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  # Automatically detect USE_MPI
  try_compile(MPI_OK ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/cmake/tests/MPI.cpp)
  option(USE_MPI "Whether to compile with MPI support" ${MPI_OK})
endif()

# MPI
if (USE_MPI)
  target_compile_definitions(gadfit PUBLIC USE_MPI)
  message("Using MPI")
else()
  message("Not using MPI")
endif()

# Options and variables (C++ only)
set(SPDLOG_PATH "" CACHE STRING "Path to spdlog installation directory")

# Determine linear algebra library
set(linalg_library_name "fallback") # Start with the fallback
# Next test for anything that passes the linear algebra test
try_compile(LINALG_TEST ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/cmake/tests/Linalg.cpp
  CMAKE_FLAGS
  "-DINCLUDE_DIRECTORIES=${INCLUDE_PATHS}"
  "-DLINK_DIRECTORIES=${LIBRARY_PATHS}"
  "-DLINK_LIBRARIES=${LIBRARIES}")
if (LINALG_TEST)
  set(linalg_library_name "lapack")
endif()
# Next test for MKL
try_compile(LINALG_TEST ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/cmake/tests/MKL.cpp
  CMAKE_FLAGS
  "-DINCLUDE_DIRECTORIES=${INCLUDE_PATHS}"
  "-DLINK_DIRECTORIES=${LIBRARY_PATHS}"
  "-DLINK_LIBRARIES=${LIBRARIES}")
if (LINALG_TEST)
  set(linalg_library_name "MKL")
endif()
if (linalg_library_name STREQUAL "MKL")
  target_sources(gadfit PRIVATE cblas_wrapper_mkl.cpp)
elseif (linalg_library_name STREQUAL "lapack")
  target_sources(gadfit PRIVATE cblas_wrapper_lapack.cpp)
else()
  target_sources(gadfit PRIVATE cblas_wrapper_fallback.cpp)
endif()
message("Using linear algebra library: ${linalg_library_name}")

# Logging
include(FetchContent)
find_package(spdlog HINTS ${SPDLOG_PATH})
if (NOT TARGET spdlog::spdlog)
  FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG bae78f7b6c32fbfb97dd5751591734c12103fc1a)
  FetchContent_MakeAvailable(spdlog)
  install(DIRECTORY ${spdlog_SOURCE_DIR}/include/spdlog DESTINATION include)
  install(TARGETS spdlog EXPORT gadfitConfig)
endif()
target_link_libraries(gadfit PUBLIC spdlog::spdlog)

# Exporting
target_include_directories(gadfit INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
set(_headers
  lm_solver.h automatic_differentiation.h exceptions.h fit_function.h
  parallel.h numerical_integration.h timer.h)
set_target_properties(gadfit PROPERTIES PUBLIC_HEADER "${_headers}")
if (NOT CMAKE_VERSION VERSION_LESS 3.14.7)
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
endif()

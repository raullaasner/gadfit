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
  numerical_integration.cpp
  timer.cpp
)

# C++20 features
target_compile_features(gadfit PUBLIC cxx_std_20)

find_package(OpenMP QUIET)
if (OpenMP_CXX_FOUND)
  message("Using OpenMP")
  target_link_libraries(gadfit PUBLIC OpenMP::OpenMP_CXX)
else()
  message("Not using OpenMP")
endif()

# Linear algebra library for tests
if (FALLBACK_LINALG)
  target_sources(gadfit PRIVATE lapack_fallback.cpp)
  message("Using fallback linear algebra routines")
else()
  target_sources(gadfit PRIVATE lapack.cpp)
  message("Using optimized linear algebra library")
endif()

# Logging
include(FetchContent)
set(SPDLOG_PATH "" CACHE STRING "Path to spdlog installation directory")
find_package(spdlog HINTS ${SPDLOG_PATH})
if (NOT TARGET spdlog::spdlog)
  if (first_run)
    message("spdlog not found, downloading")
  endif()
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
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<INSTALL_INTERFACE:include>)
set(_headers
  lm_solver.h automatic_differentiation.h exceptions.h fit_function.h
  numerical_integration.h timer.h)
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

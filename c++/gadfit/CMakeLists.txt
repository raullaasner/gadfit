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
  )

# Linear algebra library
target_sources(gadfit PRIVATE cblas_wrapper_fallback.cpp)

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
    INSTALL_COMMAND ""
    )
  add_dependencies(gadfit spdlog)
  target_include_directories(gadfit PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/spdlog-prefix/src/spdlog/include)
endif()

# C++20 features
target_compile_features(gadfit PUBLIC cxx_std_20)

# Exporting
set_target_properties(gadfit PROPERTIES PUBLIC_HEADER
  "automatic_differentiation.h;lm_solver.h"
  )
target_include_directories(gadfit INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
install(TARGETS gadfit EXPORT gadfitConfig)
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

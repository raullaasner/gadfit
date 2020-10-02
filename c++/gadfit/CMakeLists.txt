# This Source Code Form is subject to the terms of the GNU General
# Public License, v. 3.0. If a copy of the GPL was not distributed
# with this file, You can obtain one at
# http://gnu.org/copyleft/gpl.txt.

target_sources(gadfit PRIVATE
  automatic_differentiation.cpp
  exceptions.cpp
  )

target_compile_features(gadfit PUBLIC cxx_std_20)

set_target_properties(gadfit PROPERTIES PUBLIC_HEADER
  automatic_differentiation.h
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

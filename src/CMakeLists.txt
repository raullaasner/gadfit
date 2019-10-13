# This Source Code Form is subject to the terms of the GNU General
# Public License, v. 3.0. If a copy of the GPL was not distributed
# with this file, You can obtain one at
# http://gnu.org/copyleft/gpl.txt.

target_sources(gadfit PRIVATE
  automatic_differentiation.F90
  fitfunction.F90
  gadf_constants.F90
  gadfit.F90
  gauss_kronrod_parameters.F90
  gadfit_linalg.F90
  messaging.f90
  misc.F90
  numerical_integration.F90
  )

install(TARGETS gadfit EXPORT gadfitConfig)
install(DIRECTORY ${PROJECT_BINARY_DIR}/include/ DESTINATION include)
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

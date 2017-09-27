# This Source Code Form is subject to the terms of the GNU General
# Public License, v. 3.0. If a copy of the GPL was not distributed
# with this file, You can obtain one at
# http://gnu.org/copyleft/gpl.txt.

set(gadfit_srcs
  automatic_differentiation.f90
  fitfunction.f90
  gadf_constants.f90
  gadfit.f90
  gauss_kronrod_parameters.f90
  linalg.f90
  messaging.f90
  misc.f90
  numerical_integration.f90)

if (BUILD_SHARED)
  add_library(gadfit SHARED ${gadfit_srcs})
else()
  add_library(gadfit STATIC ${gadfit_srcs})
endif()

install(TARGETS gadfit DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ DESTINATION include
  FILES_MATCHING PATTERN "*.mod" PATTERN CMakeFiles EXCLUDE)

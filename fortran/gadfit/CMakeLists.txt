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

target_include_directories(gadfit PRIVATE ${PROJECT_BINARY_DIR})

target_compile_definitions(gadfit PRIVATE
  Fortran_COMPILER_ID='${CMAKE_Fortran_COMPILER_ID}'
  Fortran_COMPILER_VERSION='${CMAKE_Fortran_COMPILER_VERSION}')
foreach(_flag QUAD_PRECISION HAS_CO_SUM GSL_DIR)
  if (${_flag})
    target_compile_definitions(gadfit PRIVATE ${_flag})
  endif()
endforeach()

set_target_properties(gadfit PROPERTIES
  Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)
target_include_directories(gadfit INTERFACE
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

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

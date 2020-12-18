# This Source Code Form is subject to the terms of the GNU General
# Public License, v. 3.0. If a copy of the GPL was not distributed
# with this file, You can obtain one at
# http://gnu.org/copyleft/gpl.txt.

# CMAKE
cmake_minimum_required(VERSION 3.13)

# PROJECT
project(GADfit VERSION 1.3 LANGUAGES C)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# OPTIONS AND VARIABLES
option(BUILD_CXX_VERSION
  "If enabled, build the C++ version of GADFit, else the Fortran version" ON)
if (BUILD_CXX_VERSION)
  enable_language(CXX)
else()
  enable_language(Fortran)
endif()
option(USE_AD
  "If disabled, use finite differences (Fortran only)" ON)
option(BUILD_SHARED_LIBS
  "If enabled, builds a shared library, else static" ON)
set(TESTS_RUN_PREFIX "" CACHE STRING
  "Prefix used for running tests, e.g., 'mpirun ...'")
set(LINALG_FLAVOR "" CACHE STRING
  "Choose the linear algebra library (lapack, atlas, mkl)")
set(INCLUDE_PATHS "" CACHE STRING
  "Directories containing necessary header files")
set(LIBRARY_PATHS "" CACHE STRING
  "Directories containing libraries to be linked against")
set(LIBRARIES "" CACHE STRING
  "Libraries to be linked against")
set(SPDLOG_PATH "" CACHE STRING
  "Path to spdlog installation directory")
set(GSL_DIR "" CACHE STRING "GSL install directory")
option(INCLUDE_COVERAGE "Defines new target called 'coverage'" OFF)
set(COVERAGE_FLAGS "--coverage" CACHE STRING
  "Additional compiler flags used when running code coverage")
if (INCLUDE_COVERAGE)
  if (BUILD_CXX_VERSION)
    string(APPEND CMAKE_CXX_FLAGS " ${COVERAGE_FLAGS}")
  else()
    string(APPEND CMAKE_Fortran_FLAGS " ${COVERAGE_FLAGS}")
  endif()
endif()

# DEFINE TARGET
add_library(gadfit "")
target_compile_definitions(gadfit PRIVATE
  PROJECT_VERSION="${PROJECT_VERSION}"
  CMAKE_PLATFORM="${CMAKE_HOST_SYSTEM}")

# PROCESS USER DEFINED INCLUDE PATHS AND LIBRARIES
include(Tools)
target_include_directories(gadfit PRIVATE ${INCLUDE_PATHS})
# Convert variables to list first
foreach(_var LIBRARIES LIBRARY_PATHS INCLUDE_PATHS)
  convert_to_list(${_var})
endforeach()
generate_library_targets(LIBRARY_PATHS LIBRARIES)
target_link_libraries(gadfit PRIVATE ${LIBRARIES})
if (GSL_DIR)
  target_include_directories(gadfit PRIVATE ${GSL_DIR}/include/gsl)
  unset(GSL_INCLUDE_DIR CACHE)
  unset(GSL_LIBRARY CACHE)
  unset(GSLCBLAS_LIBRARY CACHE)
  find_path(GSL_INCLUDE_DIR gsl_cblas.h ${GSL_DIR}/include/gsl)
  find_library(GSL_LIBRARY gsl ${GSL_DIR}/lib NO_DEFAULT_PATH)
  find_library(GSLCBLAS_LIBRARY gslcblas ${GSL_DIR}/lib NO_DEFAULT_PATH)
  mark_as_advanced(GSL_INCLUDE_DIR GSL_LIBRARY GSLCBLAS_LIBRARY)
  if (GSL_INCLUDE_DIR AND GSL_LIBRARY AND GSLCBLAS_LIBRARY)
    message(STATUS "Found ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY}")
    target_link_libraries(gadfit PRIVATE ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
  else()
    set(GSL_DIR FALSE)
  endif()
endif()
option(QUAD_PRECISION
  "Whether to use quadruple or double precision (Fortran only)" OFF)
set(OPEN_COARRAYS "" CACHE STRING "Full path to libcaf_mpi.a")
if (OPEN_COARRAYS)
  target_link_libraries(gadfit PRIVATE ${OPEN_COARRAYS})
endif()

# COMPILER CAPABILITIES
set(HAS_CO_SUM TRUE)
if (CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  set(HAS_CO_SUM FALSE)
endif()

# SOURCES
if (BUILD_CXX_VERSION)
  add_subdirectory(c++/gadfit)
  add_subdirectory(c++/tests)
else()
  add_subdirectory(fortran/gadfit)
  add_subdirectory(fortran/tests)
endif()

# SUMMARY
message("")
message("Summary")
message("=======")
if (BUILD_CXX_VERSION)
  message("C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  get_all_flags(CXX all_flags)
else()
  message("Fortran compiler: ${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}")
  get_all_flags(Fortran all_flags)
endif()
message("Compiler flags: ${all_flags}")
message("Build configuration: ${CMAKE_BUILD_TYPE}")
if (BUILD_SHARED)
  message("Building a shared library")
else()
  message("Building a static library")
endif()
if (NOT BUILD_CXX_VERSION)
  if (QUAD_PRECISION)
    message("Using quadruple precision")
  else()
    message("Using double precision")
  endif()
  if (USE_AD)
    message("Using automatic differentiation")
  else()
    message("Using finite differences")
  endif()
endif()
if (GSL_DIR)
  message("Using GSL")
else()
  message("Not using GSL")
endif()
message("Install location: ${CMAKE_INSTALL_PREFIX}")
message("")

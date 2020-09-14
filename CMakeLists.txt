# This Source Code Form is subject to the terms of the GNU General
# Public License, v. 3.0. If a copy of the GPL was not distributed
# with this file, You can obtain one at
# http://gnu.org/copyleft/gpl.txt.

# CMAKE
cmake_minimum_required(VERSION 3.13)

# PROJECT
project(GADfit VERSION 1.3 LANGUAGES Fortran C)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# OPTIONS
option(USE_AD
  "If enabled, uses automatic differentiation, else finite differences" ON)
option(BUILD_SHARED_LIBS "If enabled, builds a shared library, else static." ON)
set(TESTS_RUN_PREFIX "" CACHE STRING
  "Prefix used for running tests, e.g., 'mpirun ...'")
set(LINALG_FLAVOR "" CACHE STRING
  "Choose the linear algebra library (lapack, atlas, mkl)")
set(INC_PATHS "" CACHE STRING
  "Directories containing necessary header files")
set(LIB_PATHS "" CACHE STRING
  "Directories containing libraries to be linked against")
set(LIBS "" CACHE STRING
  "Libraries to be linked against")
# Convert to list first
foreach(_var LIBS LIB_PATHS INC_PATHS)
  if (${_var})
    list(LENGTH ${_var} _length)
    if (_length EQUAL 1)
      string(REPLACE " " ";" ${_var} ${${_var}})
    endif()
  endif()
endforeach()
# Turn entries of LIBS into targets
foreach(_lib ${LIBS})
  if (NOT TARGET ${_lib})
    find_library(LIB_FULLPATH ${_lib} HINTS ${LIB_PATHS})
    if (LIB_FULLPATH)
      message(STATUS "Found ${LIB_FULLPATH}")
      add_library(${_lib} UNKNOWN IMPORTED)
      set_target_properties(${_lib} PROPERTIES
        IMPORTED_LOCATION ${LIB_FULLPATH})
      unset(LIB_FULLPATH CACHE)
    else()
      message(FATAL_ERROR "${Magenta}Could not find ${_lib}${ColorReset}")
    endif()
  endif()
endforeach()
set(GSL_DIR "" CACHE STRING "GSL install directory")
if (GSL_DIR)
  unset(GSL_INCLUDE_DIR CACHE)
  unset(GSL_LIBRARY CACHE)
  unset(GSLCBLAS_LIBRARY CACHE)
  find_path(GSL_INCLUDE_DIR gsl_cblas.h ${GSL_DIR}/include/gsl)
  find_library(GSL_LIBRARY gsl ${GSL_DIR}/lib NO_DEFAULT_PATH)
  find_library(GSLCBLAS_LIBRARY gslcblas ${GSL_DIR}/lib NO_DEFAULT_PATH)
  mark_as_advanced(GSL_INCLUDE_DIR GSL_LIBRARY GSLCBLAS_LIBRARY)
  if (GSL_INCLUDE_DIR AND GSL_LIBRARY AND GSLCBLAS_LIBRARY)
    message(STATUS "Found ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY}")
    list(APPEND LIBS ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
  else()
    set(GSL_DIR FALSE)
  endif()
endif()
option(QUAD_PRECISION
  "If enabled, uses quadruple precision, else double precision" OFF)
set(OPEN_COARRAYS "" CACHE STRING "Full path to libcaf_mpi.a")
list(APPEND LIBS ${OPEN_COARRAYS})

# COMPILER CAPABILITIES
set(HAS_CO_SUM TRUE)
if (CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  set(HAS_CO_SUM FALSE)
endif()

# DEFINE TARGET
add_library(gadfit "")
set_target_properties(gadfit PROPERTIES
  Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)
target_include_directories(gadfit INTERFACE
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(gadfit PRIVATE ${LIBS})

# CONFIG.H
configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${PROJECT_BINARY_DIR}/config.h)
target_include_directories(gadfit PRIVATE ${PROJECT_BINARY_DIR})

# SUBDIRECTORIES
add_subdirectory(doc)
add_subdirectory(src)
add_subdirectory(tests)

# SUMMARY
message("")
message("Summary")
message("=======")
message("Fortran compiler: ${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}")
set(all_flags ${CMAKE_Fortran_FLAGS})
if (CMAKE_BUILD_TYPE)
  string(TOUPPER ${CMAKE_BUILD_TYPE} buildtype)
  set(all_flags "${all_flags} ${CMAKE_Fortran_FLAGS_${buildtype}}")
  string(STRIP ${all_flags} all_flags)
  endif()
message("Compiler flags: ${all_flags}")
message("Build configuration: ${CMAKE_BUILD_TYPE}")
if (BUILD_SHARED)
  message("Building a shared library")
else()
  message("Building a static library")
endif()
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
if (GSL_DIR)
  message("Using GSL")
else()
  message("Not using GSL")
endif()
message("Install location: ${CMAKE_INSTALL_PREFIX}")
message("")

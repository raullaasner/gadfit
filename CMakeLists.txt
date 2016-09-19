# Copyright (C) 2014-2016 Raul Laasner
# This file is distributed under the terms of the GNU General Public
# License, see LICENSE in the root directory of the present
# distribution or http://gnu.org/copyleft/gpl.txt .

# CMAKE
cmake_minimum_required(VERSION 3.6)

# PROJECT
project(GADfit VERSION 1.2.3 LANGUAGES Fortran C)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# FORTRAN COMPILER
set(CMAKE_Fortran_COMPILER_VERSION ${CMAKE_Fortran_COMPILER_VERSION}
  CACHE STRING "")
mark_as_advanced(FORCE CMAKE_Fortran_COMPILER_VERSION)

# COMPILER FLAGS
# Redefine initial Release flags
if ("${CMAKE_Fortran_FLAGS_RELEASE}" STREQUAL
    ${CMAKE_Fortran_FLAGS_RELEASE_INIT})
  if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    unset(CMAKE_Fortran_FLAGS_RELEASE CACHE)
    if (NOT CMAKE_Fortran_COMPILER_VERSION VERSION_LESS "5.1.0")
      set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -fstack-arrays -fcoarray=lib"
	CACHE STRING "GFortran Release flags")
    else()
      # Multi-image support starts with GCC 5
      set(CAF_SINGLE_RELEASE TRUE)
      set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -fstack-arrays -fcoarray=single"
	CACHE STRING "GFortran Release flags")
    endif()
  elseif (CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    unset(CMAKE_Fortran_FLAGS_RELEASE CACHE)
    set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -ipo -xHost -coarray=shared"
      CACHE STRING "IFORT Release flags")
  endif()
endif()
# Redefine initial Debug flags
if ("${CMAKE_Fortran_FLAGS_DEBUG}" STREQUAL ${CMAKE_Fortran_FLAGS_DEBUG_INIT})
  if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    unset(CMAKE_Fortran_FLAGS_DEBUG CACHE)
    if (NOT CMAKE_Fortran_COMPILER_VERSION VERSION_LESS "5.1.0")
      set(CMAKE_Fortran_FLAGS_DEBUG "-O3 -g -Wall -Wno-maybe-uninitialized -Wextra -pedantic -fcheck=all -ffpe-trap=zero,overflow -fbacktrace -fcoarray=lib -std=f2008ts" CACHE STRING "GFortran Debug flags")
    else()
      set(CAF_SINGLE_DEBUG TRUE)
      set(CMAKE_Fortran_FLAGS_DEBUG "-O3 -g -Wall -Wno-maybe-uninitialized -Wextra -pedantic -fcheck=all -ffpe-trap=zero,overflow -fbacktrace -fcoarray=single -std=f2008ts" CACHE STRING "GFortran Debug flags")
    endif()
  elseif (CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    unset(CMAKE_Fortran_FLAGS_DEBUG CACHE)
    set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -debug all -check all -fpe0 -traceback -warn -coarray=shared -std08" CACHE STRING "IFORT Debug flags")
  endif()
endif()
# Current flags
macro(determine_current_flags)
  set(CURRENT_Fortran_FLAGS ${CMAKE_Fortran_FLAGS})
  if (CMAKE_BUILD_TYPE MATCHES "^Release$" OR NOT BUILD_SHARED)
    set(CURRENT_Fortran_FLAGS
      "${CURRENT_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS_RELEASE}")
  elseif (CMAKE_BUILD_TYPE MATCHES "^Debug$")
    set(CURRENT_Fortran_FLAGS
      "${CURRENT_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS_DEBUG}")
  endif()
endmacro()
determine_current_flags()
# Make sure the required flags are present
if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  foreach(flag -cpp -ffree-line-length-0)
    if (NOT CURRENT_Fortran_FLAGS MATCHES ${flag})
      set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${flag}")
    endif()
  endforeach()
endif()
if (CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  if ((NOT CURRENT_Fortran_FLAGS MATCHES "-fpp") AND
      (NOT CURRENT_Fortran_FLAGS MATCHES "-cpp"))
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fpp")
  endif()
endif()
set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} CACHE STRING
  "Flags for Fortran compiler." FORCE)
determine_current_flags()
  
# OPTIONS
option(USE_AD
  "If enabled, uses automatic differentiation, else finite differences" ON)
# Default is Release
if (NOT BUILD_SHARED) # If this is the first run
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the build configuration (None, Debug, Release)" FORCE)
endif()
option(BUILD_SHARED "If enabled, builds a shared library, else static." ON)
mark_as_advanced(CLEAR CMAKE_Fortran_COMPILER)
mark_as_advanced(CLEAR CMAKE_Fortran_FLAGS)
mark_as_advanced(CLEAR CMAKE_Fortran_FLAGS_DEBUG)
mark_as_advanced(CLEAR CMAKE_Fortran_FLAGS_RELEASE)
unset(CMAKE_Fortran_FLAGS_MINSIZEREL CACHE)
unset(CMAKE_Fortran_FLAGS_RELWITHDEBINFO CACHE)
set(GSL_DIR "" CACHE STRING
  "GSL root directory (standard location is, e.g., /usr)")
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
    include_directories(${GSL_INCLUDE_DIR})
    set(LIBS ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
  else()
    set(GSL_DIR FALSE)
  endif()
endif()
set(LINALG_FLAVOR "" CACHE STRING
  "Choose the linear algebra library (lapack, atlas, mkl)")
set(LINALG_LIB_PATH "" CACHE STRING
  "Path to the linear algebra libraries")
if (LINALG_LIB_PATH)
  string(REPLACE " " ";" LINALG_LIB_PATH ${LINALG_LIB_PATH})
  link_directories(${LINALG_LIB_PATH})
endif()
set(LINALG_LIBS "" CACHE STRING
  "Linear algebra libraries (overrides LINALG_FLAVOR)")
if (LINALG_LIBS)
  string(REPLACE " " ";" LIBS "${LIBS} ${LINALG_LIBS}")
endif()
option(QUAD_PRECISION
  "If enabled, uses quadruple precision, else double precision" OFF)
set(OPEN_COARRAYS "" CACHE STRING "Full path to libcaf_mpi.a")
set(LIBS ${LIBS} ${OPEN_COARRAYS})

# COMPILER CAPABILITIES
# Default
set(POLYM_ARRAY_SUPPORT       TRUE)
set(HAS_CO_SUM                TRUE)
# GNU
if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  if (NOT CURRENT_Fortran_FLAGS MATCHES "-fcoarray=")
    message(FATAL_ERROR "Missing a Coarray flag (-fcoarray=)")
  endif()
  set(POLYM_ARRAY_SUPPORT FALSE)
endif()
# Intel
if (CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  if (NOT CURRENT_Fortran_FLAGS MATCHES "-coarray=")
    message(FATAL_ERROR "Missing a Coarray flag (-coarray=)")
  endif()
  set(HAS_CO_SUM FALSE)
endif()

# CONFIG.H
configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${PROJECT_BINARY_DIR}/config.h)
include_directories(${PROJECT_BINARY_DIR})

# SUBDIRECTORIES
add_subdirectory(doc/user_guide)
add_subdirectory(src)
add_subdirectory(tests)

# UNINSTALL
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake" IMMEDIATE @ONLY)
 add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake)

# SUMMARY
message("")
message("Summary")
message("=======")
message("Fortran compiler: ${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}")
message("Compiler flags: ${CURRENT_Fortran_FLAGS}")
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
foreach(x RELEASE DEBUG)
  if (CAF_SINGLE_${x})
    message(WARNING "Building ${x} in serial (-fcoarray=single)")
  endif()
endforeach()
message("")

# Copyright (C) 2014-2015 Raul Laasner
# This file is distributed under the terms of the GNU General Public
# License, see LICENSE in the root directory of the present
# distribution or http://gnu.org/copyleft/gpl.txt .

# PROJECT
cmake_minimum_required(VERSION 2.8.12)
project(GADfit Fortran)
set(GADfit_VERSION_MAJOR 1)
set(GADfit_VERSION_MINOR 2)
set(GADfit_VERSION_PATCH 0)
set(GADFIT gadfit-${GADfit_VERSION_MAJOR}.${GADfit_VERSION_MINOR}.${GADfit_VERSION_PATCH})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# FORTRAN COMPILER
execute_process(COMMAND ${CMAKE_Fortran_COMPILER} --version
  OUTPUT_VARIABLE tmp)
string(REGEX MATCH "[0-9]+[.][0-9]+[.][0-9]+" COMPILER_VERSION "${tmp}")
# Replace initial Release flags set by CMake
if ("${CMAKE_Fortran_FLAGS_RELEASE}" STREQUAL
    ${CMAKE_Fortran_FLAGS_RELEASE_INIT})
  if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    unset(CMAKE_Fortran_FLAGS_RELEASE CACHE)
    set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -fstack-arrays -fcoarray=lib -ffree-line-length-0 -cpp" CACHE STRING "GFortran Release flags")
  elseif (CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    unset(CMAKE_Fortran_FLAGS_RELEASE CACHE)
    set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -ipo -xHost -coarray=shared -fpp"
      CACHE STRING "IFORT Release flags")
  endif()
endif()
# Replace initial Debug flags set by CMake
if ("${CMAKE_Fortran_FLAGS_DEBUG}" STREQUAL ${CMAKE_Fortran_FLAGS_DEBUG_INIT})
  if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    unset(CMAKE_Fortran_FLAGS_DEBUG CACHE)
    set(CMAKE_Fortran_FLAGS_DEBUG "-O3 -g -Wall -Wno-maybe-uninitialized -Wextra -pedantic -fcheck=all -ffpe-trap=zero,overflow -fbacktrace -ffree-line-length-0 -fcoarray=lib -std=f2008ts -cpp" CACHE STRING "GFortran Debug flags")
  elseif (CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    unset(CMAKE_Fortran_FLAGS_DEBUG CACHE)
    set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -debug all -check all -fpe0 -traceback -warn -coarray=shared -std08 -fpp" CACHE STRING "IFORT Debug flags")
  endif()
endif()
  
# OPTIONS
option(USE_AD
  "If enabled, uses automatic differentiation, else finite differences" ON)
# Default is Release
if (NOT BUILD_SHARED) # If this is the first run
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the build configuration (None Debug Release)" FORCE)
endif()
option(BUILD_SHARED "If enabled, builds a shared library, else static." ON)
mark_as_advanced(CLEAR CMAKE_Fortran_FLAGS)
mark_as_advanced(CLEAR CMAKE_Fortran_FLAGS_DEBUG)
mark_as_advanced(CLEAR CMAKE_Fortran_FLAGS_RELEASE)
set(GSL_DIR "" CACHE STRING "GSL root directory")
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

# COMPILER TESTS
set(FLAGS_TMP ${CMAKE_Fortran_FLAGS})
if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_Fortran_FLAGS
    "${CMAKE_Fortran_FLAGS_RELEASE} ${CMAKE_Fortran_FLAGS}")
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS_DEBUG} ${CMAKE_Fortran_FLAGS}")
endif()
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests)
message(STATUS "Testing compiler support for")
# Coarrays
try_compile(COARRAYS ${CMAKE_CURRENT_BINARY_DIR} ${TEST_DIR}/Coarrays.f90
  LINK_LIBRARIES ${OPEN_COARRAYS})
message("     Coarrays - " ${COARRAYS})
if (NOT COARRAYS AND CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  # If test failed, try again with -fcoarray=single
  foreach(x FLAGS_TMP CMAKE_Fortran_FLAGS CMAKE_Fortran_FLAGS_RELEASE
      CMAKE_Fortran_FLAGS_DEBUG)
    string(REPLACE "-fcoarray=lib" "-fcoarray=single" ${x} "${${x}}")
  endforeach()
  try_compile(COARRAYS ${CMAKE_CURRENT_BINARY_DIR} ${TEST_DIR}/Coarrays.f90
    LINK_LIBRARIES ${OPEN_COARRAYS})
  if (NOT COARRAYS)
    message(FATAL_ERROR
      "This project uses coarrays. The relevant compiler flags are missing.")
  endif()
  set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE}
    CACHE STRING "GFortran Release flags" FORCE)
  set(CMAKE_Fortran_FLAGS_DEBUG ${CMAKE_Fortran_FLAGS_DEBUG}
    CACHE STRING "GFortran Debug flags" FORCE)
  set(GF_LIB_SINGLE TRUE INTERNAL)
else()
  unset(GF_LIB_SINGLE)
endif()
# Polymorphic allocatable array
try_run(POLYM_ARRAY X ${CMAKE_CURRENT_BINARY_DIR}
  ${TEST_DIR}/PolymAllocArray.f90 LINK_LIBRARIES ${OPEN_COARRAYS})
if (${POLYM_ARRAY} MATCHES 0)
  set(POLYM_ARRAY_SUPPORT TRUE)
else()
  set(POLYM_ARRAY_SUPPORT FALSE)
endif()
message("     Polymorphic allocatable array - " ${POLYM_ARRAY_SUPPORT})
# Impure elemental
try_compile(IMPURE_ELEMENTAL_SUPPORT ${CMAKE_CURRENT_BINARY_DIR}
  ${TEST_DIR}/ImpureElemental.f90 LINK_LIBRARIES ${OPEN_COARRAYS})
message("     Impure elemental - " ${IMPURE_ELEMENTAL_SUPPORT})
# Critical
try_compile(CRITICAL_SUPPORT
  ${CMAKE_CURRENT_BINARY_DIR} ${TEST_DIR}/Critical.f90
  LINK_LIBRARIES ${OPEN_COARRAYS})
message("     Critical - " ${CRITICAL_SUPPORT})
# Associate
try_compile(ASSOCIATE_SUPPORT
  ${CMAKE_CURRENT_BINARY_DIR} ${TEST_DIR}/Associate.f90
  LINK_LIBRARIES ${OPEN_COARRAYS})
message("     Associate - " ${ASSOCIATE_SUPPORT})
# co_sum
try_compile(HAS_CO_SUM ${CMAKE_CURRENT_BINARY_DIR} ${TEST_DIR}/CoSum.f90
  LINK_LIBRARIES ${OPEN_COARRAYS})
message("     co_sum - " ${HAS_CO_SUM})
# Deferred-length char component
try_compile(DEF_LEN_CHAR_COMP_SUPPORT ${CMAKE_CURRENT_BINARY_DIR}
  ${TEST_DIR}/DefLenCharComp.f90 LINK_LIBRARIES ${OPEN_COARRAYS})
message("     Deferred-length char component - " ${DEF_LEN_CHAR_COMP_SUPPORT})
# Restore flags
set(CMAKE_Fortran_FLAGS ${FLAGS_TMP})
if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS}
    CACHE STRING "GFortran flags" FORCE)
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
message("Fortran compiler: ${CMAKE_Fortran_COMPILER_ID} ${COMPILER_VERSION}")
message("Release flags: ${CMAKE_Fortran_FLAGS_RELEASE}")
message("Debug flags: ${CMAKE_Fortran_FLAGS_DEBUG}")
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
if (GF_LIB_SINGLE)
  message(WARNING "Missing multi-image coarray support. Building in serial..")
endif()
message("")

# This Source Code Form is subject to the terms of the GNU General
# Public License, v. 3.0. If a copy of the GPL was not distributed
# with this file, You can obtain one at
# http://gnu.org/copyleft/gpl.txt.

include(FindPackageHandleStandardArgs)

if (DEFINED ENV{MKLROOT})
  set(MKL_ROOT $ENV{MKLROOT})
else()
  set(MKL_ROOT "/opt/intel/mkl")
endif()

if(EXISTS ${MKL_ROOT}/lib/intel64)
  set(MKL_LIB_DIR ${MKL_ROOT}/lib/intel64)
else()
  set(MKL_LIB_DIR ${MKL_ROOT}/lib/ia32)
endif()

find_path(MKL_INCLUDE_DIR mkl_cblas.h ${MKL_ROOT}/include)
find_package_handle_standard_args(MKL DEFAULT_MSG
  MKL_INCLUDE_DIR MKL_LIB_DIR)
mark_as_advanced(MKL_INCLUDE_DIR)

if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  set(MKL_LIBS mkl_gf_lp64 mkl_sequential mkl_core pthread)
else()
  set(MKL_LIBS mkl_intel_lp64 mkl_sequential mkl_core)
endif()
foreach(l ${MKL_LIBS})
  find_library(lib_${l} NAMES ${l} PATHS ${MKL_LIB_DIR})
  set(MKL_LIBRARIES ${MKL_LIBRARIES} ${lib_${l}})
  find_package_handle_standard_args(MKL DEFAULT_MSG lib_${l})
  mark_as_advanced(lib_${l})
endforeach()

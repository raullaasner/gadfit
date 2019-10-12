###########################

# Initial cache variables for GADfit

# The syntax used here is
#  set(<variable> <value> CACHE <type> <docstring>) .
# You only need to edit the <value> parameter. For example, using
#  set(CMAKE_Fortran_COMPILER mpif90 CACHE STRING "")
# and running
#  cmake -C <src_root>/configure.txt <src_root>
# is the Autotools equivalent of running
#  <src_root>/configure FC=mpif90

###########################

# Fortran compiler
#set (CMAKE_Fortran_COMPILER mpif90 CACHE STRING "")

# Flags for the Fortran compiler.
#set (CMAKE_Fortran_FLAGS "" CACHE STRING "")
#set (CMAKE_Fortran_FLAGS_RELEASE "-O3 -funroll-all-loops -fstack-arrays -fcoarray=lib -ffree-line-length-0" CACHE STRING "")
#set (CMAKE_Fortran_FLAGS_DEBUG "-O3 -g -Wall -Wno-maybe-uninitialized -Wextra -pedantic -fcheck=all -ffpe-trap=zero,overflow -fbacktrace -fcoarray=lib -std=f2008ts -ffree-line-length-0" CACHE STRING "")

# Whether to build in Release or Debug configuration.
set (CMAKE_BUILD_TYPE Release CACHE STRING "")

# Whether to build a shared or a static library.
set (BUILD_SHARED ON CACHE BOOL "")

# Where to install GADfit
set (CMAKE_INSTALL_PREFIX /usr/local/gadfit CACHE STRING "")

# Whether to use automatic differentiation. This only determines
# whether AD or finite differences is used for calculating the
# derivatives during the fitting procedure. Both AD and finite
# differences are still available.
set (USE_AD ON CACHE BOOL "")

# Whether to use quadruple or double precision.
set (QUAD_PRECISION OFF CACHE BOOL "")

# If the code contains calls to the GSL library, then GSL_DIR should
# point to the GSL install directory. If GSL_DIR is left empty or is
# incorrect, GADfit will compile without the GSL related procedures.
#set (GSL_DIR /usr/local/gsl ON CACHE BOOL "")

# A linear algebra library is necessary when using GADfit or when
# running the examples, but not while building the main library. If
# neither LINALG_FLAVOR nor LINALG_LIBS is given or the linalg
# libraries are not working, the fallback Lapack library is used for
# running the tests. Allowed values are
#   lapack
#   atlas
#   mkl
#set (LINALG_FLAVOR lapack CACHE STRING "")

# Linalg library directories
#set (LINALG_LIB_PATH /usr/local/atlas/lib CACHE STRING "")

# Linalg libraries. Must be shared libraries.
#set (LINALG_LIBS tatlas CACHE STRING "")

# Location of libcaf_mpi.a
#set (OPEN_COARRAYS /usr/local/opencoarrays/lib/libcaf_mpi.a CACHE STRING "")

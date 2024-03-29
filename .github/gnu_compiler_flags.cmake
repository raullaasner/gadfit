set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g -Wall -Wextra -Wcast-align -Wformat -Winvalid-pch -Wmissing-declarations -Wmissing-include-dirs -Wconversion -Wredundant-decls -Wswitch-default -Wswitch-enum -pedantic -Wno-parentheses" CACHE STRING "")
set(CMAKE_Fortran_FLAGS "-ffree-line-length-0" CACHE STRING "")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -funroll-all-loops -fcoarray=lib" CACHE STRING "")
set(CMAKE_Fortran_FLAGS_DEBUG "-O3 -g -Wall -Wno-maybe-uninitialized -Wextra -pedantic -fcheck=all -ffpe-trap=zero,overflow -fbacktrace -fcoarray=lib -std=f2008ts" CACHE STRING "")

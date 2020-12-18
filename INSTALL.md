Quick installation instructions
===============================

GADfit is developed with the CMake build system generator. CMake is a set of tools for configuring, building, and testing software. In comparison to GNU Autotools, it uses a simpler syntax and generally runs faster. It is advised to read the [user guide](https://raullaasner.github.io/gadfit/doc/user_guide/user_guide.pdf) for detailed installation instructions. This document only provides quick steps to get you started.

0. For the impatient
--------------------

Obtain CMake, the GADfit tarball, and issue

```
tar xf gadfit.tar.gz
cd gadfit
cmake .
make
make install
```

Note that this might produce only a serial build.

1. Prerequisites
----------------

GADfit comes in two implementations, one in C++ and one in Fortran. The CMake variable that controls which version is built is `BUILD_CXX_VERSION`.

* **CMake (required)**. On Ubuntu, CMake, released under the New BSD license, can be obtained by issuing

   ```
   sudo apt-get install cmake
   ```

  Source code is available at http://cmake.org.

* C++ version:
  * **C++ compiler (required)**. GADfit is expected to work with the latest GNU C++ compiler (G++). One is probably present in your system so you don't need to do anything here.
  * **spdlog (required)**. This is a C++ logger. You may install it using your distribution's package manager or build it manually and point the CMake SPDLOG_PATH variable to the installation directory (see below for how CMake cache variables work).

* Fortran version:
  * **Fortran compiler (required)**. GADfit is expected to work with the latest GNU Fortran compiler (GFortran). In principle, any other F2008 compliant compiler should also work. On Ubuntu, GFortran, released under the GPL 3+ license, can be obtained by issuing

    ```
    sudo apt-get install gfortran
    ```

    Source code is available at https://gcc.gnu.org/wiki/GFortran.

  * **OpenCoarrays**. The multi-image Coarray support of GFortran is provided by the OpenCoarrays project. This allows to run GADfit in parallel. It is available at https://github.com/sourceryinstitute/opencoarrays and can be built and installed using standard CMake commands. When running tests and if the OpenCoarrays library is not specified the user, it is automatically downloaded and built. The Coarray support of the Intel compiler does not depend on an external project and is available out of the box.

* **Linear algebra library**. While not required during compilation, a linear algebra library is necessary for using GADfit for running the tests (see `make test` below). If none is specified, the fallback Lapack library is used.

2a. Linux
---------

Note that issuing `./configure <options> && make && make install` does not work with GADfit, or in general with any CMake project. While it is possible to specify the build environment on the command line (`cmake <options> ...`), in analogy to Autotools, it is generally more convenient to use a CMake frontend or a separate file containing the CMake configuration variables. On Ubuntu, the two commonly use graphical front ends can be obtained by installing `cmake-curses-gui` and `cmake-qt-gui`.

* Untar the source code and navigate into the build directory

   ```
   tar xf gadfit-x.x.x.tar.gz
   mkdir build && cd build
   ```

   While GADfit builds fine in the source directory, it is preferred to do out-of-source builds instead in order to keep the source directory clean. In the following, the source and build directories are denoted `~gadfit` and `~build`.

* In order to use the CMake frontend `ccmake`, issue

   ```
   cmake ~gadfit
   ccmake .
   ```

   in the build directory. Some CMake variables and options appear, most of which should be self-explanatory. A short help text to each variable is displayed at the bottom in a status bar. Pressing `t` reveals all the options. When done editing, press `c` to reconfigure and `g` to generate the native build script (the unix makefile on Linux). Check that all variables are correct and reconfigure again if necessary. Exit `ccmake` and issue

   ```
   cmake --build .
   cmake --build . --target test
   cmake --build . --target install
   ```

   Tests can also be built individually by issuing `make <test>`, where `<test>` is the test name without the f90 extension. `make doc` generates the user guide, but a prebuilt one also available [here](https://raullaasner.github.io/gadfit/doc/user_guide/user_guide.pdf).

* The Qt based CMake frontend can be used by issuing

   ```
   cmake ~gadfit
   cmake-gui .
   ```

   in the build directory. The steps for configuring are analogous to `ccmake`.

* If the GUI approach seems discouraging, then there is also the option of putting all the relevant CMake configuration variables into an external configuration file, an example of which is `initial_cache.cmake.example` in the root directory. Make a copy of it and when done editing, run

   ```
   cmake -C ~gadfit/initial_cache.cmake ~gadfit
   ```

   from the build directory, followed by the make commands.

2b. Others
----------

Currently not supported (any help for porting this software to other platforms is most welcome).

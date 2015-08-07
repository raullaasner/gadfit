Quick installation instructions
===============================

GADfit is developed with the CMake build system. CMake is a set of tools for configuring, building, and testing software. In comparison to GNU Autotools, it uses a simpler syntax and generally runs faster. It is advised to read the [user guide](https://raullaasner.github.io/gadfit/doc/user_guide/user_guide.pdf) for detailed installation instructions. This document provides quick installation steps aimed mainly at Autotools oriented users.

0. For the impatient
--------------------

Obtain CMake, a Fortran compiler, the GADfit tarball, and issue

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

*If the user is running Ubuntu, it must not be older than 14.04. Otherwise, newer versions of CMake and GFortran need to be installed manually. For multi-image support, GCC at least version 5.1 and CMake at least version 3.0 are required.*

* **Fortran compiler (required)**. GADfit is expected to work with the latest GNU Fortran compiler (GFortran). In principle, any other F2008 compliant compiler should also work. On Ubuntu, GFortran, released under the GPL 3+ license, can be obtained by issuing

   ```
   sudo apt-get install gfortran
   ```

   Source code is available at https://gcc.gnu.org/wiki/GFortran.

* **CMake (required)**. On Ubuntu, CMake, released under the New BSD license, can be obtained by issuing
   
   ```
   sudo apt-get install cmake
   ```
   
  Source code is available at http://cmake.org.

* **OpenCoarrays**. The multi-image Coarray support of GFortran is provided by the OpenCoarrays project, which allows to run GADfit in parallel. It is available at https://github.com/sourceryinstitute/opencoarrays and can be built and installed using standard CMake commands. By default, the OpenCoarrays is automatically downloaded and built (only if the compiler flag `-fcoarray=lib` is given). The Coarray support of the Intel compiler does not depend on an external project and is available out of the box.

* **Linear algebra library**. While not required during compilation, a linear algebra library is necessary for using GADfit or running the tests (see `make check` below). By default, the supplied Lapack library is used.

2a. Linux
---------

To be clear, issuing `./configure <options> && make && make install` does not work with GADfit, or in general with any CMake project. While it is possible to specify the build environment on the command line (`cmake <options> ...`), in analogy to Autotools, it is generally more convenient to use a CMake frontend or a separate file containing CMake configuration variables. On Ubuntu, the two commonly use graphical front ends can be obtained by installing `cmake-curses-gui` and `cmake-qt-gui`.

* Untar the source code and navigate into the build directory
   
   ```
   tar xf gadfit-x.x.x.tar.gz
   mkdir build && cd build
   ```
   
   While GADfit builds fine in the source directory, it is generally a good habit to do out-of-source builds in order to keep the source directory clean. The source and build directories are denoted `~gadfit` and `~build`.

* In order to use the CMake frontend `ccmake`, issue
   
   ```
   cmake ~gadfit
   ccmake .
   ```
   
   in the build directory. Some CMake variables and options appear, most of which should be self-explanatory. A short help text to each variable is displayed at the bottom in a status bar. Pressing `t` reveals all the options. When done editing, press `c` to reconfigure and `g` to generate the native build script (the unix makefile on Linux). Check that all variables are correct and reconfigure again if necessary. Exit `ccmake` and issue
   
   ```
   make
   make check
   make install
   ```
   
   `check` runs the tests found in `~gadfit/tests`. Tests can also be built individually by issuing `make <test>`, where `<test>` is the test name without the f90 extension. `make doc` generates the user guide, although a prebuilt one is supplied with the source code. All `make` commands can be used with `-j<n>`, where `<n>` is the number of threads.
   
* The Qt based CMake frontend can be used by issuing
   
   ```
   cmake ~gadfit
   cmake-gui .
   ```
   
   in the build directory. The steps for configuring are analogous to `ccmake`.
   
* If the GUI approach seems discouraging, then there is also the option of putting all the relevant CMake configuration variables into an external configuration file, an example of which is [configure.txt](https://github.com/raullaasner/gadfit/blob/master/configure.txt) in the root directory. When done editing, run
   
   ```
   cmake -C ~gadfit/configure.txt ~gadfit
   ```
   
   from the build directory, followed by the make commands.

2b. Others
----------

Currently not supported (any help for porting this software to other platforms is most welcome).

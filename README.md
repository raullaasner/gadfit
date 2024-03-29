[![Build Status](https://github.com/raullaasner/gadfit/workflows/main/badge.svg)](https://github.com/raullaasner/gadfit/actions)

GADfit
======

Global nonlinear optimization with automatic differentiation

Brief description
-----------------

GADfit is an implementation of global nonlinear curve fitting, based on a modified Levenberg-Marquardt algorithm. Global fitting refers to fitting many datasets simultaneously with some parameters shared among the datasets. The fitting procedure is very fast and accurate thanks to the use of automatic differentiation. The model curves (fitting functions) can be of essentially arbitrary complexity. This includes any nonlinear combination of elementary and special functions, single and/or double integrals, and any control flow statement allowed by the programming language. See the latest [user guide](https://raullaasner.github.io/gadfit/doc/user_guide.pdf).

Features
--------

* Modified Levenberg-Marquardt
* Multi-curve fitting with shared parameters
* Automatic differentiation
* MPI, Scalapack (C++) and Coarray (Fortran) parallelism

Downloading
-----------

* The official versioned releases are available [here](https://github.com/raullaasner/gadfit/releases).
* The latest stable development code is available [here](https://github.com/raullaasner/gadfit/archive/master.zip).
* For importing the repository, issue `git clone https://github.com/raullaasner/gadfit.git`

Installation
------------

* See [INSTALL.md](https://github.com/raullaasner/gadfit/blob/master/INSTALL.md) for quick installation instructions.
* See the [user guide](https://raullaasner.github.io/gadfit/doc/user_guide.pdf) for detailed instructions (recommended).

Screenshots
-----------

* [C++ Example driver code](https://raullaasner.github.io/gadfit/cxx_driver_code.png)
* [C++ output](https://raullaasner.github.io/gadfit/cxx_output.png)
* [Fortran example driver code](https://raullaasner.github.io/gadfit/fortran_driver_code.png)
* [Fortran example fitting function](https://raullaasner.github.io/gadfit/fortran_fitting_function.png)
* [Fortran output](https://raullaasner.github.io/gadfit/fortran_output.png)

Authors and how to contact them
-------------------------------

See the [user guide](https://raullaasner.github.io/gadfit/doc/user_guide.pdf), Section 1.1.

Documentation
-------------

Details about theory, implementation, configuring and compilation of GADfit, and usage are found in the [user guide](https://raullaasner.github.io/gadfit/doc/user_guide.pdf).

Troubleshooting
---------------

A good place to bring up any issues is https://github.com/raullaasner/gadfit/issues.

License
-------

This project is distributed under the terms of Apache License 2.0. See LICENSE in the root directory of the project or go to http://www.apache.org/licenses/LICENSE-2.0.

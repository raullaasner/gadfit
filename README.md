GADfit
======

Global nonlinear optimization with automatic differentiation

Brief description
-----------------

GADfit is a Fortran implementation of global nonlinear curve fitting, based on a modified Levenberg-Marquardt algorithm. Global fitting refers to fitting many datasets simultaneously with some parameters shared among the datasets. The fitting procedure is very fast and accurate thanks to the use of automatic differentiation. The model curves (fitting functions) can be of essentially arbitrary complexity. This includes any nonlinear combination of elementary and special functions, single and/or double integrals, and any control flow statement allowed by the programming language. See the latest [user guide](https://github.com/raullaasner/gadfit/blob/master/doc/user_guide/user_guide.pdf).

Features
--------

* Modified Levenberg-Marquardt
* Multi-curve fitting with shared parameters
* Automatic differentiation
* Adaptive parallelism using coarrays

Downloading
-----------

* The official versioned releases are available [here](https://github.com/raullaasner/gadfit/releases).
* The latest stable development code is available [here](https://github.com/raullaasner/gadfit/archive/master.zip).
* For importing the repository, issue `git clone https://github.com/raullaasner/gadfit.git`

Installation
------------

* See INSTALL.md for quick installation instructions.
* See the [user guide](https://github.com/raullaasner/gadfit/blob/master/doc/user_guide/user_guide.pdf) for detailed instructions (recommended).

Authors and how to contact them
-------------------------------

See the [user guide](https://github.com/raullaasner/gadfit/blob/master/doc/user_guide/user_guide.pdf), Section 1.1.

Documentation
-------------

See the [user guide](https://github.com/raullaasner/gadfit/blob/master/doc/user_guide/user_guide.pdf)

Troubleshooting
---------------

A good place to bring up any issues is https://github.com/raullaasner/gadfit/issues.

License
-------

This project is distributed under the terms of the GNU General Public License, see LICENSE in the root directory of the present distribution or http://gnu.org/copyleft/gpl.txt.

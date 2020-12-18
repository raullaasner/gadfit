// This Source Code Form is subject to the terms of the GNU General
// Public License, v. 3.0. If a copy of the GPL was not distributed
// with this file, You can obtain one at
// http://gnu.org/copyleft/gpl.txt.

#pragma once

#include <mkl_cblas.h>
#include <vector>

namespace gadfit {

// Computes C = A^T*A
auto dsyrk_wrapper(const int N,
                   const int K,
                   const std::vector<double>& A,
                   std::vector<double>& C) -> void;

// Computes y = A^T*x, where x is a vector
auto dgemv_tr_wrapper(const int M,
                      const int N,
                      const std::vector<double>& A,
                      const std::vector<double>& x,
                      std::vector<double>& y) -> void;

// Solves A*x = b with A a real symmetric positive definite matrix in
// two steps: first by computing the Cholesky factorization of the
// form L*L^T of A, and then solving the linear system of equations
// L*y = b followed by L^T*x = y.
auto dpotrs_wrapper(const int dimension,
                    std::vector<double>& A,
                    std::vector<double>& b) -> void;

} // namespace gadfit

// Licensed under the Apache License, Version 2.0 (the "License"); you
// may not use this file except in compliance with the License.  You
// may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied.  See the License for the specific language governing
// permissions and limitations under the License.

#pragma once

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
auto dpptrs_wrapper(const int dimension,
                    std::vector<double>& A,
                    std::vector<double>& b) -> void;

} // namespace gadfit

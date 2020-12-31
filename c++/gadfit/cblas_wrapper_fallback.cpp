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

// Reference implementations of linear algebra routines used in
// GADfit. These are used when not linking against a linear algebra
// library. These are intentionally unoptimized for debugging
// purposes.

#include "cblas_wrapper.h"

#include <cmath>

namespace gadfit {

auto dsyrk_wrapper(const int N,
                   const int K,
                   const std::vector<double>& A,
                   std::vector<double>& C) -> void
{
    std::fill(C.begin(), C.end(), 0.0);
    for (int i {}; i < N; ++i) {
        for (int j {}; j < N; ++j) {
            for (int k {}; k < K; ++k) {
                C[i * N + j] += A[k * N + i] * A[k * N + j];
            }
        }
    }
}

auto dgemv_tr_wrapper(const int M,
                      const int N,
                      const std::vector<double>& A,
                      const std::vector<double>& x,
                      std::vector<double>& y) -> void
{
    std::fill(y.begin(), y.end(), 0.0);
    for (int i {}; i < N; ++i) {
        for (int j {}; j < M; ++j) {
            y[i] += A[j * N + i] * x[j];
        }
    }
}

auto dpptrs_wrapper(const int dimension,
                    std::vector<double>& A,
                    std::vector<double>& b) -> void
{
    // Cholesky decomposition
    std::vector<double> L(A.size(), 0.0);
    for (int i {}; i < dimension; ++i) {
        // L_ij
        for (int j {}; j < i; ++j) {
            const int ij { i * dimension + j };
            L[ij] = A[ij];
            for (int k {}; k < j; ++k) {
                L[ij] -= L[i * dimension + k] * L[j * dimension + k];
            }
            L[ij] = L[ij] / L[j * dimension + j];
        }
        // L_jj
        const int ii { i * dimension + i };
        L[ii] = A[ii];
        for (int k {}; k < i; ++k) {
            L[ii] -= L[i * dimension + k] * L[i * dimension + k];
        }
        L[ii] = std::sqrt(L[ii]);
    }
    // Forward substitution (solving L*y=b)
    std::vector<double> y(b.size());
    for (int i {}; i < dimension; ++i) {
        y[i] = b[i];
        for (int j {}; j < i; ++j) {
            y[i] -= L[i * dimension + j] * y[j];
        }
        y[i] /= L[i * dimension + i];
    }
    // Backward substitution (solving L^T*x=y)
    for (int i { dimension - 1 }; i >= 0; --i) {
        b[i] = y[i]; // Here b=x
        for (int j { dimension - 1 }; j > i; --j) {
            // No need to explicitly use L^T here
            b[i] -= L[j * dimension + i] * b[j];
        }
        b[i] /= L[i * dimension + i];
    }
}

} // namespace gadfit

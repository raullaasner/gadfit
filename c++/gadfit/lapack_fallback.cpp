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

#include "lapack.h"

#include <algorithm>
#include <cmath>

namespace gadfit {

auto dsyrk(const char,
           const int N,
           const int K,
           const std::vector<double>& A,
           std::vector<double>& C) -> void
{
    std::ranges::fill(C, 0.0);
#pragma omp parallel
    for (int k {}; k < K; ++k) {
#pragma omp for nowait
        for (int i = 0; i < N; ++i) {
            for (int j {}; j < N; ++j) {
                C[i * N + j] += A[k * N + i] * A[k * N + j];
            }
        }
    }
}

auto dgemv(const char,
           const int M,
           const int N,
           const std::vector<double>& A,
           const std::vector<double>& X,
           std::vector<double>& Y) -> void
{
    std::fill(Y.begin(), Y.end(), 0.0);
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j {}; j < M; ++j) {
            Y[i] += A[j * N + i] * X[j];
        }
    }
}

auto dpptrf(const int dimension, std::vector<double>& A) -> void
{
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
    A = L;
}

auto dpptrs(const int dimension,
            const std::vector<double>& A,
            std::vector<double>& B) -> void
{
    // Forward substitution (solving A*y=b)
    std::vector<double> y(B.size());
    for (int i {}; i < dimension; ++i) {
        y[i] = B[i];
        for (int j {}; j < i; ++j) {
            y[i] -= A[i * dimension + j] * y[j];
        }
        y[i] /= A[i * dimension + i];
    }
    // Backward substitution (solving A^T*x=y)
    for (int i { dimension - 1 }; i >= 0; --i) {
        B[i] = y[i]; // Here B=x
        for (int j { dimension - 1 }; j > i; --j) {
            // No need to explicitly use A^T here
            B[i] -= A[j * dimension + i] * B[j];
        }
        B[i] /= A[i * dimension + i];
    }
}

auto dpotri(const int dimension, std::vector<double>& A) -> void
{
    // Populate upper triangle
    for (int j {}; j < dimension; ++j) {
        for (int i {}; i < j; ++i) {
            A[i * dimension + j] = A[j * dimension + i];
        }
    }
    std::vector<double> B(dimension * dimension, 0.0);
    for (int j {}; j < dimension; ++j) {
        std::vector<double> B_slice(dimension, 0.0);
        B_slice[j] = 1.0;
        dpptrs(dimension, A, B_slice);
        for (int i {}; i < dimension; ++i) {
            B[i * dimension + j] = B_slice[i];
        }
    }
    A = B;
}

} // namespace gadfit

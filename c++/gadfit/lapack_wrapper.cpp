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

#include "lapack_wrapper.h"

extern "C"
{
    auto dsyrk_(const char*,
                const char*,
                const int*,
                const int*,
                const double*,
                const double*,
                const int*,
                const double*,
                double*,
                const int*) -> void;
    auto dgemv_(const char*,
                const int*,
                const int*,
                const double*,
                const double*,
                const int*,
                const double*,
                const int*,
                const double*,
                double*,
                const int*) -> void;
    auto dpptrf_(const char*, const int*, double*, int*) -> void;
    auto dpptrs_(const char*,
                 const int*,
                 const int*,
                 const double*,
                 const double*,
                 const int*,
                 int*) -> void;
}

namespace gadfit {

auto dsyrk_wrapper(const int N,
                   const int K,
                   const std::vector<double>& A,
                   std::vector<double>& C) -> void
{
    // 'l' because this is Fortran ordering
    constexpr char uplo { 'l' };
    constexpr char trans { 'n' };
    constexpr double alpha { 1.0 };
    constexpr double beta { 0.0 };
    dsyrk_(&uplo, &trans, &N, &K, &alpha, A.data(), &N, &beta, C.data(), &N);
}

auto dgemv_tr_wrapper(const int M,
                      const int N,
                      const std::vector<double>& A,
                      const std::vector<double>& x,
                      std::vector<double>& y) -> void
{
    constexpr char trans { 'n' };
    constexpr double alpha { 1.0 };
    constexpr double beta { 0.0 };
    constexpr int inc { 1 };
    dgemv_(&trans,
           &N,
           &M,
           &alpha,
           A.data(),
           &N,
           x.data(),
           &inc,
           &beta,
           y.data(),
           &inc);
}

auto dpptrs_wrapper(const int dimension,
                    std::vector<double>& A,
                    std::vector<double>& b) -> void
{
    std::vector<double> A_packed((A.size() * (A.size() + 1)) / 2);
    auto it { A_packed.begin() };
    for (int i {}; i < dimension; ++i) {
        for (int j { i }; j < dimension; ++j) {
            *it++ = A.at(i * dimension + j);
        }
    }
    // 'l' because this is Fortran ordering
    constexpr char uplo { 'l' };
    int info {};
    dpptrf_(&uplo, &dimension, A_packed.data(), &info);
    constexpr int nrhs { 1 };
    dpptrs_(
      &uplo, &dimension, &nrhs, A_packed.data(), b.data(), &dimension, &info);
}

} // namespace gadfit

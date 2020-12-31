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

#include "cblas_wrapper.h"

#include <mkl_cblas.h>
#include <mkl_lapack.h>

namespace gadfit {

auto dsyrk_wrapper(const int N,
                   const int K,
                   const std::vector<double>& A,
                   std::vector<double>& C) -> void
{
    cblas_dsyrk(CblasRowMajor,
                CblasUpper,
                CblasTrans,
                N,
                K,
                1.0,
                A.data(),
                N,
                0.0,
                C.data(),
                N);
}

auto dgemv_tr_wrapper(const int M,
                      const int N,
                      const std::vector<double>& A,
                      const std::vector<double>& x,
                      std::vector<double>& y) -> void
{
    cblas_dgemv(CblasRowMajor,
                CblasTrans,
                M,
                N,
                1.0,
                A.data(),
                N,
                x.data(),
                1,
                0.0,
                y.data(),
                1);
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
    dpptrf(&uplo, &dimension, A_packed.data(), &info);
    constexpr int nrhs { 1 };
    dpptrs(
      &uplo, &dimension, &nrhs, A_packed.data(), b.data(), &dimension, &info);
}

} // namespace gadfit

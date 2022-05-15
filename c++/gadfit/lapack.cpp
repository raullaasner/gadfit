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

#include "lapack.h"

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

constexpr char uplo_l { 'l' };

auto dsyrk(const char trans_c,
           const int N,
           const int K,
           const std::vector<double>& A,
           std::vector<double>& C) -> void
{
    constexpr double alpha { 1.0 };
    constexpr double beta { 0.0 };
    const char trans_fort { trans_c == 'n' ? 't' : 'n' };
    dsyrk_(
      &uplo_l, &trans_fort, &N, &K, &alpha, A.data(), &N, &beta, C.data(), &N);
}

auto dgemv(const char trans_c,
           const int M,
           const int N,
           const std::vector<double>& A,
           const std::vector<double>& X,
           std::vector<double>& Y) -> void
{
    constexpr double alpha { 1.0 };
    constexpr double beta { 0.0 };
    constexpr int inc { 1 };
    const char trans_fort { trans_c == 'n' ? 't' : 'n' };
    dgemv_(&trans_fort,
           &N,
           &M,
           &alpha,
           A.data(),
           &N,
           X.data(),
           &inc,
           &beta,
           Y.data(),
           &inc);
}

auto dpptrf(const int dimension, std::vector<double>& A) -> void
{
    std::vector<double> A_packed((A.size() * (A.size() + 1)) / 2);
    auto it { A_packed.begin() };
    for (int i {}; i < dimension; ++i) {
        for (int j { i }; j < dimension; ++j) {
            *it++ = A.at(i * dimension + j);
        }
    }
    int info {};
    dpptrf_(&uplo_l, &dimension, A_packed.data(), &info);
    A = A_packed;
}

auto dpptrs(const int dimension,
            const std::vector<double>& A,
            std::vector<double>& B) -> void
{
    int info {};
    constexpr int nrhs { 1 };
    dpptrs_(&uplo_l, &dimension, &nrhs, A.data(), B.data(), &dimension, &info);
}

} // namespace gadfit

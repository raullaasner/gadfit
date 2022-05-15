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

auto dsyrk(const char trans_c,
           const int N,
           const int K,
           const std::vector<double>& A,
           std::vector<double>& C) -> void;

auto dgemv(const char trans_c,
           const int M,
           const int N,
           const std::vector<double>& A,
           const std::vector<double>& X,
           std::vector<double>& Y) -> void;

auto dpptrf(const int dimension, std::vector<double>& A) -> void;

auto dpptrs(const int dimension,
            const std::vector<double>& A,
            std::vector<double>& B) -> void;

} // namespace gadfit

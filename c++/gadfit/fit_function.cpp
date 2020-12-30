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

#include "fit_function.h"

namespace gadfit {

auto FitFunction::par(const int i_par) -> AdVar&
{
    if (i_par >= static_cast<int>(parameters.size())) {
        parameters.resize(i_par + 1);
    }
    return parameters.at(i_par);
}

auto FitFunction::getParValue(const int i_par) const -> double
{
    return parameters.at(i_par).val;
}

auto FitFunction::getNumPars() const -> int
{
    return static_cast<int>(parameters.size());
}

auto FitFunction::activateParReverse(const int par_i, const int idx) -> void
{
    AdVar& par { parameters.at(par_i) };
    par.idx = idx;
}

auto FitFunction::activateParForward(const int par_i, const double seed) -> void
{
    AdVar& par { parameters.at(par_i) };
    par.d = seed;
    par.dd = 0.0;
    par.idx = forward_active_idx;
}

auto FitFunction::deactivatePar(const int par_i) -> void
{
    parameters.at(par_i).idx = passive_idx;
}

auto FitFunction::operator()(double arg) const -> AdVar
{
    return function(parameters, arg);
}

} // namespace gadfit

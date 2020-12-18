// This Source Code Form is subject to the terms of the GNU General
// Public License, v. 3.0. If a copy of the GPL was not distributed
// with this file, You can obtain one at
// http://gnu.org/copyleft/gpl.txt.

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

auto FitFunction::activateParForward(const int par_i) -> void
{
    AdVar& par { parameters.at(par_i) };
    par.d = 1.0;
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

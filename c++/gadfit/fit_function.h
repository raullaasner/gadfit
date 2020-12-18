// This Source Code Form is subject to the terms of the GNU General
// Public License, v. 3.0. If a copy of the GPL was not distributed
// with this file, You can obtain one at
// http://gnu.org/copyleft/gpl.txt.

// Provides a class for the fitting function. Instances of this class
// are used by LMsolver in the fitting procedure (one instance for
// each data set). The main members of FitFunction are the body of the
// fitting function and the parameter vector.

#pragma once

#include "automatic_differentiation.h"

#include <functional>
#include <vector>

namespace gadfit {

using fitSignature = std::function<AdVar(const std::vector<AdVar>&, double)>;

class FitFunction
{
private:
    fitSignature function;
    std::vector<AdVar> parameters;

public:
    FitFunction() = delete;
    FitFunction(fitSignature function) : function { std::move(function) } {}
    FitFunction(const FitFunction&) = default;
    FitFunction(FitFunction&&) = default;
    auto operator=(const FitFunction&) -> FitFunction& = default;
    auto operator=(FitFunction&&) -> FitFunction& = default;

    // Return reference to a parameter but resize first if
    // necessary. As such, there is no upper bound check.
    auto par(const int i_par) -> AdVar&;
    [[nodiscard]] auto getParValue(const int i_par) const -> double;
    // Number of all parameters (active or passive)
    [[nodiscard]] auto getNumPars() const -> int;
    // Activate a parameter for the reverse mode
    auto activateParReverse(const int par_i, const int idx) -> void;
    // Activate a parameter for the forward mode
    auto activateParForward(const int par_i) -> void;
    // Deactivate a parameter regardless of mode
    auto deactivatePar(const int par_i) -> void;

    // This is the central function. Evaluates the function body at a
    // given argument using parameters from the parameter vector.
    auto operator()(double arg) const -> AdVar;

    ~FitFunction() = default;
};

} // namespace gadfit

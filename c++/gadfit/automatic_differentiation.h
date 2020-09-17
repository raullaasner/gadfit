// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>

namespace gadfit {

// The AD variable
class AdVar
{
public:
    // Value and first and second derivatives of the AD variable
    // NOLINTNEXTLINE(modernize-use-default-member-init)
    double val, d, dd;
    // The index serves three purposes. If zero, this is an
    // inactive AD variable, so just compute the value and no
    // derivatives. If positive, we are in the reverse mode and if
    // negative in the forward mode of AD.
    // NOLINTNEXTLINE(modernize-use-default-member-init)
    int idx;

    // For performance reasons, don't use any default values.
    AdVar() = default;
    // NOLINTNEXTLINE
    AdVar(const double val) : val { val } {}
    constexpr AdVar(const double val,
                    const double d,
                    const double dd,
                    const int idx)
      : val { val }, d { d }, dd { dd }, idx { idx }
    {}
};

} // namespace gadfit

auto operator<<(std::ostream& out, const gadfit::AdVar& x) -> std::ostream&;

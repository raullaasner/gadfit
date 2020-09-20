// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>

namespace gadfit {

// If AdVar::idx has this value then it is an active variable in the
// forward mode.
constexpr int forward_active_idx { -1 };

// The AD variable
class AdVar
{
public:
    // Value and first and second derivatives of the AD variable
    // NOLINTNEXTLINE(modernize-use-default-member-init)
    double val, d, dd;
    // The index serves three purposes. If zero, this is an inactive
    // AD variable, so just compute the value and no derivatives. If
    // positive, we are in the reverse mode and if negative
    // (forward_active_idx) in the forward mode of AD.
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
    AdVar(const AdVar&) = default;
    AdVar(AdVar&&) = default;

    auto operator=(const AdVar& x) -> AdVar& = default;
    auto operator=(AdVar &&) -> AdVar& = default;
    template <typename T>
    auto operator=(const T x) -> AdVar&
    {
        val = static_cast<double>(x);
        return *this;
    }

    template <typename T>
    operator T() const { return static_cast<T>(val); }

    ~AdVar() = default;
};

template <typename T>
auto operator>(const AdVar& x, T y) -> bool
{
    return x.val > static_cast<double>(y);
}
auto operator>(const AdVar& x, const AdVar& y) -> bool;
template <typename T>
auto operator<(const AdVar& x, T y) -> bool
{
    return x.val < static_cast<double>(y);
}
auto operator<(const AdVar& x, const AdVar& y) -> bool;

} // namespace gadfit

auto operator<<(std::ostream& out, const gadfit::AdVar& x) -> std::ostream&;

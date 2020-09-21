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
    explicit AdVar(const double val) : val { val } {}
    explicit constexpr AdVar(const double val,
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
    operator T() const
    {
        return static_cast<T>(val);
    }

    auto operator-() const -> AdVar;
    auto operator+() const -> AdVar;

    ~AdVar() = default;
};

// void initialize_ad_reverse();

// Greater than
template <typename T>
auto operator>(const AdVar& x, const T y) -> bool
{
    return x.val > static_cast<double>(y);
}

template <typename T>
auto operator>(const T y, const AdVar& x) -> bool
{
    return static_cast<double>(y) > x.val;
}

auto operator>(const AdVar& x, const AdVar& y) -> bool;

// Less than
template <typename T>
auto operator<(const AdVar& x, const T y) -> bool
{
    return x.val < static_cast<double>(y);
}

template <typename T>
auto operator<(const T y, const AdVar& x) -> bool
{
    return static_cast<double>(y) < x.val;
}

auto operator<(const AdVar& x, const AdVar& y) -> bool;

// Addition
template <typename T>
auto operator+(const AdVar& x1, const T x2) -> AdVar
{
    AdVar y { x1.val + x2 };
    if (x1.idx < 0) {
        y.d = x1.d;
        y.dd = x1.dd;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

template <typename T>
auto operator+(const T x1, const AdVar& x2) -> AdVar
{
    AdVar y { x1 + x2.val };
    if (x2.idx < 0) {
        y.d = x2.d;
        y.dd = x2.dd;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto operator+(const AdVar& x1, const AdVar& x2) -> AdVar;

// Subtraction
template <typename T>
auto operator-(const AdVar& x1, const T x2) -> AdVar
{
    AdVar y { x1.val - x2 };
    if (x1.idx < 0) {
        y.d = x1.d;
        y.dd = x1.dd;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

template <typename T>
auto operator-(const T x1, const AdVar& x2) -> AdVar
{
    AdVar y { x1 - x2.val };
    if (x2.idx < 0) {
        y.d = -x2.d;
        y.dd = -x2.dd;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto operator-(const AdVar& x1, const AdVar& x2) -> AdVar;

// Multiplication
template <typename T>
auto operator*(const AdVar& x1, const T x2) -> AdVar
{
    AdVar y { x1.val * x2 };
    if (x1.idx == forward_active_idx) {
        y.d = x1.d * x2;
        y.dd = x1.dd * x2;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

template <typename T>
auto operator*(const T x1, const AdVar& x2) -> AdVar
{
    AdVar y { x1 * x2.val };
    if (x2.idx == forward_active_idx) {
        y.d = x1 * x2.d;
        y.dd = x1 * x2.dd;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto operator*(const AdVar& x1, const AdVar& x2) -> AdVar;

// Division
template <typename T>
auto operator/(const AdVar& x1, const T x2) -> AdVar
{
    AdVar y { x1.val / x2 };
    if (x1.idx == forward_active_idx) {
        y.d = x1.d / x2;
        y.dd = x1.dd / x2;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

template <typename T>
auto operator/(const T x1, const AdVar& x2) -> AdVar
{
    AdVar y { x1 / x2.val };
    if (x2.idx == forward_active_idx) {
        y.d = -y.val * x2.d / x2.val;
        y.dd = (-x2.dd * y.val - 2 * y.d * x2.d) / x2.val;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto operator/(const AdVar& x1, const AdVar& x2) -> AdVar;

} // namespace gadfit

auto operator<<(std::ostream& out, const gadfit::AdVar& x) -> std::ostream&;

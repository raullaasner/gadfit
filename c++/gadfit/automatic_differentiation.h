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

// Provides an implementation for both the forward and reverse modes
// of automatic differentiation. Defines the AD variable (AdVar), all
// operations between AD variables and other AD variables or
// fundamental types, and a few additional functions mainly related to
// initialization and memory management of the reverse mode.

#pragma once

#include <cmath>
#include <iostream>
#include <vector>

namespace gadfit {

// If AdVar::idx has this value then it is a passive variable in both
// modes. If its index is higher than passive_idx then it is active in
// the reverse mode.
constexpr int passive_idx { -1 };
// If AdVar::idx has this value then it is an active variable in the
// forward mode.
constexpr int forward_active_idx { -2 };

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

    // For performance reasons, don't use any default values in the
    // constructors.
    AdVar() = default;
    // NOLINTNEXTLINE
    explicit AdVar(const double val) : val { val } {}
    // NOLINTNEXTLINE
    explicit AdVar(const double val, const int idx) : val { val }, idx { idx }
    {}
    explicit constexpr AdVar(const double val,
                             const double d,
                             const double dd,
                             const int idx)
      : val { val }, d { d }, dd { dd }, idx { idx }
    {}
    AdVar(const AdVar&) = default;
    AdVar(AdVar&&) = default;

    auto operator=(const AdVar&) -> AdVar& = default;
    auto operator=(AdVar&&) -> AdVar& = default;
    template <typename T>
    auto operator=(const T x) -> AdVar&
    {
        val = static_cast<double>(x);
        return *this;
    }

    auto operator-() const -> AdVar;
    auto operator+() const -> AdVar;

    template <typename T>
    auto operator-=(const T& rhs) -> AdVar&;
    template <typename T>
    auto operator+=(const T& rhs) -> AdVar&;

    ~AdVar() = default;
};

// Work arrays for the reverse mode
namespace reverse {

// Value of index assigned to the most recent AD variable
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern int last_index;
// Forward values
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern std::vector<double> forwards;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern std::vector<double> adjoints;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern std::vector<double> constants;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern int const_count;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern std::vector<int> trace;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern int trace_count;

} // namespace reverse

// Operation codes
enum class Op
{
    add_a_a,
    add_subtract_a_r,
    subtract_a_a,
    subtract_r_a,
    multiply_a_a,
    multiply_divide_a_r,
    divide_a_a,
    divide_r_a,
    pow_a_a,
    pow_a_r,
    pow_r_a,
    log_a,
    exp_a,
    sqrt_a,
    abs_a,
    sin_a,
    cos_a,
    tan_a,
    asin_a,
    acos_a,
    atan_a,
    sinh_a,
    cosh_a,
    tanh_a,
    asinh_a,
    acosh_a,
    atanh_a,
    erf_a,
    integration_bound,
};

constexpr int default_sweep_size { 1000 };

auto initializeADReverse(const int sweep_size = default_sweep_size) -> void;

auto adResetSweep() -> void;

auto addADSeed(const AdVar& x) -> void;

// BEGIN AD ELEMENTAL OPERATIONS

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
    if (x1.idx > passive_idx) {
        y.idx = ++reverse::last_index;
        reverse::forwards[reverse::last_index] = y.val;
        reverse::trace[++reverse::trace_count] = x1.idx;
        reverse::trace[++reverse::trace_count] = y.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::add_subtract_a_r);
    } else if (x1.idx == forward_active_idx) {
        y.d = x1.d;
        y.dd = x1.dd;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

template <typename T>
auto operator+(const T x1, const AdVar& x2) -> AdVar
{
    AdVar y { x1 + x2.val };
    if (x2.idx > passive_idx) {
        y.idx = ++reverse::last_index;
        reverse::forwards[reverse::last_index] = y.val;
        reverse::trace[++reverse::trace_count] = x2.idx;
        reverse::trace[++reverse::trace_count] = y.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::add_subtract_a_r);
    } else if (x2.idx == forward_active_idx) {
        y.d = x2.d;
        y.dd = x2.dd;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

template <typename T>
auto AdVar::operator+=(const T& rhs) -> AdVar&
{
    *this = *this + rhs;
    return *this;
}

auto operator+(const AdVar& x1, const AdVar& x2) -> AdVar;

// Subtraction
template <typename T>
// LCOV_EXCL_START
auto operator-(const AdVar& x1, const T x2) -> AdVar
{
    AdVar y { x1.val - x2 };
    if (x1.idx > passive_idx) {
        y.idx = ++reverse::last_index;
        reverse::forwards[reverse::last_index] = y.val;
        reverse::trace[++reverse::trace_count] = x1.idx;
        reverse::trace[++reverse::trace_count] = y.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::add_subtract_a_r);
    } else if (x1.idx == forward_active_idx) {
        y.d = x1.d;
        y.dd = x1.dd;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}
// LCOV_EXCL_STOP

template <typename T>
auto operator-(const T x1, const AdVar& x2) -> AdVar
{
    AdVar y { x1 - x2.val };
    if (x2.idx > passive_idx) {
        y.idx = ++reverse::last_index;
        reverse::forwards[reverse::last_index] = y.val;
        reverse::trace[++reverse::trace_count] = x2.idx;
        reverse::trace[++reverse::trace_count] = y.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::subtract_r_a);
    } else if (x2.idx == forward_active_idx) {
        y.d = -x2.d;
        y.dd = -x2.dd;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

template <typename T>
auto AdVar::operator-=(const T& rhs) -> AdVar&
{
    *this = *this - rhs;
    return *this;
}

auto operator-(const AdVar& x1, const AdVar& x2) -> AdVar;

// Multiplication
template <typename T>
// LCOV_EXCL_START
auto operator*(const AdVar& x1, const T x2) -> AdVar
{
    AdVar y { x1.val * x2 };
    if (x1.idx > passive_idx) {
        y.idx = ++reverse::last_index;
        reverse::forwards[reverse::last_index] = y.val;
        reverse::constants[++reverse::const_count] = static_cast<double>(x2);
        reverse::trace[++reverse::trace_count] = x1.idx;
        reverse::trace[++reverse::trace_count] = y.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::multiply_divide_a_r);
    } else if (x1.idx == forward_active_idx) {
        y.d = x1.d * x2;
        y.dd = x1.dd * x2;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}
// LCOV_EXCL_STOP

template <typename T>
// LCOV_EXCL_START
auto operator*(const T x1, const AdVar& x2) -> AdVar
{
    AdVar y { x1 * x2.val };
    if (x2.idx > passive_idx) {
        y.idx = ++reverse::last_index;
        reverse::forwards[reverse::last_index] = y.val;
        reverse::constants[++reverse::const_count] = static_cast<double>(x1);
        reverse::trace[++reverse::trace_count] = x2.idx;
        reverse::trace[++reverse::trace_count] = y.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::multiply_divide_a_r);
    } else if (x2.idx == forward_active_idx) {
        y.d = x1 * x2.d;
        y.dd = x1 * x2.dd;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}
// LCOV_EXCL_STOP

auto operator*(const AdVar& x1, const AdVar& x2) -> AdVar;

// Division
template <typename T>
auto operator/(const AdVar& x1, const T x2) -> AdVar
{
    const double& inv_x2 { 1.0 / x2 };
    AdVar y { x1.val * inv_x2 };
    if (x1.idx > passive_idx) {
        y.idx = ++reverse::last_index;
        reverse::forwards[reverse::last_index] = y.val;
        reverse::constants[++reverse::const_count] = inv_x2;
        reverse::trace[++reverse::trace_count] = x1.idx;
        reverse::trace[++reverse::trace_count] = y.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::multiply_divide_a_r);
    } else if (x1.idx == forward_active_idx) {
        y.d = x1.d * inv_x2;
        y.dd = x1.dd * inv_x2;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

template <typename T>
auto operator/(const T x1, const AdVar& x2) -> AdVar
{
    const double& inv_x2 { 1.0 / x2.val };
    AdVar y { x1 * inv_x2 };
    if (x2.idx > passive_idx) {
        y.idx = ++reverse::last_index;
        reverse::forwards[reverse::last_index] = y.val;
        reverse::constants[++reverse::const_count] = static_cast<double>(x1);
        reverse::trace[++reverse::trace_count] = x2.idx;
        reverse::trace[++reverse::trace_count] = y.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::divide_r_a);
    } else if (x2.idx == forward_active_idx) {
        y.d = -y.val * x2.d * inv_x2;
        y.dd = (-x2.dd * y.val - 2 * y.d * x2.d) * inv_x2;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto operator/(const AdVar& x1, const AdVar& x2) -> AdVar;

// Exponentiation
auto pow(const AdVar& x1, const AdVar& x2) -> AdVar;

template <typename T>
auto pow(const AdVar& x1, const T x2) -> AdVar
{
    AdVar y { std::pow(x1.val, x2) };
    if (x1.idx > passive_idx) {
        y.idx = ++reverse::last_index;
        reverse::forwards[reverse::last_index] = y.val;
        reverse::constants[++reverse::const_count] = static_cast<double>(x2);
        reverse::trace[++reverse::trace_count] = x1.idx;
        reverse::trace[++reverse::trace_count] = y.idx;
        reverse::trace[++reverse::trace_count] = static_cast<int>(Op::pow_a_r);
    } else if (x1.idx == forward_active_idx) {
        y.d = x1.d * x2 * std::pow(x1.val, x2 - 1);
        const double& inv_x1 { 1.0 / x1.val };
        y.dd = y.d * y.d / y.val
               + y.val * x2 * (x1.dd - x1.d * x1.d * inv_x1) * inv_x1;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

template <typename T>
auto pow(const T x1, const AdVar& x2) -> AdVar
{
    AdVar y { std::pow(x1, x2.val) };
    if (x2.idx > passive_idx) {
        y.idx = ++reverse::last_index;
        reverse::forwards[reverse::last_index] = y.val;
        reverse::constants[++reverse::const_count] = static_cast<double>(x1);
        reverse::trace[++reverse::trace_count] = x2.idx;
        reverse::trace[++reverse::trace_count] = y.idx;
        reverse::trace[++reverse::trace_count] = static_cast<int>(Op::pow_r_a);
    } else if (x2.idx == forward_active_idx) {
        const double& log_value { std::log(x1) };
        y.d = y.val * x2.d * log_value;
        y.dd = y.d * y.d / y.val + y.val * x2.dd * log_value;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

// Logarithm, exponential, other
auto log(const AdVar& x) -> AdVar;
auto exp(const AdVar& x) -> AdVar;
auto sqrt(const AdVar& x) -> AdVar;
auto abs(const AdVar& x) -> AdVar;

// Trigonometric
auto sin(const AdVar& x) -> AdVar;
auto cos(const AdVar& x) -> AdVar;
auto tan(const AdVar& x) -> AdVar;
auto asin(const AdVar& x) -> AdVar;
auto acos(const AdVar& x) -> AdVar;
auto atan(const AdVar& x) -> AdVar;
auto sinh(const AdVar& x) -> AdVar;
auto cosh(const AdVar& x) -> AdVar;
auto tanh(const AdVar& x) -> AdVar;
auto asinh(const AdVar& x) -> AdVar;
auto acosh(const AdVar& x) -> AdVar;
auto atanh(const AdVar& x) -> AdVar;

// Special
auto erf(const AdVar& x) -> AdVar;

// END AD ELEMENTAL OPERATIONS

// Reverse mode specific
auto returnSweep() -> void;
auto freeAdReverse() -> void;

} // namespace gadfit

// Utilities
auto operator<<(std::ostream& out, const gadfit::AdVar& x) -> std::ostream&;

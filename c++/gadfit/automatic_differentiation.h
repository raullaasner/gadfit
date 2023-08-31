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
// of automatic differentiation. Defines the AD variable (AdVar) and
// all operations between AD variables and other AD variables or
// fundamental types.

#pragma once

#include <cmath>
#include <iostream>
#include <vector>

namespace gadfit {

template <typename T>
concept Number = std::is_arithmetic_v<T>;

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

    // For performance reasons don't use any default values in the
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
    auto operator=(const Number auto x) -> AdVar&
    {
        val = static_cast<double>(x);
        return *this;
    }

    auto operator-() const -> AdVar;
    auto operator+() const -> AdVar;

    auto operator-=(const auto& rhs) -> AdVar&;
    auto operator+=(const auto& rhs) -> AdVar&;

    ~AdVar() = default;
};

// AD tape (work arrays for the reverse mode). These arrays are
// resized by repeatedly calling push_back and cleared at the end of
// the return sweep. In a fitting procedure, memory is only allocated
// when evaluating the first function call. For all other data points
// the vector capacities remain constant and calling push_back incurs
// minimal overhead. If function evaluations and the return sweep are
// called correctly and everything went well, the trace is empty and
// the forwards array contains a few elements. For that reason there
// is no memory deallocation function. If you need to properly
// deallocate both variables (set their capacities to 0) you'd need to
// do that manually.
namespace reverse {

// Forward values
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local extern std::vector<double> forwards;
// Execution trace tracking the operation codes and variable indices
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local extern std::vector<int> trace;

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

// Start recording to the tape with variable initializations. Each
// call records a forward value.
auto addADSeed(const AdVar& x) -> void;

// BEGIN AD ELEMENTAL OPERATIONS

// Greater than
auto operator>(const AdVar& x, const Number auto y) -> bool
{
    return x.val > static_cast<double>(y);
}

auto operator>(const Number auto y, const AdVar& x) -> bool
{
    return static_cast<double>(y) > x.val;
}

auto operator>(const AdVar& x, const AdVar& y) -> bool;

// Less than
auto operator<(const AdVar& x, const Number auto y) -> bool
{
    return x.val < static_cast<double>(y);
}

auto operator<(const Number auto y, const AdVar& x) -> bool
{
    return static_cast<double>(y) < x.val;
}

auto operator<(const AdVar& x, const AdVar& y) -> bool;

// Addition
auto operator+(const AdVar& x1, const Number auto x2) -> AdVar
{
    AdVar y { x1.val + x2 };
    if (x1.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::add_subtract_a_r));
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

auto operator+(const Number auto x1, const AdVar& x2) -> AdVar
{
    AdVar y { x1 + x2.val };
    if (x2.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::add_subtract_a_r));
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

auto AdVar::operator+=(const auto& rhs) -> AdVar&
{
    *this = *this + rhs;
    return *this;
}

auto operator+(const AdVar& x1, const AdVar& x2) -> AdVar;

// Subtraction
// LCOV_EXCL_START
auto operator-(const AdVar& x1, const Number auto x2) -> AdVar
{
    AdVar y { x1.val - x2 };
    if (x1.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::add_subtract_a_r));
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

auto operator-(const Number auto x1, const AdVar& x2) -> AdVar
{
    AdVar y { x1 - x2.val };
    if (x2.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::subtract_r_a));
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

auto AdVar::operator-=(const auto& rhs) -> AdVar&
{
    *this = *this - rhs;
    return *this;
}

auto operator-(const AdVar& x1, const AdVar& x2) -> AdVar;

// Multiplication
// LCOV_EXCL_START
auto operator*(const AdVar& x1, const Number auto x2) -> AdVar
{
    AdVar y { x1.val * x2 };
    if (x1.idx > passive_idx) {
        reverse::forwards.push_back(static_cast<double>(x2));
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::multiply_divide_a_r));
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

// LCOV_EXCL_START
auto operator*(const Number auto x1, const AdVar& x2) -> AdVar
{
    AdVar y { x1 * x2.val };
    if (x2.idx > passive_idx) {
        reverse::forwards.push_back(static_cast<double>(x1));
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::multiply_divide_a_r));
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
auto operator/(const AdVar& x1, const Number auto x2) -> AdVar
{
    const double& inv_x2 { 1.0 / x2 };
    AdVar y { x1.val * inv_x2 };
    if (x1.idx > passive_idx) {
        reverse::forwards.push_back(inv_x2);
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::multiply_divide_a_r));
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

auto operator/(const Number auto x1, const AdVar& x2) -> AdVar
{
    const double& inv_x2 { 1.0 / x2.val };
    AdVar y { x1 * inv_x2 };
    if (x2.idx > passive_idx) {
        reverse::forwards.push_back(static_cast<double>(x1));
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::divide_r_a));
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

auto pow(const AdVar& x1, const Number auto x2) -> AdVar
{
    AdVar y { std::pow(x1.val, x2) };
    if (x1.idx > passive_idx) {
        reverse::forwards.push_back(static_cast<double>(x2));
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::pow_a_r));
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

auto pow(const Number auto x1, const AdVar& x2) -> AdVar
{
    AdVar y { std::pow(x1, x2.val) };
    if (x2.idx > passive_idx) {
        reverse::forwards.push_back(static_cast<double>(x1));
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::pow_r_a));
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

// Run the return sweep. The derivatives are stored in the first few
// elements of the adjoint vector. After each return sweep the AD tape
// needs to be rewound. However, we don't need to clear all the
// forward values because the first few ones, representing variable
// initializations, don't typically change and can be reused in the
// next function evaluation and return sweep. In a fitting procedure
// with e.g. 3 fitting parameters, the rewinding index should be set
// to 2 (the last fitting parameter).
auto returnSweep(const int rewind_index, std::vector<double>& adjoints) -> void;

} // namespace gadfit

// Utilities
auto operator<<(std::ostream& out, const gadfit::AdVar& x) -> std::ostream&;

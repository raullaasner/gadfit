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

#include "automatic_differentiation.h"

#include "exceptions.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <numbers>

namespace gadfit {

namespace reverse {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local std::vector<double> forwards;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local std::vector<int> trace;

} // namespace reverse

auto addADSeed(const AdVar& x) -> void
{
    reverse::forwards.resize(x.idx + 1);
    reverse::forwards.back() = x.val;
}

auto AdVar::operator-() const -> AdVar
{
    return -1 * (*this);
}

auto AdVar::operator+() const -> AdVar
{
    return *this;
}

auto operator>(const AdVar& x, const AdVar& y) -> bool
{
    return x.val > y.val;
}

auto operator<(const AdVar& x, const AdVar& y) -> bool
{
    return x.val < y.val;
}

auto operator+(const AdVar& x1, const AdVar& x2) -> AdVar
{
    // Can't have one variable active in reverse mode and the other
    // one active in forward mode.
    assert(!(x1.idx > passive_idx && x2.idx == forward_active_idx));
    assert(!(x1.idx == forward_active_idx && x2.idx > passive_idx));
    AdVar y { x1.val + x2.val };
    if (x1.idx > passive_idx && x2.idx > passive_idx) {
        // Index of the last element added to forward values is always
        // related to the size of the forward values array.
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::add_a_a));
    } else if (x1.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::add_subtract_a_r));
    } else if (x2.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::add_subtract_a_r));
    } else if (x1.idx == forward_active_idx && x2.idx == forward_active_idx) {
        y.d = x1.d + x2.d;
        y.dd = x1.dd + x2.dd;
        y.idx = forward_active_idx;
    } else if (x1.idx == forward_active_idx) {
        y.d = x1.d;
        y.dd = x1.dd;
        y.idx = forward_active_idx;
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

auto operator-(const AdVar& x1, const AdVar& x2) -> AdVar
{
    assert(!(x1.idx > passive_idx && x2.idx == forward_active_idx));
    assert(!(x1.idx == forward_active_idx && x2.idx > passive_idx));
    AdVar y { x1.val - x2.val };
    if (x1.idx > passive_idx && x2.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::subtract_a_a));
    } else if (x1.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::add_subtract_a_r));
    } else if (x2.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::subtract_r_a));
    } else if (x1.idx == forward_active_idx && x2.idx == forward_active_idx) {
        y.d = x1.d - x2.d;
        y.dd = x1.dd - x2.dd;
        y.idx = forward_active_idx;
    } else if (x1.idx == forward_active_idx) {
        y.d = x1.d;
        y.dd = x1.dd;
        y.idx = forward_active_idx;
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

auto operator*(const AdVar& x1, const AdVar& x2) -> AdVar
{
    assert(!(x1.idx > passive_idx && x2.idx == forward_active_idx));
    assert(!(x1.idx == forward_active_idx && x2.idx > passive_idx));
    AdVar y { x1.val * x2.val };
    if (x1.idx > passive_idx && x2.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::multiply_a_a));
    } else if (x1.idx > passive_idx) {
        // Typically we start by setting the index and value of the
        // result (y). However, if there is a constant involved we
        // save that first. It can be referenced in the return sweep
        // as forwards[trace[i_tr-1]-1] where trace[i_tr-1] is the
        // index of y and by subtracting 1 we refer to the value that
        // precedes y in the forward array, i.e. the constant.
        reverse::forwards.push_back(x2.val);
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::multiply_divide_a_r));
    } else if (x2.idx > passive_idx) {
        reverse::forwards.push_back(x1.val);
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::multiply_divide_a_r));
    } else if (x1.idx == forward_active_idx && x2.idx == forward_active_idx) {
        y.d = x1.d * x2.val + x1.val * x2.d;
        y.dd = x1.dd * x2.val + 2 * x1.d * x2.d + x1.val * x2.dd;
        y.idx = forward_active_idx;
    } else if (x1.idx == forward_active_idx) {
        y.d = x1.d * x2.val;
        y.dd = x1.dd * x2.val;
        y.idx = forward_active_idx;
    } else if (x2.idx == forward_active_idx) {
        y.d = x1.val * x2.d;
        y.dd = x1.val * x2.dd;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto operator/(const AdVar& x1, const AdVar& x2) -> AdVar
{
    assert(!(x1.idx > passive_idx && x2.idx == forward_active_idx));
    assert(!(x1.idx == forward_active_idx && x2.idx > passive_idx));
    const double inv_x2 { 1.0 / x2.val };
    AdVar y { x1.val * inv_x2 };
    if (x1.idx > passive_idx && x2.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::divide_a_a));
    } else if (x1.idx > passive_idx) {
        reverse::forwards.push_back(inv_x2);
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::multiply_divide_a_r));
    } else if (x2.idx > passive_idx) {
        reverse::forwards.push_back(x1.val);
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::divide_r_a));
    } else if (x1.idx == forward_active_idx && x2.idx == forward_active_idx) {
        y.d = (x1.d - x2.d * y.val) * inv_x2;
        y.dd = (x1.dd - x2.dd * y.val - 2 * y.d * x2.d) * inv_x2;
        y.idx = forward_active_idx;
    } else if (x1.idx == forward_active_idx) {
        y.d = x1.d * inv_x2;
        y.dd = x1.dd * inv_x2;
        y.idx = forward_active_idx;
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

auto pow(const AdVar& x1, const AdVar& x2) -> AdVar
{
    assert(!(x1.idx > passive_idx && x2.idx == forward_active_idx));
    assert(!(x1.idx == forward_active_idx && x2.idx > passive_idx));
    AdVar y { std::pow(x1.val, x2.val) };
    if (x1.idx > passive_idx && x2.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::pow_a_a));
    } else if (x1.idx > passive_idx) {
        reverse::forwards.push_back(x2.val);
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x1.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::pow_a_r));
    } else if (x2.idx > passive_idx) {
        reverse::forwards.push_back(x1.val);
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x2.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::pow_r_a));
    } else if (x1.idx == forward_active_idx && x2.idx == forward_active_idx) {
        const double log_value { std::log(x1.val) };
        y.d = x1.d * x2.val * std::pow(x1.val, x2.val - 1)
              + x2.d * log_value * y.val;
        const double inv_x1 { 1.0 / x1.val };
        y.dd =
          y.d * y.d / y.val
          + y.val
              * (x2.dd * log_value
                 + (2 * x1.d * x2.d + x2.val * (x1.dd - x1.d * x1.d * inv_x1))
                     * inv_x1);
        y.idx = forward_active_idx;
    } else if (x1.idx == forward_active_idx) {
        y.d = x1.d * x2.val * std::pow(x1.val, x2.val - 1);
        const double inv_x1 { 1.0 / x1.val };
        y.dd = y.d * y.d / y.val
               + y.val * x2.val * (x1.dd - x1.d * x1.d * inv_x1) * inv_x1;
        y.idx = forward_active_idx;
    } else if (x2.idx == forward_active_idx) {
        const double log_value { std::log(x1.val) };
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

auto log(const AdVar& x) -> AdVar
{
    AdVar y { std::log(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::log_a));
    } else if (x.idx == forward_active_idx) {
        const double inv_x { 1.0 / x.val };
        y.d = x.d * inv_x;
        y.dd = (x.dd - x.d * y.d) * inv_x;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto exp(const AdVar& x) -> AdVar
{
    AdVar y { std::exp(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::exp_a));
    } else if (x.idx == forward_active_idx) {
        y.d = x.d * y.val;
        y.dd = x.dd * y.val + x.d * y.d;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto sqrt(const AdVar& x) -> AdVar
{
    AdVar y { std::sqrt(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::sqrt_a));
    } else if (x.idx == forward_active_idx) {
        const double inv_y { 1.0 / y.val };
        y.d = x.d / 2 * inv_y;
        y.dd = (x.dd * inv_y - y.d * x.d / x.val) / 2;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto abs(const AdVar& x) -> AdVar
{
    AdVar y { std::abs(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::abs_a));
    } else if (x.idx == forward_active_idx) {
        y.d = x.d * (std::signbit(x.val) ? -1 : 1);
        y.dd = x.dd * (std::signbit(x.val) ? -1 : 1);
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto sin(const AdVar& x) -> AdVar
{
    AdVar y { std::sin(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::sin_a));
    } else if (x.idx == forward_active_idx) {
        const double cos_value { std::cos(x.val) };
        y.d = x.d * cos_value;
        y.dd = x.dd * cos_value - x.d * x.d * y.val;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto cos(const AdVar& x) -> AdVar
{
    AdVar y { std::cos(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::cos_a));
    } else if (x.idx == forward_active_idx) {
        const double sin_value { -std::sin(x.val) };
        y.d = x.d * sin_value;
        y.dd = x.dd * sin_value - x.d * x.d * y.val;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto tan(const AdVar& x) -> AdVar
{
    AdVar y { std::tan(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::tan_a));
    } else if (x.idx == forward_active_idx) {
        const double inv_cos_value { 1.0 / std::cos(x.val) };
        const double inv_cos_value2 { inv_cos_value * inv_cos_value };
        y.d = x.d * inv_cos_value2;
        y.dd = x.dd * inv_cos_value2 + 2 * x.d * y.val * y.d;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto asin(const AdVar& x) -> AdVar
{
    AdVar y { std::asin(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::asin_a));
    } else if (x.idx == forward_active_idx) {
        const double tmp { 1.0 / std::sqrt(1 - x.val * x.val) };
        y.d = x.d * tmp;
        y.dd = tmp * (x.dd + x.val * y.d * y.d);
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto acos(const AdVar& x) -> AdVar
{
    AdVar y { std::acos(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::acos_a));
    } else if (x.idx == forward_active_idx) {
        const double tmp { -1.0 / std::sqrt(1 - x.val * x.val) };
        y.d = x.d * tmp;
        y.dd = tmp * (x.dd + x.val * y.d * y.d);
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto atan(const AdVar& x) -> AdVar
{
    AdVar y { std::atan(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::atan_a));
    } else if (x.idx == forward_active_idx) {
        const double tmp { 1.0 / (1 + x.val * x.val) };
        y.d = x.d * tmp;
        y.dd = x.dd * tmp - 2 * x.val * y.d * y.d;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto sinh(const AdVar& x) -> AdVar
{
    AdVar y { std::sinh(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::sinh_a));
    } else if (x.idx == forward_active_idx) {
        const double cosh_value { std::cosh(x.val) };
        y.d = x.d * cosh_value;
        y.dd = x.dd * cosh_value + x.d * x.d * y.val;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto cosh(const AdVar& x) -> AdVar
{
    AdVar y { std::cosh(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::cosh_a));
    } else if (x.idx == forward_active_idx) {
        const double sinh_value { std::sinh(x.val) };
        y.d = x.d * sinh_value;
        y.dd = x.dd * sinh_value + x.d * x.d * y.val;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto tanh(const AdVar& x) -> AdVar
{
    AdVar y { std::tanh(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::tanh_a));
    } else if (x.idx == forward_active_idx) {
        const double inv_cosh_value { 1.0 / std::cosh(x.val) };
        const double inv_cosh_value2 { inv_cosh_value * inv_cosh_value };
        y.d = x.d * inv_cosh_value2;
        y.dd = x.dd * inv_cosh_value2 - 2 * x.d * y.val * y.d;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto asinh(const AdVar& x) -> AdVar
{
    AdVar y { std::asinh(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::asinh_a));
    } else if (x.idx == forward_active_idx) {
        const double tmp { 1.0 / std::sqrt(x.val * x.val + 1) };
        y.d = x.d * tmp;
        y.dd = (x.dd - x.val * y.d * y.d) * tmp;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto acosh(const AdVar& x) -> AdVar
{
    AdVar y { std::acosh(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::acosh_a));
    } else if (x.idx == forward_active_idx) {
        const double tmp { 1.0 / std::sqrt(x.val * x.val - 1) };
        y.d = x.d * tmp;
        y.dd = (x.dd - x.val * y.d * y.d) * tmp;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto atanh(const AdVar& x) -> AdVar
{
    AdVar y { std::atanh(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::atanh_a));
    } else if (x.idx == forward_active_idx) {
        const double tmp { 1.0 / (1 - x.val * x.val) };
        y.d = x.d * tmp;
        y.dd = (x.dd + 2 * x.val * x.d * y.d) * tmp;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto erf(const AdVar& x) -> AdVar
{
    AdVar y { std::erf(x.val) };
    if (x.idx > passive_idx) {
        y.idx = static_cast<int>(reverse::forwards.size());
        reverse::forwards.push_back(y.val);
        reverse::trace.push_back(x.idx);
        reverse::trace.push_back(y.idx);
        reverse::trace.push_back(static_cast<int>(Op::erf_a));
    } else if (x.idx == forward_active_idx) {
        const double tmp { 2 * std::numbers::inv_sqrtpi
                           * std::exp(-x.val * x.val) };
        y.d = x.d * tmp;
        y.dd = (x.dd - 2 * x.d * x.d * x.val) * tmp;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = passive_idx;
    }
    return y;
}

auto returnSweep(const int rewind_index, std::vector<double>& adjoints) -> void
{
    using reverse::forwards;
    using reverse::trace;
    adjoints.assign(forwards.size(), 0.0);
    if (!adjoints.empty()) {
        adjoints.back() = 1.0;
    }
    // Current position on the AD trace
    int i_tr { static_cast<int>(trace.size()) - 1 };
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    double tmp;
    while (i_tr > 0) {
        switch (static_cast<Op>(trace[i_tr])) {
        case Op::add_a_a:
            adjoints[trace[i_tr - 3]] += adjoints[trace[i_tr - 1]];
            adjoints[trace[i_tr - 2]] += adjoints[trace[i_tr - 1]];
            i_tr -= 4;
            break;
        case Op::add_subtract_a_r:
            adjoints[trace[i_tr - 2]] += adjoints[trace[i_tr - 1]];
            i_tr -= 3;
            break;
        case Op::subtract_a_a:
            adjoints[trace[i_tr - 3]] += adjoints[trace[i_tr - 1]];
            adjoints[trace[i_tr - 2]] -= adjoints[trace[i_tr - 1]];
            i_tr -= 4;
            break;
        case Op::subtract_r_a:
            adjoints[trace[i_tr - 2]] -= adjoints[trace[i_tr - 1]];
            i_tr -= 3;
            break;
        case Op::multiply_a_a:
            adjoints[trace[i_tr - 3]] +=
              adjoints[trace[i_tr - 1]] * forwards[trace[i_tr - 2]];
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] * forwards[trace[i_tr - 3]];
            i_tr -= 4;
            break;
        case Op::multiply_divide_a_r:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] * forwards[trace[i_tr - 1] - 1];
            i_tr -= 3;
            break;
        case Op::divide_a_a:
            adjoints[trace[i_tr - 3]] +=
              adjoints[trace[i_tr - 1]] / forwards[trace[i_tr - 2]];
            adjoints[trace[i_tr - 2]] -= adjoints[trace[i_tr - 1]]
                                         * forwards[trace[i_tr - 1]]
                                         / forwards[trace[i_tr - 2]];
            i_tr -= 4;
            break;
        case Op::divide_r_a:
            adjoints[trace[i_tr - 2]] -=
              adjoints[trace[i_tr - 1]] * forwards[trace[i_tr - 1] - 1]
              / forwards[trace[i_tr - 2]] / forwards[trace[i_tr - 2]];
            i_tr -= 3;
            break;
        case Op::pow_a_a:
            adjoints[trace[i_tr - 3]] +=
              adjoints[trace[i_tr - 1]] * forwards[trace[i_tr - 2]]
              * std::pow(forwards[trace[i_tr - 3]],
                         forwards[trace[i_tr - 2]] - 1);
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] * std::log(forwards[trace[i_tr - 3]])
              * std::pow(forwards[trace[i_tr - 3]], forwards[trace[i_tr - 2]]);
            i_tr -= 4;
            break;
        case Op::pow_a_r:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] * forwards[trace[i_tr - 1] - 1]
              * std::pow(forwards[trace[i_tr - 2]],
                         forwards[trace[i_tr - 1] - 1] - 1);
            i_tr -= 3;
            break;
        case Op::pow_r_a:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]]
              * std::log(forwards[trace[i_tr - 1] - 1])
              * std::pow(forwards[trace[i_tr - 1] - 1],
                         forwards[trace[i_tr - 2]]);
            i_tr -= 3;
            break;
        case Op::log_a:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] / forwards[trace[i_tr - 2]];
            i_tr -= 3;
            break;
        case Op::exp_a:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] * forwards[trace[i_tr - 1]];
            i_tr -= 3;
            break;
        case Op::sqrt_a:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] / 2 / forwards[trace[i_tr - 1]];
            i_tr -= 3;
            break;
        case Op::abs_a:
            if (forwards[trace[i_tr - 2]] < 0.0) {
                adjoints[trace[i_tr - 2]] -= adjoints[trace[i_tr - 1]];
            } else {
                adjoints[trace[i_tr - 2]] += adjoints[trace[i_tr - 1]];
            }
            i_tr -= 3;
            break;
        case Op::sin_a:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] * std::cos(forwards[trace[i_tr - 2]]);
            i_tr -= 3;
            break;
        case Op::cos_a:
            adjoints[trace[i_tr - 2]] -=
              adjoints[trace[i_tr - 1]] * std::sin(forwards[trace[i_tr - 2]]);
            i_tr -= 3;
            break;
        case Op::tan_a:
            tmp = std::cos(forwards[trace[i_tr - 2]]);
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] / (tmp * tmp);
            i_tr -= 3;
            break;
        case Op::asin_a:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]]
              / std::sqrt(
                1 - forwards[trace[i_tr - 2]] * forwards[trace[i_tr - 2]]);
            i_tr -= 3;
            break;
        case Op::acos_a:
            adjoints[trace[i_tr - 2]] -=
              adjoints[trace[i_tr - 1]]
              / std::sqrt(
                1 - forwards[trace[i_tr - 2]] * forwards[trace[i_tr - 2]]);
            i_tr -= 3;
            break;
        case Op::atan_a:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]]
              / (1 + forwards[trace[i_tr - 2]] * forwards[trace[i_tr - 2]]);
            i_tr -= 3;
            break;
        case Op::sinh_a:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] * std::cosh(forwards[trace[i_tr - 2]]);
            i_tr -= 3;
            break;
        case Op::cosh_a:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] * std::sinh(forwards[trace[i_tr - 2]]);
            i_tr -= 3;
            break;
        case Op::tanh_a:
            tmp = std::cosh(forwards[trace[i_tr - 2]]);
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] / (tmp * tmp);
            i_tr -= 3;
            break;
        case Op::asinh_a:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]]
              / std::sqrt(forwards[trace[i_tr - 2]] * forwards[trace[i_tr - 2]]
                          + 1);
            i_tr -= 3;
            break;
        case Op::acosh_a:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]]
              / std::sqrt(forwards[trace[i_tr - 2]] * forwards[trace[i_tr - 2]]
                          - 1);
            i_tr -= 3;
            break;
        case Op::atanh_a:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]]
              / (1 - forwards[trace[i_tr - 2]] * forwards[trace[i_tr - 2]]);
            i_tr -= 3;
            break;
        case Op::erf_a:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] * 2 * std::numbers::inv_sqrtpi
              * std::exp(-forwards[trace[i_tr - 2]]
                         * forwards[trace[i_tr - 2]]);
            i_tr -= 3;
            break;
        case Op::integration_bound:
            adjoints[trace[i_tr - 2]] +=
              adjoints[trace[i_tr - 1]] * forwards[trace[i_tr - 3]];
            i_tr -= 4;
            break;
        default:
            throw UnknownOperation { trace[i_tr] };
        }
    }
    trace.clear();
    forwards.resize(rewind_index + 1);
}

} // namespace gadfit

auto operator<<(std::ostream& out, const gadfit::AdVar& x) -> std::ostream&
{
    constexpr int precision { 16 };
    out << std::setprecision(precision) << "val=" << x.val << " d=" << x.d
        << " dd=" << x.dd << " idx=" << x.idx;
    return out;
}

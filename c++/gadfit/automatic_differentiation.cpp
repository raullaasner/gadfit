// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "automatic_differentiation.h"

#include <iomanip>
#include <iostream>
#include <numbers>

using gadfit::AdVar;

auto AdVar::operator-() const -> AdVar
{
    return -1 * (*this);
}

auto AdVar::operator+() const -> AdVar
{
    return *this;
}

auto gadfit::operator>(const AdVar& x, const AdVar& y) -> bool
{
    return x.val > y.val;
}

auto gadfit::operator<(const AdVar& x, const AdVar& y) -> bool
{
    return x.val < y.val;
}

auto gadfit::operator+(const AdVar& x1, const AdVar& x2) -> AdVar
{
    AdVar y { x1.val + x2.val };
    if (x1.idx == forward_active_idx && x2.idx == forward_active_idx) {
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
        y.idx = 0;
    }
    return y;
}

auto gadfit::operator-(const AdVar& x1, const AdVar& x2) -> AdVar
{
    AdVar y { x1.val - x2.val };
    if (x1.idx == forward_active_idx && x2.idx == forward_active_idx) {
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
        y.idx = 0;
    }
    return y;
}

auto gadfit::operator*(const AdVar& x1, const AdVar& x2) -> AdVar
{
    AdVar y { x1.val * x2.val };
    if (x1.idx == forward_active_idx && x2.idx == forward_active_idx) {
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
        y.idx = 0;
    }
    return y;
}

auto gadfit::operator/(const AdVar& x1, const AdVar& x2) -> AdVar
{
    const double& inv_x2 { 1.0 / x2.val };
    AdVar y { x1.val * inv_x2 };
    if (x1.idx == forward_active_idx && x2.idx == forward_active_idx) {
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
        y.idx = 0;
    }
    return y;
}

auto gadfit::pow(const AdVar& x1, const AdVar& x2) -> AdVar
{
    AdVar y { std::pow(x1.val, x2.val) };
    if (x1.idx == forward_active_idx && x2.idx == forward_active_idx) {
        const double& log_value { std::log(x1.val) };
        y.d = x1.d * x2.val * std::pow(x1.val, x2.val - 1)
              + x2.d * log_value * y.val;
        const double& inv_x1 { 1.0 / x1.val };
        // clang-format off
        y.dd = y.d * y.d / y.val
               + y.val * (x2.dd * log_value
                          + (2 * x1.d * x2.d + x2.val
                             * (x1.dd - x1.d * x1.d * inv_x1)) * inv_x1);
        // clang-format on
        y.idx = forward_active_idx;
    } else if (x1.idx == forward_active_idx) {
        y.d = x1.d * x2 * std::pow(x1.val, x2 - 1);
        const double& inv_x1 { 1.0 / x1.val };
        y.dd = y.d * y.d / y.val
               + y.val * x2.val * (x1.dd - x1.d * x1.d * inv_x1) * inv_x1;
        y.idx = forward_active_idx;
    } else if (x2.idx == forward_active_idx) {
        const double& log_value { std::log(x1.val) };
        y.d = y.val * x2.d * log_value;
        y.dd = y.d * y.d / y.val + y.val * x2.dd * log_value;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::log(const AdVar& x) -> AdVar
{
    AdVar y { std::log(x.val) };
    if (x.idx == forward_active_idx) {
        const double& inv_x { 1.0 / x.val };
        y.d = x.d * inv_x;
        y.dd = (x.dd - x.d * y.d) * inv_x;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::exp(const AdVar& x) -> AdVar
{
    AdVar y { std::exp(x.val) };
    if (x.idx == forward_active_idx) {
        y.d = x.d * y.val;
        y.dd = x.dd * y.val + x.d * y.d;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::sqrt(const AdVar& x) -> AdVar
{
    AdVar y { std::sqrt(x.val) };
    if (x.idx == forward_active_idx) {
        const double& inv_y { 1.0 / y.val };
        y.d = x.d / 2 * inv_y;
        y.dd = (x.dd * inv_y - y.d * x.d / x.val) / 2;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::abs(const AdVar& x) -> AdVar
{
    AdVar y { std::abs(x.val) };
    if (x.idx == forward_active_idx) {
        y.d = x.d * (std::signbit(x.val) ? -1 : 1);
        y.dd = x.dd * (std::signbit(x.val) ? -1 : 1);
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::sin(const AdVar& x) -> AdVar
{
    AdVar y { std::sin(x.val) };
    if (x.idx == forward_active_idx) {
        const double& cos_value { std::cos(x.val) };
        y.d = x.d * cos_value;
        y.dd = x.dd * cos_value - x.d * x.d * y.val;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::cos(const AdVar& x) -> AdVar
{
    AdVar y { std::cos(x.val) };
    if (x.idx == forward_active_idx) {
        const double& sin_value { -std::sin(x.val) };
        y.d = x.d * sin_value;
        y.dd = x.dd * sin_value - x.d * x.d * y.val;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::tan(const AdVar& x) -> AdVar
{
    AdVar y { std::tan(x.val) };
    if (x.idx == forward_active_idx) {
        const double& inv_cos_value { 1.0 / std::cos(x.val) };
        const double& inv_cos_value2 { inv_cos_value * inv_cos_value };
        y.d = x.d * inv_cos_value2;
        y.dd = x.dd * inv_cos_value2 + 2 * x.d * y.val * y.d;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::asin(const AdVar& x) -> AdVar
{
    AdVar y { std::asin(x.val) };
    if (x.idx == forward_active_idx) {
        const double& tmp { 1.0 / std::sqrt(1 - x.val * x.val) };
        y.d = x.d * tmp;
        y.dd = tmp * (x.dd + x.val * y.d * y.d);
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::acos(const AdVar& x) -> AdVar
{
    AdVar y { std::acos(x.val) };
    if (x.idx == forward_active_idx) {
        const double& tmp { -1.0 / std::sqrt(1 - x.val * x.val) };
        y.d = x.d * tmp;
        y.dd = tmp * (x.dd + x.val * y.d * y.d);
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::atan(const AdVar& x) -> AdVar
{
    AdVar y { std::atan(x.val) };
    if (x.idx == forward_active_idx) {
        const double& tmp { 1.0 / (1 + x.val * x.val) };
        y.d = x.d * tmp;
        y.dd = x.dd * tmp - 2 * x.val * y.d * y.d;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::sinh(const AdVar& x) -> AdVar
{
    AdVar y { std::sinh(x.val) };
    if (x.idx == forward_active_idx) {
        const double& cosh_value { std::cosh(x.val) };
        y.d = x.d * cosh_value;
        y.dd = x.dd * cosh_value + x.d * x.d * y.val;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::cosh(const AdVar& x) -> AdVar
{
    AdVar y { std::cosh(x.val) };
    if (x.idx == forward_active_idx) {
        const double& sinh_value { std::sinh(x.val) };
        y.d = x.d * sinh_value;
        y.dd = x.dd * sinh_value + x.d * x.d * y.val;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::tanh(const AdVar& x) -> AdVar
{
    AdVar y { std::tanh(x.val) };
    if (x.idx == forward_active_idx) {
        const double& inv_cosh_value { 1.0 / std::cosh(x.val) };
        const double& inv_cosh_value2 { inv_cosh_value * inv_cosh_value };
        y.d = x.d * inv_cosh_value2;
        y.dd = x.dd * inv_cosh_value2 - 2 * x.d * y.val * y.d;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::asinh(const AdVar& x) -> AdVar
{
    AdVar y { std::asinh(x.val) };
    if (x.idx == forward_active_idx) {
        const double& tmp { 1.0 / std::sqrt(x.val * x.val + 1) };
        y.d = x.d * tmp;
        y.dd = (x.dd - x.val * y.d * y.d) * tmp;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::acosh(const AdVar& x) -> AdVar
{
    AdVar y { std::acosh(x.val) };
    if (x.idx == forward_active_idx) {
        const double& tmp { 1.0 / std::sqrt(x.val * x.val - 1) };
        y.d = x.d * tmp;
        y.dd = (x.dd - x.val * y.d * y.d) * tmp;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::atanh(const AdVar& x) -> AdVar
{
    AdVar y { std::atanh(x.val) };
    if (x.idx == forward_active_idx) {
        const double& tmp { 1.0 / (1 - x.val * x.val) };
        y.d = x.d * tmp;
        y.dd = (x.dd + 2 * x.val * x.d * y.d) * tmp;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto gadfit::erf(const AdVar& x) -> AdVar
{
    AdVar y { std::erf(x.val) };
    if (x.idx == forward_active_idx) {
        const double& tmp {
            2 * std::numbers::inv_sqrtpi_v<double> * std::exp(-x.val * x.val)
        };
        y.d = x.d * tmp;
        y.dd = (x.dd - 2 * x.d * x.d * x.val) * tmp;
        y.idx = forward_active_idx;
    } else {
        y.d = 0.0;
        y.dd = 0.0;
        y.idx = 0;
    }
    return y;
}

auto operator<<(std::ostream& out, const AdVar& x) -> std::ostream&
{
    constexpr int precision { 16 };
    out << std::setprecision(precision) << "val=" << x.val << " d=" << x.d
        << " dd=" << x.dd << " idx=" << x.idx;
    return out;
}

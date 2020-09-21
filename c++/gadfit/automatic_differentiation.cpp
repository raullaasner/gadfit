// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "automatic_differentiation.h"

#include <cmath>
#include <iomanip>
#include <iostream>

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
    AdVar y { x1.val / x2.val };
    if (x1.idx == forward_active_idx && x2.idx == forward_active_idx) {
        y.d = (x1.d - x2.d * y.val) / x2.val;
        y.dd = (x1.dd - x2.dd * y.val - 2 * y.d * x2.d) / x2.val;
        y.idx = forward_active_idx;
    } else if (x1.idx == forward_active_idx) {
        y.d = x1.d / x2.val;
        y.dd = x1.dd / x2.val;
        y.idx = forward_active_idx;
    } else if (x2.idx == forward_active_idx) {
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

// Utilities
auto operator<<(std::ostream& out, const AdVar& x) -> std::ostream&
{
    constexpr int precision { 16 };
    out << std::setprecision(precision) << "val=" << x.val << " d=" << x.d
        << " dd=" << x.dd << " idx=" << x.idx;
    return out;
}

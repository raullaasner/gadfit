// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "automatic_differentiation.h"

#include <iostream>
#include <vector>

using gadfit::AdVar;

// Work arrays for the reverse mode
namespace reverse_work {

std::vector<double> forwards; // Forward values
std::vector<double> adjoints;
std::vector<double> constants;
std::vector<int> trace;

} // namespace reverse_work

auto gadfit::operator>(const AdVar& x, const AdVar& y) -> bool
{
    return x.val > y.val;
}

auto gadfit::operator<(const AdVar& x, const AdVar& y) -> bool
{
    return x.val < y.val;
}

// Utilities
auto operator<<(std::ostream& out, const AdVar& x) -> std::ostream&
{
    out << "val=" << x.val << " d=" << x.d << " dd=" << x.dd
        << " idx=" << x.idx;
    return out;
}

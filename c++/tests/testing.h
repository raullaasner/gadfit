// Fixtures and helper functions for the tests

#pragma once

// Test the accuracy of an AD variable
#define TEST_AD(x, ref_val, ref_d, ref_dd, ref_idx)     \
    REQUIRE( x.val == approx(ref_val) );                \
    REQUIRE( x.d   == approx(ref_d)   );                \
    REQUIRE( x.dd  == approx(ref_dd)  );                \
    REQUIRE( x.idx == ref_idx         );

#include <automatic_differentiation.h>
#include <exceptions.h>
#include <catch2/catch.hpp>
#include <array>

constexpr std::array fix_d {
    6.1360420701563498,  2.9606444748278875,  9.9253736972586246,
    0.5356792380861322,  -1.4727205961479033, 4.2512661200877879,
    -2.9410316453781444, 3.4797551257539538,  2.8312317178378699,
    -1.4900798993157309, -9.7526376845644123, -8.8824179995985126,
    6.7638244484618752,  -7.1130268493509963, -4.5835128246417494,
    -8.9059115759599745, 32.898784649467867,  218.75264606693996,
    7.5767671483267520,  9.7405995203640394
};

// Tolerance in units of epsilon for all floating point comparisons
constexpr double defeault_tolerance_factor { 1e2 };

// Small wrapper around the Catch2 Approx class. The optional argument
// can be used to increase the tolerance for the cases where different
// configurations lead to slightly different results, for instance MKL
// vs the fallback linear algebra routines.
template <typename T>
auto approx(const T x,
            const double tolerance_factor = defeault_tolerance_factor) -> Approx
{
    Approx app { x };
    return app.epsilon(tolerance_factor * std::numeric_limits<T>::epsilon())
      .margin(1e-300);
}

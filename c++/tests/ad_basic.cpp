#include "fixtures.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <gadfit/automatic_differentiation.h>
#include <sstream>

TEST_CASE("Assignments")
{
    gadfit::AdVar a { fix_d[0] };
    CHECK_THAT(a.val, Catch::Matchers::WithinRel(fix_d[0], 1e-14));
    a = static_cast<int>(fix_d[0]);
    CHECK(a.val == static_cast<int>(fix_d[0]));
    a = fix_d[0];
    const int i { static_cast<int>(a.val) };
    CHECK(i == static_cast<int>(fix_d[0]));
    const double d { a.val };
    CHECK_THAT(d, Catch::Matchers::WithinRel(fix_d[0], 1e-14));
}

TEST_CASE("Comparisons")
{
    const gadfit::AdVar a { fix_d[0] };
    const gadfit::AdVar b { fix_d[1] };
    CHECK((a > b));
    CHECK(a > fix_d[5]);
    CHECK(a > static_cast<int>(fix_d[5]));
    CHECK((fix_d[5] < a));
    CHECK((static_cast<int>(fix_d[5]) < a));
    CHECK((b < a));
    CHECK(b < fix_d[5]);
    CHECK(b < static_cast<int>(fix_d[5]));
    CHECK((fix_d[5] > b));
    CHECK((static_cast<int>(fix_d[5]) > b));
}

TEST_CASE("Output")
{
    const gadfit::AdVar a {
        fix_d[0], fix_d[1], fix_d[2], static_cast<int>(fix_d[0])
    };
    std::stringstream out;
    out << a;
    CHECK(out.str()
          == "val=6.13604207015635 d=2.960644474827888 "
             "dd=9.925373697258625 idx=6");
}

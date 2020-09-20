#include "testing.h"

#include <sstream>

using gadfit::AdVar;

TEST_CASE( "Assignments" )
{
    AdVar a { fix_d[0] };
    REQUIRE( a.val == approx(fix_d[0]) );
    a = fix_f[0];
    REQUIRE( a.val == approx(fix_f[0]) );
    a = fix_i[0];
    REQUIRE( a.val == fix_i[0]);
    a = fix_d[0];
    const int i { a };
    REQUIRE( i == fix_i[0] );
    const float f { a };
    REQUIRE( f == approx(fix_f[0]) );
    const double d { a };
    REQUIRE( d == approx(fix_d[0]) );
}

TEST_CASE( "Comparisons" )
{
    const AdVar a { fix_d[0] };
    const AdVar b { fix_d[1] };
    REQUIRE( a > b );
    REQUIRE( a > fix_f[5] );
    REQUIRE( a > fix_d[5] );
    REQUIRE( a > fix_i[5] );
    REQUIRE( fix_f[5] < a );
    REQUIRE( fix_d[5] < a );
    REQUIRE( fix_i[5] < a );
    REQUIRE( b < a );
    REQUIRE( b < fix_f[5] );
    REQUIRE( b < fix_d[5] );
    REQUIRE( b < fix_i[5] );
    REQUIRE( fix_f[5] > b );
    REQUIRE( fix_d[5] > b );
    REQUIRE( fix_i[5] > b );
}

TEST_CASE( "Output" )
{
    const AdVar a { fix_d[0], fix_d[1], fix_d[2], fix_i[0] };
    std::stringstream out;
    out << a;
    REQUIRE( out.str() == "val=6.13604 d=2.96064 dd=9.92537 idx=6" );
}

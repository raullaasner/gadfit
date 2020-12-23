#include "testing.h"

using gadfit::AdVar;

TEST_CASE( "Basic arithmetic (add, subtract, multiply, divide)" )
{
    constexpr double a_ref { 0.3757755919598275 };
    constexpr double b_ref { 11.28454180719661 };
    constexpr double c_ref { -4.368251216348313 };
    constexpr double val_ref { 67.7091669953527 };
    constexpr double par_d { fix_d[0] };
    constexpr int par_i { static_cast<int>(fix_d[8]) };
    const auto expression {
        [=](const AdVar& a, const AdVar& b, const AdVar& c) {
            return par_d * (a / par_d / par_i + par_d) + b * (par_d - c)
              - (c - par_d) / (par_d + a - b + par_d + par_i) + par_d / (-b) *
              par_d;
        }
    };
    AdVar a { fix_d[4], 0.0, 0.0, gadfit::passive_idx };
    AdVar b { fix_d[5], 0.0, 0.0, gadfit::passive_idx };
    AdVar c { fix_d[6], 0.0, 0.0, gadfit::passive_idx };
    gadfit::initializeADReverse();

    SECTION( "Active: none" ) {
        const AdVar result { expression(a, b, c) };
        TEST_AD( result, val_ref, 0.0, 0.0, gadfit::passive_idx );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: a" ) {
        a.idx = 0; gadfit::addADSeed(a);
        const AdVar result { expression(a, b, c) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 12 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(a_ref) );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: b" ) {
        b.idx = 0; gadfit::addADSeed(b);
        const AdVar result { expression(a, b, c) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 11 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(b_ref) );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: c" ) {
        c.idx = 0; gadfit::addADSeed(c);
        const AdVar result { expression(a, b, c) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 7 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(c_ref) );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: a, b" ) {
        a.idx = 0; gadfit::addADSeed(a);
        b.idx = 1; gadfit::addADSeed(b);
        const AdVar result { expression(a, b, c) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 17 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(a_ref) );
        REQUIRE( gadfit::reverse::adjoints[1] == approx(b_ref) );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: a, c (with resetting)" ) {
        a.idx = 0; gadfit::addADSeed(a);
        c.idx = 1; gadfit::addADSeed(c);
        expression(a, b, c);
        gadfit::returnSweep();
        gadfit::adResetSweep();
        a.idx = 0; gadfit::addADSeed(a);
        c.idx = 1; gadfit::addADSeed(c);
        const AdVar result { expression(a, b, c) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 16 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(a_ref) );
        REQUIRE( gadfit::reverse::adjoints[1] == approx(c_ref) );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: b, c" ) {
        b.idx = 0; gadfit::addADSeed(b);
        c.idx = 1; gadfit::addADSeed(c);
        const AdVar result { expression(a, b, c) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 14 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(b_ref) );
        REQUIRE( gadfit::reverse::adjoints[1] == approx(c_ref) );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: a, b, c" ) {
        a.idx = 0; gadfit::addADSeed(a);
        b.idx = 1; gadfit::addADSeed(b);
        c.idx = 2; gadfit::addADSeed(c);
        const AdVar result { expression(a, b, c) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 20 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(a_ref) );
        REQUIRE( gadfit::reverse::adjoints[1] == approx(b_ref) );
        REQUIRE( gadfit::reverse::adjoints[2] == approx(c_ref) );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: a (error handling)" ) {
        a.idx = 0; gadfit::addADSeed(a);
        const AdVar result { expression(a, b, c) };
        // Artificially introduce a wrong operation code
        gadfit::reverse::trace[2] = -1;
        try {
            gadfit::returnSweep();
            REQUIRE( 1 == 0 );
        } catch (const gadfit::UnknownOperationException& e) {
            e.what();
            REQUIRE( 0 == 0 );
        }
        gadfit::freeAdReverse();
    }
}

TEST_CASE( "Exponentiation, logarithm, other" )
{
    constexpr double a_ref { -4.467047498107922 };
    constexpr double b_ref { 387.314505060867 };
    constexpr double val_ref { 402.2477537977381 };
    constexpr double par_d { fix_d[0] };
    constexpr int par_i { static_cast<int>(fix_d[8]) };
    const auto expression {
        [=](const AdVar& a, const AdVar& b) {
            return
              pow(b, a)
              + pow(b, par_d)/pow(b, par_i)
              - pow(par_d, +a)/pow(par_i, a)*abs(a)*abs(b)
              + exp(sqrt(abs(a)) + log(b))/sqrt(log(-b/a)*par_d);
        }
    };
    AdVar a { fix_d[4], 0.0, 0.0, gadfit::passive_idx };
    AdVar b { fix_d[5], 0.0, 0.0, gadfit::passive_idx };
    gadfit::initializeADReverse();

    SECTION( "Active: none" ) {
        const AdVar result { expression(a, b) };
        TEST_AD( result, val_ref, 0.0, 0.0, gadfit::passive_idx );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: a" ) {
        a.idx = 0; gadfit::addADSeed(a);
        const AdVar result { expression(a, b) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 19 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(a_ref) );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: b" ) {
        b.idx = 0; gadfit::addADSeed(b);
        const AdVar result { expression(a, b) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 18 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(b_ref) );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: a, b" ) {
        a.idx = 0; gadfit::addADSeed(a);
        b.idx = 1; gadfit::addADSeed(b);
        const AdVar result { expression(a, b) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 26 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(a_ref) );
        REQUIRE( gadfit::reverse::adjoints[1] == approx(b_ref) );
        gadfit::freeAdReverse();
    }
}

TEST_CASE( "Trigonometric" )
{
    constexpr double val_ref { -0.5540770421819348 };
    constexpr double a_ref { -1.549501027521998 };
    constexpr double b_ref { 0.4556944655440529 };
    const auto expression {
        [=](const AdVar& a, const AdVar& b) {
            return
              sin(a*b)*cos(a)/cos(b)
              + tan(cos(a))/atan(b*asin(1/a)/acos(a/b))
              + sinh(a/b)*pow(cosh(b/a), tanh(b/a))
              + asinh(a/b)*pow(acosh(abs(b/a)),atanh(abs(a/b)));
        }
    };
    AdVar a { fix_d[4], 0.0, 0.0, gadfit::passive_idx };
    AdVar b { fix_d[5], 0.0, 0.0, gadfit::passive_idx };
    gadfit::initializeADReverse();

    SECTION( "Active: none" ) {
        const AdVar result { expression(a, b) };
        TEST_AD( result, val_ref, 0.0, 0.0, gadfit::passive_idx );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: a" ) {
        a.idx = 0; gadfit::addADSeed(a);
        const AdVar result { expression(a, b) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 36 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(a_ref) );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: b" ) {
        b.idx = 0; gadfit::addADSeed(b);
        const AdVar result { expression(a, b) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 32 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(b_ref) );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: a, b" ) {
        a.idx = 0; gadfit::addADSeed(a);
        b.idx = 1; gadfit::addADSeed(b);
        const AdVar result { expression(a, b) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 38 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(a_ref) );
        REQUIRE( gadfit::reverse::adjoints[1] == approx(b_ref) );
        gadfit::freeAdReverse();
    }
}

TEST_CASE( "Special" )
{
    constexpr double val_ref { 0.5512884666654083 };
    constexpr double a_ref { 0.8469022413858851 };
    const auto expression { [=](const AdVar& a) { return erf(a); } };
    AdVar a { fix_d[3], 0.0, 0.0, gadfit::passive_idx };
    gadfit::initializeADReverse();

    SECTION( "Active: none" ) {
        const AdVar result { expression(a) };
        TEST_AD( result, val_ref, 0.0, 0.0, gadfit::passive_idx );
        gadfit::freeAdReverse();
    }
    SECTION( "Active: a" ) {
        a.idx = 0; gadfit::addADSeed(a);
        const AdVar result { expression(a) };
        gadfit::returnSweep();
        TEST_AD( result, val_ref, 0.0, 0.0, 1 );
        REQUIRE( gadfit::reverse::adjoints[0] == approx(a_ref) );
        gadfit::freeAdReverse();
    }
}

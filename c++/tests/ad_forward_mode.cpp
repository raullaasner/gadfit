#include "testing.h"

using gadfit::AdVar;

TEST_CASE( "Basic arithmetic (add, subtract, multiply, divide)" )
{
    constexpr double par_d { fix_d[0] };
    constexpr int par_i { fix_i[8] };
    const auto expression {
        [=](const AdVar& a, const AdVar& b, const AdVar& c) {
            return par_d * (a / par_d / par_i + par_d) + b * (par_d - c)
              - (c - par_d) / (par_d + a - b + par_d + par_i) + par_d / (-b) *
              par_d;
        }
    };
    AdVar a { fix_d[4], 0.0, 0.0, 0 };
    AdVar b { fix_d[5], 0.0, 0.0, 0 };
    AdVar c { fix_d[6], 0.0, 0.0, 0 };

    SECTION( "Active: none" ) {
        const AdVar result { expression(a, b, c) };
        TEST_AD( result, 67.7091669953527, 0.0, 0.0, 0 );
    }
    SECTION( "Active: a" ) {
        a.d = a.dd = 1.0; a.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a, b, c) };
        TEST_AD( result,
                 67.7091669953527, 0.3757755919598275, 0.4048404006248003, -1 );
    }
    SECTION( "Active: b" ) {
        b.d = b.dd = 1.0; b.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a, b, c) };
        TEST_AD( result,
                 67.7091669953527, 11.28454180719661, 10.33354861677357, -1 );
    }
    SECTION( "Active: c" ) {
        c.d = c.dd = 1.0; c.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a, b, c) };
        TEST_AD( result,
                 67.7091669953527, -4.368251216348313, -4.368251216348313, -1 );
    }
    SECTION( "Active: a, b" ) {
        a.d = a.dd = 1.0; a.idx = gadfit::forward_active_idx;
        b.d = b.dd = 1.0; b.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a, b, c) };
        TEST_AD( result,
                 67.7091669953527, 11.66031739915644, 10.68025940006843, -1 );
    }
    SECTION( "Active: a, c" ) {
        a.d = a.dd = 1.0; a.idx = gadfit::forward_active_idx;
        c.d = c.dd = 1.0; c.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a, b, c) };
        TEST_AD( result,
                 67.7091669953527, -3.992475624388486, -3.936039790229344, -1 );
    }
    SECTION( "Active: b, c" ) {
        b.d = b.dd = 1.0; b.idx = gadfit::forward_active_idx;
        c.d = c.dd = 1.0; c.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a, b, c) };
        TEST_AD( result,
                 67.7091669953527, 6.916290590848295, 3.937926374931089, -1 );
    }
    SECTION( "Active: a, b, c" ) {
        a.d = a.dd = 1.0; a.idx = gadfit::forward_active_idx;
        b.d = b.dd = 1.0; b.idx = gadfit::forward_active_idx;
        c.d = c.dd = 1.0; c.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a, b, c) };
        TEST_AD( result,
                 67.7091669953527, 7.292066182808123, 4.312008183720113, -1 );
    }
}

TEST_CASE( "Exponentiation, logarithm, other" )
{
    constexpr double par_d { fix_d[0] };
    constexpr int par_i { fix_i[8] };
    const auto expression {
        [=](const AdVar& a, const AdVar& b) {
            return
              pow(b, a)
              + pow(b, par_d)/pow(b, par_i)
              - pow(par_d, +a)/pow(par_i, a)*abs(a)*abs(b)
              + exp(sqrt(abs(a)) + log(b))/sqrt(log(-b/a)*par_d);
        }
    };
    AdVar a { fix_d[4], 0.0, 0.0, 0 };
    AdVar b { fix_d[5], 0.0, 0.0, 0 };
    SECTION( "Active: none" ) {
        const AdVar result { expression(a, b) };
        TEST_AD( result, 402.2477537977381, 0.0, 0.0, 0 );
    }
    SECTION( "Active: a" ) {
        a.d = a.dd = 1.0; a.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a, b) };
        TEST_AD( result,
                 402.2477537977381, -4.467047498107922, -1.74515271385656, -1 );
    }
    SECTION( "Active: b" ) {
        b.d = b.dd = 1.0; b.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a, b) };
        TEST_AD( result,
                 402.2477537977381, 387.314505060867, 672.8348977983287, -1 );
    }
    SECTION( "Active: a, b" ) {
        a.d = a.dd = 1.0; a.idx = gadfit::forward_active_idx;
        b.d = b.dd = 1.0; b.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a, b) };
        TEST_AD( result,
                 402.2477537977381, 382.8474575627591, 670.5530084135362, -1 );
    }
}

TEST_CASE( "Trigonometric" )
{
    const auto expression {
        [=](const AdVar& a, const AdVar& b) {
            return
              sin(a*b)*cos(a)/cos(b)
              + tan(cos(a))/atan(b*asin(1/a)/acos(a/b))
              + sinh(a/b)*pow(cosh(b/a), tanh(b/a))
              + asinh(a/b)*pow(acosh(abs(b/a)),atanh(abs(a/b)));
        }
    };
    AdVar a { fix_d[4], 0.0, 0.0, 0 };
    AdVar b { fix_d[5], 0.0, 0.0, 0 };
    SECTION( "Active: none" ) {
        const AdVar result { expression(a, b) };
        TEST_AD( result, -0.5540770421819348, 0.0, 0.0, 0 );
    }
    SECTION( "Active: a" ) {
        a.d = a.dd = 1.0; a.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a, b) };
        TEST_AD( result,
                 -0.5540770421819348, -1.549501027521998, -19.82580971358727,
                 -1 );
    }
    SECTION( "Active: b" ) {
        b.d = b.dd = 1.0; b.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a, b) };
        TEST_AD( result,
                 -0.5540770421819348, 0.4556944655440529, 1.648782695748266,
                 -1 );
    }
    SECTION( "Active: a, b" ) {
        a.d = a.dd = 1.0; a.idx = gadfit::forward_active_idx;
        b.d = b.dd = 1.0; b.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a, b) };
        TEST_AD( result,
                 -0.5540770421819348, -1.093806561977945, -16.07874055136707,
                 -1 );
    }
}

TEST_CASE( "Special" )
{
    const auto expression { [=](const AdVar& a) { return erf(a); } };
    AdVar a { fix_d[3], 0.0, 0.0, 0 };
    SECTION( "Active: none" ) {
        const AdVar result { expression(a) };
        TEST_AD( result, 0.5512884666654083, 0.0, 0.0, 0 );
    }
    SECTION( "Active: a" ) {
        a.d = a.dd = 1.0; a.idx = gadfit::forward_active_idx;
        const AdVar result { expression(a) };
        TEST_AD( result,
                 0.5512884666654083, 0.8469022413858851, -0.06043365341217193,
                 -1 );
    }
}

#include "testing.h"

#include "numerical_integration.h"
#include "numerical_integration_data.h"

#include <lm_solver.h>
#include <spdlog/spdlog.h>

// Except for the fit function, the fitting procedures use the same
// initialization.
static auto setSolverState(gadfit::LMsolver& solver) -> void {
    solver.addDataset(x_data_single, y_data_single);
    solver.setPar(0, 10.0, true);
    solver.setPar(1, 1.0, true);
    solver.settings.iteration_limit = 4;
    solver.settings.acceleration_threshold = 0.9;
}

TEST_CASE( "Single integral" )
{
    spdlog::set_level(spdlog::level::off);
    SECTION( "No bounds" ) {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pow(x, pars[0]) * exp(-pars[1] * x * x);
        } };
        const auto f {
            [&](const std::vector<gadfit::AdVar>& pars, const double x) {
                return fix_d[1] * integrate(integrand, pars, 0.0, x, 1e-12);
            }
        };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverState(solver);
        solver.fit(10.0);
        REQUIRE( solver.chi2() == approx(4994.8010481036144) );
        REQUIRE( solver.getParValue(0) == approx(9.3456933979838333) );
        REQUIRE( solver.getParValue(1) == approx(1.0863418220603043) );
    }
    SECTION( "Lower bound" ) {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[2] * pow(x, pars[0]) * exp(-pars[1] * x * x);
        } };
        const auto f {
            [&](const std::vector<gadfit::AdVar>& pars, const double x) {
                std::vector<gadfit::AdVar> pars2(3);
                pars2[0] = pars[0];
                pars2[1] = pars[1];
                pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
                return -fix_d[1]
                  * integrate(integrand, pars2, pars[0]/fix_d[0], 0.0, 1e-12);
            }
        };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverState(solver);
        solver.setPar(1, 1.0, false);
        solver.fit(10.0);
        REQUIRE( solver.chi2() == approx(3359.4027609550717) );
        REQUIRE( solver.getParValue(0) == approx(9.6386865163774367) );
        REQUIRE( solver.getParValue(1) == approx(1.0) );
        solver.setPar(1, 1.0, true);
        solver.fit(10.0);
        REQUIRE( solver.chi2() == approx(3359.3605256978799) );
        REQUIRE( solver.getParValue(0) == approx(9.6383735850836487) );
        REQUIRE( solver.getParValue(1) == approx(1.000164288516687) );
    }
    SECTION( "Lower bound (no parameters in integrand)" ) {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[2] * pow(x, fix_d[2]) * exp(-x * x);
        } };
        const auto f {
            [&](const std::vector<gadfit::AdVar>& pars, const double x) {
                std::vector<gadfit::AdVar> pars2(3);
                pars2[0] = pars[0];
                pars2[1] = pars[1];
                pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
                return -fix_d[1]
                  * integrate(integrand, pars2, pars[0]/fix_d[0], 0.0, 1e-12);
            }
        };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverState(solver);
        solver.setPar(1, 1.0, false);
        solver.fit(10.0);
        REQUIRE( solver.chi2() == approx(3359.374808601714) );
        REQUIRE( solver.getParValue(0) == approx(9.513801290676248) );
        REQUIRE( solver.getParValue(1) == approx(1.0) );
    }
    SECTION( "Upper bound" ) {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[2] * pow(x, pars[0]) * exp(-pars[1] * x * x);
        } };
        const auto f {
            [&](const std::vector<gadfit::AdVar>& pars, const double x) {
                std::vector<gadfit::AdVar> pars2(3);
                pars2[0] = pars[0];
                pars2[1] = pars[1];
                pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
                return fix_d[1]
                  * integrate(integrand, pars2, 0.0, pars[0]/fix_d[0], 1e-12);
            }
        };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverState(solver);
        solver.setPar(1, 1.0, false);
        solver.fit(10.0);
        REQUIRE( solver.chi2() == approx(3359.4027609550717) );
        REQUIRE( solver.getParValue(0) == approx(9.6386865163774367) );
        REQUIRE( solver.getParValue(1) == approx(1.0) );
        solver.setPar(1, 1.0, true);
        solver.fit(10.0);
        REQUIRE( solver.chi2() == approx(3359.3605256978799) );
        REQUIRE( solver.getParValue(0) == approx(9.6383735850836487) );
        REQUIRE( solver.getParValue(1) == approx(1.000164288516687) );
    }
    SECTION( "Upper bound (no parameters in integrand)" ) {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[0] * pow(x, fix_d[2]) * exp(-x * x);
        } };
        const auto f {
            [&](const std::vector<gadfit::AdVar>& pars, const double x) {
                std::vector<gadfit::AdVar> pars2 { gadfit::AdVar {
                  x, gadfit::passive_idx } };
                return fix_d[1]
                  * integrate(integrand, pars2, 0.0, pars[0]/fix_d[0], 1e-12);
            }
        };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverState(solver);
        solver.setPar(1, 1.0, false);
        solver.fit(10.0);
        REQUIRE( solver.chi2() == approx(3359.374808601714) );
        REQUIRE( solver.getParValue(0) == approx(9.513801290676248) );
        REQUIRE( solver.getParValue(1) == approx(1.0) );
    }
    SECTION( "Both bounds" ) {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[2] * pow(x, pars[0]) * exp(-pars[1] * x * x);
        } };
        const auto f {
            [&](const std::vector<gadfit::AdVar>& pars, const double x) {
                std::vector<gadfit::AdVar> pars2(3);
                pars2[0] = pars[0];
                pars2[1] = pars[1];
                pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
                return -fix_d[1] * integrate(
                  integrand, pars2, pars[0]/fix_d[0], pars[1], 1e-12);
            }
        };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverState(solver);
        solver.setPar(1, 1.0, false);
        solver.fit(10.0);
        REQUIRE( solver.chi2() == approx(3359.3921367899) );
        REQUIRE( solver.getParValue(0) == approx(9.664371097350363) );
        REQUIRE( solver.getParValue(1) == approx(1.0) );
        solver.setPar(1, 1.0, true);
        solver.fit(10.0);
        REQUIRE( solver.chi2() == approx(3359.360525697832) );
        REQUIRE( solver.getParValue(0) == approx(9.664108472227593) );
        REQUIRE( solver.getParValue(1) == approx(1.000124158231295) );
    }
    SECTION( "Both bounds (lower inactive)" ) {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[2] * pow(x, pars[0]) * exp(-pars[1] * x * x);
        } };
        const auto f {
            [&](const std::vector<gadfit::AdVar>& pars, const double x) {
                std::vector<gadfit::AdVar> pars2(3);
                pars2[0] = pars[0];
                pars2[1] = pars[1];
                pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
                return -fix_d[1] * integrate(
                  integrand, pars2, pars[1], pars[0]/fix_d[0], 1e-12);
            }
        };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverState(solver);
        solver.setPar(1, 1.0, false);
        solver.fit(10.0);
        REQUIRE( solver.chi2() == approx(96283.63738642583) );
        REQUIRE( solver.getParValue(0) == approx(4.023936467213234) );
        REQUIRE( solver.getParValue(1) == approx(1.0) );
    }
    SECTION( "Both bounds (no parameters in integrand)" ) {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[0] * pow(x, fix_d[2]) * exp(-x * x);
        } };
        const auto f {
            [&](const std::vector<gadfit::AdVar>& pars, const double x) {
                std::vector<gadfit::AdVar> pars2 { gadfit::AdVar {
                  x, gadfit::passive_idx } };
                return -fix_d[1] * integrate(
                  integrand, pars2, pars[0]/fix_d[0], pars[1], 1e-12);
            }
        };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverState(solver);
        solver.fit(10.0);
        REQUIRE( solver.chi2() == approx(3359.360587615626) );
        REQUIRE( solver.getParValue(0) == approx(9.834021674777725) );
        REQUIRE( solver.getParValue(1) == approx(1.301193106585964) );
    }
}

static auto setSolverStateNested(gadfit::LMsolver& solver) -> void {
    gadfit::initIntegration(1000, 2);
    solver.addDataset(x_data_double, y_data_double, weights_double);
    solver.setPar(0, 7.0, true);
    solver.settings.iteration_limit = 2;
    solver.settings.acceleration_threshold = 0.9;
}

TEST_CASE( "Double integral (nested)" )
{
    spdlog::set_level(spdlog::level::off);
    const auto integrand_inner { [](const std::vector<gadfit::AdVar>& pars,
                                    const gadfit::AdVar& x) {
        return log((exp(x) - 0.9) * pars[0] + 1.0) / x;
    } };
    SECTION( "Active bounds: y1 y2 x1 x2" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       pars[3],
                                       pars[4] * pars[2] / pars[1],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(integrand_outer,
                             pars2,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(0.21322347907024705, 1e3) );
        REQUIRE( solver.getParValue(0) == approx(15.267560288033625) );
        REQUIRE( solver.getParValue(1) == approx(1.3863816932256614) );
        REQUIRE( solver.getParValue(2) == approx(0.84864756619753778) );
        REQUIRE( solver.getParValue(3) == approx(1.6742467335884419) );
        REQUIRE( solver.getParValue(4) == approx(0.18857413777419035) );
        REQUIRE( solver.getParValue(5) == approx(1.9418026197099345) );
    }
    SECTION( "Active bounds: y1 y2 x1" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       pars[3],
                                       pars[4] * fix_d[2],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(integrand_outer,
                             pars2,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(20529.862144585662) );
        REQUIRE( solver.getParValue(0) == approx(9.5451264445366082, 1e3) );
        REQUIRE( solver.getParValue(1) == approx(1.0509494096733416, 1e3) );
        REQUIRE( solver.getParValue(2) == approx(1.4070141491563901, 1e3) );
        REQUIRE( solver.getParValue(3) == approx(2.2466015973378544) );
        REQUIRE( solver.getParValue(4) == approx(0.095815584889715602, 1e4) );
        REQUIRE( solver.getParValue(5) == approx(2.1000000000000001) );
    }
    SECTION( "Active bounds: y1 y2 x2" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       fix_d[3],
                                       pars[4] * pars[2] / pars[1],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(integrand_outer,
                             pars2,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(8.5111904781092687) );
        REQUIRE( solver.getParValue(0) == approx(31.354025346686672) );
        REQUIRE( solver.getParValue(1) == approx(1.3432353695501853) );
        REQUIRE( solver.getParValue(2) == approx(0.98808053073566204) );
        REQUIRE( solver.getParValue(3) == approx(1.9151538799470511) );
        REQUIRE( solver.getParValue(4) == approx(0.63014716047505859) );
        REQUIRE( solver.getParValue(5) == approx(2.0414729418746558) );
    }
    SECTION( "Active bounds: y1 x1 x2" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       pars[3],
                                       pars[4] * pars[2] / pars[1],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(integrand_outer,
                             pars2,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(0.54533843744981803) );
        REQUIRE( solver.getParValue(0) == approx(14.519188784964555) );
        REQUIRE( solver.getParValue(1) == approx(1.4018015531760157) );
        REQUIRE( solver.getParValue(2) == approx(0.77040828428310459) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.22435256517682181) );
        REQUIRE( solver.getParValue(5) == approx(1.9119631628520173) );
    }
    SECTION( "Active bounds: y2 x1 x2" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       pars[3],
                                       pars[4] * pars[2] / pars[1],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return -integrate(integrand_outer,
                             pars2,
                             pars[3],
                             pars[4] * (pars[1] - pars[2]),
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(0.54533844279237143, 1e3) );
        REQUIRE( solver.getParValue(0) == approx(14.51918879971193) );
        REQUIRE( solver.getParValue(1) == approx(1.4018015530359746) );
        REQUIRE( solver.getParValue(2) == approx(0.77040828482222012) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.22435256564298342) );
        REQUIRE( solver.getParValue(5) == approx(1.9119631630806118) );
    }
    SECTION( "Active bounds: y1 y2" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       1/fix_d[16],
                                       pars[4],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(integrand_outer,
                             pars2,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(31829.013336620694) );
        REQUIRE( solver.getParValue(0) == approx(8.4292941631208986) );
        REQUIRE( solver.getParValue(1) == approx(1.5691886469730438) );
        REQUIRE( solver.getParValue(2) == approx(1.4762979593902223) );
        REQUIRE( solver.getParValue(3) == approx(2.3508567481628528) );
        REQUIRE( solver.getParValue(4) == approx(0.14450095129620641) );
        REQUIRE( solver.getParValue(5) == approx(2.1000000000000001) );
    }
    SECTION( "Active bounds: x1 x2" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       pars[3],
                                       pars[4] * pars[2] / pars[1],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(integrand_outer,
                             pars2,
                             pars[4] * pars[2],
                             pars[3],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(0.063818114751678021) );
        REQUIRE( solver.getParValue(0) == approx(15.543190497406581) );
        REQUIRE( solver.getParValue(1) == approx(1.3376522801060835) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.20000000000000001) );
        REQUIRE( solver.getParValue(5) == approx(2.0604239148555297) );
    }
    SECTION( "Active bounds: y1 x2" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       fix_d[3],
                                       pars[4] * pars[2] / pars[1],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(integrand_outer,
                             pars2,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(8.3103844256867898) );
        REQUIRE( solver.getParValue(0) == approx(31.057028905358738) );
        REQUIRE( solver.getParValue(1) == approx(1.3374474619525787) );
        REQUIRE( solver.getParValue(2) == approx(0.9731477156400089) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.66766007253694248) );
        REQUIRE( solver.getParValue(5) == approx(2.0424787013200016) );
    }
    SECTION( "Active bounds: y2 x1" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return -exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       pars[4],
                                       pars[3],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(integrand_outer,
                             pars2,
                             // pars[4] * (pars[1] - pars[2]),
                             pars[4],
                             pars[3],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(20530.2001603592) );
        REQUIRE( solver.getParValue(0) == approx(72.098119609180372) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(10.902185246040824) );
        REQUIRE( solver.getParValue(4) == approx(0.20000000000000001) );
        REQUIRE( solver.getParValue(5) == approx(15.562627445121468) );
    }
    SECTION( "Active bounds: y1 x1" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return -exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       pars[4] * pars[2] / pars[1],
                                       pars[3],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(integrand_outer,
                             pars2,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(0.37084591046164767) );
        REQUIRE( solver.getParValue(0) == approx(13.426187303222719) );
        REQUIRE( solver.getParValue(1) == approx(1.4078791924572198) );
        REQUIRE( solver.getParValue(2) == approx(0.75742145244657488) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.20000000000000001) );
        REQUIRE( solver.getParValue(5) == approx(1.8962901940867134) );
    }
    SECTION( "Active bounds: y2 x2" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       pars[3],
                                       pars[4] * pars[2] / pars[1],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return -integrate(integrand_outer,
                             pars2,
                             pars[3],
                             pars[4] * (pars[1] - pars[2]),
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(0.36456834401215527) );
        REQUIRE( solver.getParValue(0) == approx(13.455498136730942) );
        REQUIRE( solver.getParValue(1) == approx(1.4080606493533714) );
        REQUIRE( solver.getParValue(2) == approx(0.75703615932599089) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.20000000000000001) );
        REQUIRE( solver.getParValue(5) == approx(1.8959804133198128) );
    }
    SECTION( "Active bounds: y1" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       pars[3],
                                       pars[4],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(integrand_outer,
                             pars2,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(33404.970478265212) );
        REQUIRE( solver.getParValue(0) == approx(18.746116684579437) );
        REQUIRE( solver.getParValue(1) == approx(3.1273507059264882) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.20000000000000001) );
        REQUIRE( solver.getParValue(5) == approx(2.1000000000000001) );
    }
    SECTION( "Active bounds: y2" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       pars[3],
                                       pars[4],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return -integrate(integrand_outer,
                             pars2,
                             pars[3],
                             pars[4] * (pars[1] - pars[2]),
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(33404.970478274736) );
        REQUIRE( solver.getParValue(0) == approx(18.746116684739075) );
        REQUIRE( solver.getParValue(1) == approx(3.1273507058426082) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.20000000000000001) );
        REQUIRE( solver.getParValue(5) == approx(2.1000000000000001) );
    }
    SECTION( "Active bounds: x1" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return -exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       pars[4],
                                       pars[3],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(integrand_outer,
                             pars2,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(20529.868741915368) );
        REQUIRE( solver.getParValue(0) == approx(80.959889654657957) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.20000000000000001) );
        REQUIRE( solver.getParValue(5) == approx(15.632316246923653) );
    }
    SECTION( "Active bounds: x2" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       pars[3],
                                       pars[4],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(integrand_outer,
                             pars2,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(20529.868955027843) );
        REQUIRE( solver.getParValue(0) == approx(80.959894823689893) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.20000000000000001) );
        REQUIRE( solver.getParValue(5) == approx(15.632339735341827) );
    }
    SECTION( "No active bounds" ) {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x) * integrate(integrand_inner,
                                       pars2,
                                       pars[3],
                                       pars[4] * pars[2] / pars[1],
                                       1e-6);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(integrand_outer,
                             pars2,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(1035.6708902846135) );
        REQUIRE( solver.getParValue(0) == approx(-0.75577153013394582) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.20000000000000001) );
        REQUIRE( solver.getParValue(5) == approx(2.1000000000000001) );
    }
}

TEST_CASE( "Exceptions" )
{
    const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                              const gadfit::AdVar& x) {
        return pow(x, pars[0]) * exp(-pars[1] * x * x);
    } };
    const auto f { [&](const std::vector<gadfit::AdVar>& pars, const double x) {
        return fix_d[1] * integrate(integrand, pars, 0.0, x, 1e-12);
    } };
    gadfit::initIntegration(2);
    gadfit::LMsolver solver { f, MPI_COMM_WORLD };
    setSolverState(solver);
    try {
        solver.fit(10.0);
        REQUIRE( 1 == 0 );
    } catch (const gadfit::InsufficientIntegrationWorkspace& e) {
        e.what();
        REQUIRE( 0 == 0 );
    }
}

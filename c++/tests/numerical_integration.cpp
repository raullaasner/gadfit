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
        REQUIRE( solver.chi2() == approx(0.2132234848898489, 1e3) );
        REQUIRE( solver.getParValue(0) == approx(15.26756028671046) );
        REQUIRE( solver.getParValue(1) == approx(1.386381693033272) );
        REQUIRE( solver.getParValue(2) == approx(0.8486475665132621) );
        REQUIRE( solver.getParValue(3) == approx(1.674246733588323) );
        REQUIRE( solver.getParValue(4) == approx(0.1885741381233513) );
        REQUIRE( solver.getParValue(5) == approx(1.941802620042361) );
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
        REQUIRE( solver.chi2() == approx(20529.86214447366) );
        REQUIRE( solver.getParValue(0) == approx(9.54512644882254, 1e3) );
        REQUIRE( solver.getParValue(1) == approx(1.050949410185254, 1e4) );
        REQUIRE( solver.getParValue(2) == approx(1.407014149357748, 1e3) );
        REQUIRE( solver.getParValue(3) == approx(2.246601597751148, 1e3) );
        REQUIRE( solver.getParValue(4) == approx(0.09581558493185093, 1e4) );
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
        REQUIRE( solver.chi2() == approx(8.511190478331665) );
        REQUIRE( solver.getParValue(0) == approx(31.35402534977554) );
        REQUIRE( solver.getParValue(1) == approx(1.343235369539531) );
        REQUIRE( solver.getParValue(2) == approx(0.9880805307694033) );
        REQUIRE( solver.getParValue(3) == approx(1.915153879647961) );
        REQUIRE( solver.getParValue(4) == approx(0.6301471603862133, 1e3) );
        REQUIRE( solver.getParValue(5) == approx(2.041472941899317) );
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
        REQUIRE( solver.chi2() == approx(0.5453384421466564, 1e3) );
        REQUIRE( solver.getParValue(0) == approx(14.51918880106175) );
        REQUIRE( solver.getParValue(1) == approx(1.40180155304181) );
        REQUIRE( solver.getParValue(2) == approx(0.7704082848128452) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.2243525656468343) );
        REQUIRE( solver.getParValue(5) == approx(1.911963163069007) );
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
        REQUIRE( solver.chi2() == approx(0.5453384421466757, 1e3) );
        REQUIRE( solver.getParValue(0) == approx(14.51918880106178) );
        REQUIRE( solver.getParValue(1) == approx(1.401801553041809) );
        REQUIRE( solver.getParValue(2) == approx(0.7704082848128471) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.2243525656468352) );
        REQUIRE( solver.getParValue(5) == approx(1.911963163069006) );
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
        REQUIRE( solver.chi2() == approx(0.06381811471197225, 1e3) );
        REQUIRE( solver.getParValue(0) == approx(15.54319049642039) );
        REQUIRE( solver.getParValue(1) == approx(1.337652280082468) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.20000000000000001) );
        REQUIRE( solver.getParValue(5) == approx(2.060423914878352) );
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
        REQUIRE( solver.chi2() == approx(8.310384425769623) );
        REQUIRE( solver.getParValue(0) == approx(31.05702890747916) );
        REQUIRE( solver.getParValue(1) == approx(1.337447461962609) );
        REQUIRE( solver.getParValue(2) == approx(0.9731477156477512) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.6676600724901515) );
        REQUIRE( solver.getParValue(5) == approx(2.042478701325957) );
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
        REQUIRE( solver.chi2() == approx(0.364568344007835) );
        REQUIRE( solver.getParValue(0) == approx(13.45549813738398) );
        REQUIRE( solver.getParValue(1) == approx(1.408060649359838) );
        REQUIRE( solver.getParValue(2) == approx(0.7570361593190074) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.20000000000000001) );
        REQUIRE( solver.getParValue(5) == approx(1.895980413306825) );
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
        REQUIRE( solver.chi2() == approx(33404.97047827223) );
        REQUIRE( solver.getParValue(0) == approx(18.74611668464275) );
        REQUIRE( solver.getParValue(1) == approx(3.12735070583082) );
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
        REQUIRE( solver.chi2() == approx(33404.97047827223) );
        REQUIRE( solver.getParValue(0) == approx(18.74611668464276) );
        REQUIRE( solver.getParValue(1) == approx(3.127350705830819) );
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
        REQUIRE( solver.chi2() == approx(20529.86895502715) );
        REQUIRE( solver.getParValue(0) == approx(80.95989482456348) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.20000000000000001) );
        REQUIRE( solver.getParValue(5) == approx(15.63233973598849) );
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
                             pars[3] / pars[5],
                             1e-5) / x;
        } };
        gadfit::LMsolver solver { f_double, MPI_COMM_WORLD };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.settings.iteration_limit = 1;
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(158.6361952236741) );
        REQUIRE( solver.getParValue(0) == approx(24.355838792806) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.20000000000000001) );
        REQUIRE( solver.getParValue(5) == approx(2.1000000000000001) );
    }
}

static auto setSolverStateDirect(gadfit::LMsolver& solver) -> void {
    solver.addDataset(x_data_double, y_data_double_direct);
    solver.setPar(0, 7.0, true);
    solver.settings.iteration_limit = 2;
    solver.settings.acceleration_threshold = 0.9;
}

// Convergence criterion for the 2D (direct) integrals
constexpr double integration_tolerance { 1e-7 };

TEST_CASE( "Double integral (direct)" )
{
    spdlog::set_level(spdlog::level::off);
    const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                              const gadfit::AdVar& x,
                              const gadfit::AdVar& y) {
        const gadfit::AdVar tmp { 1 + pars[0] * pars[1] * erf(y) };
        return exp(-y) * log((exp(x) - 0.9) * tmp + 1.0) / x;
    } };
    const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                       const double x) {
        return integrate(integrand,
                         pars,
                         pars[4] * (pars[1] - pars[2]),
                         pars[3] * pars[6],
                         pars[4] * pars[6],
                         pars[5] / pars[1],
                         integration_tolerance) / x;
    } };
    SECTION( "Active bounds: y1 y2 x1 x2" ) {
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(0, 7.0, false);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(1.654408836391586e-06, 1e8) );
        REQUIRE( solver.getParValue(0) == approx(7) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(2.066882758206843, 1e4) );
        REQUIRE( solver.getParValue(3) == approx(2.462337187903755, 1e4) );
        REQUIRE( solver.getParValue(4) == approx(0.1286061342451516, 1e4) );
        REQUIRE( solver.getParValue(5) == approx(2.370219137187704, 1e4) );
        REQUIRE( solver.getParValue(6) == approx(1.537929298611836, 1e5) );
    }
    SECTION( "Active bounds: y1 y2 x1" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3] * pars[6],
                             pars[4] * pars[6],
                             (pars[5] / pars[1]).val,
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(4.789557944916188e-09, 1e6) );
        REQUIRE( solver.getParValue(0) == approx(9.175204378219606) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2.516290078703742) );
        REQUIRE( solver.getParValue(4) == approx(0.1241748579573703) );
        REQUIRE( solver.getParValue(5) == approx(2.1) );
        REQUIRE( solver.getParValue(6) == approx(1) );
    }
    SECTION( "Active bounds: y1 y2 x2" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3] * pars[6],
                             (pars[4] * pars[6]).val,
                             pars[5] / pars[1],
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(8.089281781673563e-09, 1e9) );
        REQUIRE( solver.getParValue(0) == approx(8.65075381982302, 1e4) );
        REQUIRE( solver.getParValue(1) == approx(1.12784211406605, 1e3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2.391312133726172, 1e4) );
        REQUIRE( solver.getParValue(4) == approx(0.2) );
        REQUIRE( solver.getParValue(5) == approx(2.307754231404975, 1e3) );
        REQUIRE( solver.getParValue(6) == approx(1) );
    }
    SECTION( "Active bounds: y1 x1 x2" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             (pars[3] * pars[6]).val,
                             pars[4] * pars[6],
                             pars[5] / pars[1],
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(7.948949141467101e-09, 1e8) );
        REQUIRE( solver.getParValue(0) == approx(8.6232172479134, 1e4) );
        REQUIRE( solver.getParValue(1) == approx(1.129991818462001, 1e3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.1432926201689035, 1e4) );
        REQUIRE( solver.getParValue(5) == approx(2.304776756562628, 1e3) );
        REQUIRE( solver.getParValue(6) == approx(1) );
    }
    SECTION( "Active bounds: y2 x1 x2" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[4] * (pars[1] - pars[2])).val,
                             pars[3] * pars[6],
                             pars[4] * pars[6],
                             pars[5] / pars[1],
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(0.0002133518972430697, 1e7) );
        REQUIRE( solver.getParValue(0) == approx(9.671381025316821) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2.471195528937655) );
        REQUIRE( solver.getParValue(4) == approx(0.2) );
        REQUIRE( solver.getParValue(5) == approx(2.436352525792008) );
        REQUIRE( solver.getParValue(6) == approx(1.282878780376695) );
    }
    SECTION( "Active bounds: y1 y2" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3] * pars[6],
                             (pars[4] * pars[6]).val,
                             pars[5].val,
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(6.665229530210581e-09, 1e6) );
        REQUIRE( solver.getParValue(0) == approx(7.66643074044873) );
        REQUIRE( solver.getParValue(1) == approx(1.520366958728734) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2.149840331428628) );
        REQUIRE( solver.getParValue(4) == approx(0.2) );
        REQUIRE( solver.getParValue(5) == approx(2.1) );
        REQUIRE( solver.getParValue(6) == approx(1) );
    }
    SECTION( "Active bounds: x1 x2" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[1] - pars[2]).val,
                             (pars[3] * pars[6]).val,
                             pars[4] * pars[6],
                             pars[5] / pars[1],
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(1.055552450975446e-08) );
        REQUIRE( solver.getParValue(0) == approx(9.45619615764357) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.1108266932800704) );
        REQUIRE( solver.getParValue(5) == approx(2.419211644009542) );
        REQUIRE( solver.getParValue(6) == approx(1) );
    }
    SECTION( "Active bounds: y1 x2" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             (pars[3] * pars[6]).val,
                             (pars[4] * pars[6]).val,
                             pars[5] / pars[1],
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(1.837955799624567e-08, 1e9) );
        REQUIRE( solver.getParValue(0) == approx(9.133671488841946) );
        REQUIRE( solver.getParValue(1) == approx(1.07743469753068) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.2) );
        REQUIRE( solver.getParValue(5) == approx(2.369246895709936) );
        REQUIRE( solver.getParValue(6) == approx(1) );
    }
    SECTION( "Active bounds: y2 x1" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[4] * (pars[1] - pars[2])).val,
                             pars[3],
                             pars[4] * pars[6],
                             (pars[5] / pars[1]).val,
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, true);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(3.85525275809486e-09) );
        REQUIRE( solver.getParValue(0) == approx(9.161295650686158) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2.513226840732331) );
        REQUIRE( solver.getParValue(4) == approx(0.2) );
        REQUIRE( solver.getParValue(5) == approx(2.1) );
        REQUIRE( solver.getParValue(6) == approx(0.6086752417881788) );
    }
    SECTION( "Active bounds: y1 x1" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             (pars[3] * pars[6]).val,
                             pars[4] * pars[6],
                             (pars[5] / pars[1]).val,
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(4.539054719419867e-09, 1e6) );
        REQUIRE( solver.getParValue(0) == approx(9.972875362221904) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.09633003574503578) );
        REQUIRE( solver.getParValue(5) == approx(2.1) );
        REQUIRE( solver.getParValue(6) == approx(1) );
    }
    SECTION( "Active bounds: y2 x2" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[4] * (pars[1] - pars[2])).val,
                             pars[3] * pars[6],
                             (pars[4] * pars[6]).val,
                             pars[5] / pars[1],
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(5.378351994319598e-08, 1e8) );
        REQUIRE( solver.getParValue(0) == approx(9.40548500418182) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2.566159777315758) );
        REQUIRE( solver.getParValue(4) == approx(0.2) );
        REQUIRE( solver.getParValue(5) == approx(2.403368494537303) );
        REQUIRE( solver.getParValue(6) == approx(1) );
    }
    SECTION( "Active bounds: y1" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             (pars[3] * pars[6]).val,
                             (pars[4] * pars[6]).val,
                             pars[5].val,
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(1.44377272556494e-07) );
        REQUIRE( solver.getParValue(0) == approx(8.13832629325684) );
        REQUIRE( solver.getParValue(1) == approx(1.657624243151093) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.2) );
        REQUIRE( solver.getParValue(5) == approx(2.1) );
        REQUIRE( solver.getParValue(6) == approx(1) );
    }
    SECTION( "Active bounds: y2" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[4] * (pars[1] - pars[2])).val,
                             pars[3] * pars[6],
                             (pars[4] * pars[6]).val,
                             (pars[5] / pars[1]).val,
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(0, 7.0, false);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(0.3053679099467725) );
        REQUIRE( solver.getParValue(0) == approx(7) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(6.663707133578559) );
        REQUIRE( solver.getParValue(4) == approx(0.2) );
        REQUIRE( solver.getParValue(5) == approx(2.1) );
        REQUIRE( solver.getParValue(6) == approx(1) );
    }
    SECTION( "Active bounds: x1" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[1] - pars[2]).val,
                             (pars[3] * pars[6]).val,
                             pars[4] * pars[6],
                             (pars[5] / pars[1]).val,
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(0, 7.0, false);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(7.119223309181207e-07) );
        REQUIRE( solver.getParValue(0) == approx(7) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.02430156104375773) );
        REQUIRE( solver.getParValue(5) == approx(2.1) );
        REQUIRE( solver.getParValue(6) == approx(1) );
    }
    SECTION( "Active bounds: x2" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[4] * (pars[1] - pars[2])).val,
                             (pars[3] * pars[6]).val,
                             (pars[4] * pars[6]).val,
                             pars[5] / pars[1],
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(0, 7.0, false);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(2.851421257605273e-06) );
        REQUIRE( solver.getParValue(0) == approx(7) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.2) );
        REQUIRE( solver.getParValue(5) == approx(3.034543663944978) );
        REQUIRE( solver.getParValue(6) == approx(1) );
    }
    SECTION( "No active bounds" ) {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3] * pars[6],
                             pars[4] * pars[6],
                             pars[5] / pars[1],
                             integration_tolerance) / x;
        } };
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        REQUIRE( solver.chi2() == approx(4.090863439323005e-05) );
        REQUIRE( solver.getParValue(0) == approx(16.70423678510393) );
        REQUIRE( solver.getParValue(1) == approx(1.3) );
        REQUIRE( solver.getParValue(2) == approx(1.2) );
        REQUIRE( solver.getParValue(3) == approx(2) );
        REQUIRE( solver.getParValue(4) == approx(0.2) );
        REQUIRE( solver.getParValue(5) == approx(2.1) );
        REQUIRE( solver.getParValue(6) == approx(1) );
    }
}

TEST_CASE( "Exceptions" )
{
    SECTION( "Single integral" ) {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pow(x, pars[0]) * exp(-pars[1] * x * x);
        } };
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return fix_d[1] * integrate(integrand, pars, 0.0, x, 1e-12);
        } };
        gadfit::initIntegration(2);
        gadfit::LMsolver solver { f };
        setSolverState(solver);
        try {
            solver.fit(10.0);
            REQUIRE( 1 == 0 );
        } catch (const gadfit::InsufficientIntegrationWorkspace& e) {
            e.what();
            REQUIRE( 0 == 0 );
        }
    }
    SECTION( "Double integral (direct)" ) {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x,
                                  const gadfit::AdVar& y) {
            const gadfit::AdVar tmp { 1 + pars[0] * pars[1] * erf(y) };
            return exp(-y) * log((exp(x) - 0.9) * tmp + 1.0) / x;
        } };
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3] * pars[6],
                             pars[4] * pars[6],
                             pars[5] / pars[1],
                             integration_tolerance) / x;
        } };
        gadfit::initIntegration(2);
        gadfit::LMsolver solver { f, MPI_COMM_WORLD };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, true);
        try {
            solver.fit(0.1);
            REQUIRE( 1 == 0 );
        } catch (const gadfit::InsufficientIntegrationWorkspace& e) {
            e.what();
            REQUIRE( 0 == 0 );
        }
    }
}

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

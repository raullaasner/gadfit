#include "fixtures.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <gadfit/lm_solver.h>
#include <gadfit/numerical_integration.h>
#include <omp.h>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// Except for the fit function, the fitting procedures use the same
// initialization.
static auto setSolverState(gadfit::LMsolver& solver) -> void
{
    gadfit::initIntegration();
    solver.addDataset(x_data_single, y_data_single);
    solver.setPar(0, 10.0, true);
    solver.setPar(1, 1.0, true);
    solver.settings.iteration_limit = 4;
    solver.settings.acceleration_threshold = 0.9;
    solver.settings.n_threads = omp_get_max_threads();
}

TEST_CASE("Single integral")
{
    SECTION("No bounds")
    {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pow(x, pars[0]) * exp(-pars[1] * x * x);
        } };
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return fix_d[1] * integrate(integrand, pars, 0.0, x, 1e-12);
        } };
        gadfit::LMsolver solver { f };
        setSolverState(solver);
        solver.fit(10.0);
        CHECK_THAT(solver.chi2(), WithinRel(4994.801048103614, 1e-14));
        gadfit::freeIntegration();
        CHECK_THAT(solver.getParValue(0), WithinRel(9.345693397983833, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.086341822060304, 1e-14));
    }
    SECTION("Lower bound")
    {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[2] * pow(x, pars[0]) * exp(-pars[1] * x * x);
        } };
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            std::vector<gadfit::AdVar> pars2(3);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            return -fix_d[1]
                   * integrate(
                     integrand, pars2, pars[0] / fix_d[0], 0.0, 1e-12);
        } };
        gadfit::LMsolver solver { f };
        setSolverState(solver);
        solver.setPar(1, 1.0, false);
        solver.fit(10.0);
        CHECK_THAT(solver.chi2(), WithinRel(3359.402760955073, 1e-14));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.638686516377437, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.0, 1e-14));
        solver.setPar(1, 1.0, true);
        solver.fit(10.0);
        CHECK_THAT(solver.chi2(), WithinRel(3359.360525697878, 1e-14));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.63837358508365, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.000164288516688, 1e-14));
    }
    SECTION("Lower bound (no parameters in integrand)")
    {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[2] * pow(x, fix_d[2]) * exp(-x * x);
        } };
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            std::vector<gadfit::AdVar> pars2(3);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            return -fix_d[1]
                   * integrate(
                     integrand, pars2, pars[0] / fix_d[0], 0.0, 1e-12);
        } };
        gadfit::LMsolver solver { f };
        setSolverState(solver);
        solver.setPar(1, 1.0, false);
        solver.fit(10.0);
        CHECK_THAT(solver.chi2(), WithinRel(3359.374808601714, 1e-14));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.513801290676248, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.0, 1e-14));
    }
    SECTION("Upper bound")
    {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[2] * pow(x, pars[0]) * exp(-pars[1] * x * x);
        } };
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            std::vector<gadfit::AdVar> pars2(3);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            return fix_d[1]
                   * integrate(
                     integrand, pars2, 0.0, pars[0] / fix_d[0], 1e-12);
        } };
        gadfit::LMsolver solver { f };
        setSolverState(solver);
        solver.setPar(1, 1.0, false);
        solver.fit(10.0);
        CHECK_THAT(solver.chi2(), WithinRel(3359.402760955071, 1e-14));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.638686516377437, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.0, 1e-14));
        solver.setPar(1, 1.0, true);
        solver.fit(10.0);
        CHECK_THAT(solver.chi2(), WithinRel(3359.360525697879, 1e-14));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.638373585083652, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.000164288516688, 1e-14));
    }
    SECTION("Upper bound (no parameters in integrand)")
    {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[0] * pow(x, fix_d[2]) * exp(-x * x);
        } };
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            std::vector<gadfit::AdVar> pars2 { gadfit::AdVar {
              x, gadfit::passive_idx } };
            return fix_d[1]
                   * integrate(
                     integrand, pars2, 0.0, pars[0] / fix_d[0], 1e-12);
        } };
        gadfit::LMsolver solver { f };
        setSolverState(solver);
        solver.setPar(1, 1.0, false);
        solver.fit(10.0);
        CHECK_THAT(solver.chi2(), WithinRel(3359.374808601714, 1e-14));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.513801290676248, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.0, 1e-14));
    }
    SECTION("Both bounds")
    {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[2] * pow(x, pars[0]) * exp(-pars[1] * x * x);
        } };
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            std::vector<gadfit::AdVar> pars2(3);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            return -fix_d[1]
                   * integrate(
                     integrand, pars2, pars[0] / fix_d[0], pars[1], 1e-12);
        } };
        gadfit::LMsolver solver { f };
        setSolverState(solver);
        solver.setPar(1, 1.0, false);
        solver.fit(10.0);
        CHECK_THAT(solver.chi2(), WithinRel(3359.392136789901, 1e-14));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.664371097350363, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.0, 1e-14));
        solver.setPar(1, 1.0, true);
        solver.fit(10.0);
        CHECK_THAT(solver.chi2(), WithinRel(3359.360525697834, 1e-14));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.664108472227593, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.000124158231295, 1e-14));
    }
    SECTION("Both bounds (lower inactive)")
    {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[2] * pow(x, pars[0]) * exp(-pars[1] * x * x);
        } };
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            std::vector<gadfit::AdVar> pars2(3);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            return -fix_d[1]
                   * integrate(
                     integrand, pars2, pars[1], pars[0] / fix_d[0], 1e-12);
        } };
        gadfit::LMsolver solver { f };
        setSolverState(solver);
        solver.setPar(1, 1.0, false);
        solver.fit(10.0);
        CHECK_THAT(solver.chi2(), WithinRel(96283.63738642586, 1e-14));
        CHECK_THAT(solver.getParValue(0), WithinRel(4.023936467213234, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.0, 1e-14));
    }
    SECTION("Both bounds (no parameters in integrand)")
    {
        const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                                  const gadfit::AdVar& x) {
            return pars[0] * pow(x, fix_d[2]) * exp(-x * x);
        } };
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            std::vector<gadfit::AdVar> pars2 { gadfit::AdVar {
              x, gadfit::passive_idx } };
            return -fix_d[1]
                   * integrate(
                     integrand, pars2, pars[0] / fix_d[0], pars[1], 1e-12);
        } };
        gadfit::LMsolver solver { f };
        setSolverState(solver);
        solver.fit(10.0);
        CHECK_THAT(solver.chi2(), WithinRel(3359.360587615625, 1e-14));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.834021674777725, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.301193106585963, 1e-14));
    }
}

static auto setSolverStateNested(gadfit::LMsolver& solver) -> void
{
    // Normally the integration initialization function is called
    // automatically unless dealing with nested integrals in which
    // case the nesting level (max 2) must be given manually. This is
    // in contrast with the direct double integral formula where all
    // the relevant work arrays are allocated automatically.
    gadfit::initIntegration(1000, 2);
    solver.addDataset(x_data_double, y_data_double, weights_double);
    solver.setPar(0, 7.0, true);
    solver.settings.n_threads = omp_get_max_threads();
    solver.settings.iteration_limit = 2;
    solver.settings.acceleration_threshold = 0.9;
}

TEST_CASE("Double integral (nested)")
{
    // Convergence criterion for the inner and outer integrals
    constexpr double tol_inner { 1e-3 };
    constexpr double tol_outer { 1e-2 };
    const auto integrand_inner { [](const std::vector<gadfit::AdVar>& pars,
                                    const gadfit::AdVar& x) {
        return log((exp(x) - 0.9) * pars[0] + 1.0) / x;
    } };
    SECTION("Active bounds: y1 y2 x1 x2")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x)
                   * integrate(integrand_inner,
                               pars2,
                               pars[3],
                               pars[4] * pars[2] / pars[1],
                               tol_inner);
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
                             tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(0.2131810550497416, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(15.26735468164642, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.386383105456653, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(0.8486391644471797, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(1.674240469615365, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.1885677628244937, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(1.941800275111635, 1e-12));
    }
    SECTION("Active bounds: y1 y2 x1")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x)
                   * integrate(integrand_inner,
                               pars2,
                               pars[3],
                               pars[4] * fix_d[2],
                               tol_inner);
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
                             tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(20529.86214956253, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.545073737454485, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.050947728780064, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.407011447112184, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.246597745517819, 1e-12));
        CHECK_THAT(solver.getParValue(4),
                   WithinRel(0.09581638730674913, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.1, 1e-12));
    }
    SECTION("Active bounds: y1 y2 x2")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x)
                   * integrate(integrand_inner,
                               pars2,
                               fix_d[3],
                               pars[4] * pars[2] / pars[1],
                               tol_inner);
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
                             tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(8.511262427426729, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(31.35420758618348, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.343236097449233, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(0.9880791189004298, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(1.915159447508319, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.6301502301640346, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.041471780774121, 1e-12));
    }
    SECTION("Active bounds: y1 x1 x2")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x)
                   * integrate(integrand_inner,
                               pars2,
                               pars[3],
                               pars[4] * pars[2] / pars[1],
                               tol_inner);
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
                             tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(0.5452442448229686, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(14.51912799259439, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.401803657027402, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(0.7703969798462069, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2243476865643863, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(1.911960222088238, 1e-12));
    }
    SECTION("Active bounds: y2 x1 x2")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x)
                   * integrate(integrand_inner,
                               pars2,
                               pars[3],
                               pars[4] * pars[2] / pars[1],
                               tol_inner);
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
                              tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(0.5452442448229419, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(14.51912799259429, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.401803657027403, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(0.7703969798462061, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2243476865643837, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(1.911960222088238, 1e-12));
    }
    SECTION("Active bounds: y1 y2")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x)
                   * integrate(
                     integrand_inner, pars2, 1 / fix_d[16], pars[4], tol_inner);
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
                             tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(31829.01194465925, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(8.429293418556341, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.569188491899031, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.476297876086944, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.350856627400455, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.1445015201991888, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.1, 1e-12));
    }
    SECTION("Active bounds: x1 x2")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x)
                   * integrate(integrand_inner,
                               pars2,
                               pars[3],
                               pars[4] * pars[2] / pars[1],
                               tol_inner);
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
                             tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(0.0638207048968614, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(15.54318299637472, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.337653916227864, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.060422119015556, 1e-12));
    }
    SECTION("Active bounds: y1 x2")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x)
                   * integrate(integrand_inner,
                               pars2,
                               fix_d[3],
                               pars[4] * pars[2] / pars[1],
                               tol_inner);
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
                             tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(8.310466833011295, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(31.05730169163706, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.337447872754693, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(0.9731467848827562, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.6676623753034178, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.042477682607804, 1e-12));
    }
    SECTION("Active bounds: y2 x1")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return -exp(-x)
                   * integrate(
                     integrand_inner, pars2, pars[4], pars[3], tol_inner);
        } };
        const auto f_double { [&](const std::vector<gadfit::AdVar>& pars,
                                  const double x) {
            std::vector<gadfit::AdVar> pars2(5);
            pars2[0] = pars[0];
            pars2[1] = pars[1];
            pars2[2] = gadfit::AdVar { x, gadfit::passive_idx };
            pars2[3] = pars[4];
            pars2[4] = pars[5];
            return integrate(
                     integrand_outer, pars2, pars[4], pars[3], tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(20530.20016213086, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(72.09812547421947, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(10.90218525163188, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(15.56263330043302, 1e-12));
    }
    SECTION("Active bounds: y1 x1")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return -exp(-x)
                   * integrate(integrand_inner,
                               pars2,
                               pars[4] * pars[2] / pars[1],
                               pars[3],
                               tol_inner);
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
                             tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(0.3708459104616477, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(13.42618730322273, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.40787919245722, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(0.7574214524465727, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(1.896290194086714, 1e-12));
    }
    SECTION("Active bounds: y2 x2")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x)
                   * integrate(integrand_inner,
                               pars2,
                               pars[3],
                               pars[4] * pars[2] / pars[1],
                               tol_inner);
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
                              tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(0.3645778424347108, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(13.45556877476804, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.408061308403743, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(0.7570256924263207, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(1.895981142726112, 1e-12));
    }
    SECTION("Active bounds: y1")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x)
                   * integrate(
                     integrand_inner, pars2, pars[3], pars[4], tol_inner);
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
                             tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(33404.97047824427, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(18.74611668457635, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(3.127350705902004, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.1, 1e-12));
    }
    SECTION("Active bounds: y2")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x)
                   * integrate(
                     integrand_inner, pars2, pars[3], pars[4], tol_inner);
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
                              tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(33404.97047824427, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(18.74611668457635, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(3.127350705902005, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.1, 1e-12));
    }
    SECTION("Active bounds: x1")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return -exp(-x)
                   * integrate(
                     integrand_inner, pars2, pars[4], pars[3], tol_inner);
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
                             tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(20529.86874184859, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(80.95988477911882, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(15.63231901313966, 1e-12));
    }
    SECTION("Active bounds: x2")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x)
                   * integrate(
                     integrand_inner, pars2, pars[3], pars[4], tol_inner);
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
                             tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(20529.86896231501, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(80.95988738910319, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(15.63232441412172, 1e-12));
    }
    SECTION("No active bounds")
    {
        const auto integrand_outer { [&](const std::vector<gadfit::AdVar>& pars,
                                         const gadfit::AdVar& x) {
            std::vector<gadfit::AdVar> pars2 { 1 + pars[0] * pars[1] * erf(x) };
            return exp(-x)
                   * integrate(integrand_inner,
                               pars2,
                               pars[3],
                               pars[4] * pars[2] / pars[1],
                               tol_inner);
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
                             tol_outer)
                   / x;
        } };
        gadfit::LMsolver solver { f_double };
        setSolverStateNested(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.settings.iteration_limit = 1;
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinRel(158.6303014282949, 1e-12));
        CHECK_THAT(solver.getParValue(0), WithinRel(24.35593003546224, 1e-12));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-12));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-12));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-12));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-12));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.1, 1e-12));
    }
}

static auto setSolverStateDirect(gadfit::LMsolver& solver) -> void
{
    gadfit::initIntegration();
    solver.addDataset(x_data_double, y_data_double_direct);
    solver.setPar(0, 7.0, true);
    solver.settings.iteration_limit = 2;
    solver.settings.acceleration_threshold = 0.9;
    solver.settings.n_threads = omp_get_max_threads();
}

TEST_CASE("Double integral (direct)")
{
    // Convergence criterion for the 2D (direct) integrals
    constexpr double integration_tolerance { 1e-4 };

    const auto integrand { [](const std::vector<gadfit::AdVar>& pars,
                              const gadfit::AdVar& x,
                              const gadfit::AdVar& y) {
        const gadfit::AdVar tmp { 1 + pars[0] * pars[1] * erf(y) };
        return exp(-y) * log((exp(x) - 0.9) * tmp + 1.0) / x;
    } };
    const auto f { [&](const std::vector<gadfit::AdVar>& pars, const double x) {
        return integrate(integrand,
                         pars,
                         pars[4] * (pars[1] - pars[2]),
                         pars[3] * pars[6],
                         pars[4] * pars[6],
                         pars[5] / pars[1],
                         integration_tolerance)
               / x;
    } };
    SECTION("Active bounds: y1 y2 x1 x2")
    {
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(0, 7.0, false);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, true);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(1.654886495874691e-06, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(7, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(2.066882698115843, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.462337242876448, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.1286061153388938, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.370219247402999, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.537928956329218, 1e-7));
    }
    SECTION("Active bounds: y1 y2 x1")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3] * pars[6],
                             pars[4] * pars[6],
                             (pars[5] / pars[1]).val,
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(4.790523057594158e-09, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.175204980541729, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.516290186341045, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.1241748448388979, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.1, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.0, 1e-7));
    }
    SECTION("Active bounds: y1 y2 x2")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3] * pars[6],
                             (pars[4] * pars[6]).val,
                             pars[5] / pars[1],
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(8.068219436506581e-09, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(8.65075393701988, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.127842104542012, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.391312159920373, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.307754246034183, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.0, 1e-7));
    }
    SECTION("Active bounds: y1 x1 x2")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             (pars[3] * pars[6]).val,
                             pars[4] * pars[6],
                             pars[5] / pars[1],
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(7.949911068776061e-09, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(8.623217421789654, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.12999180500759, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.143292615316067, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.304776775635509, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.0, 1e-7));
    }
    SECTION("Active bounds: y2 x1 x2")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[4] * (pars[1] - pars[2])).val,
                             pars[3] * pars[6],
                             pars[4] * pars[6],
                             pars[5] / pars[1],
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(0.000213350703610027, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.671381032914667, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.471195537765232, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.436352543805309, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.282878741340198, 1e-7));
    }
    SECTION("Active bounds: y1 y2")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3] * pars[6],
                             (pars[4] * pars[6]).val,
                             pars[5].val,
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(6.66506150605225e-09, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(7.666430772544548, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.520366990688495, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.149840327725317, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.1, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.0, 1e-7));
    }
    SECTION("Active bounds: x1 x2")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[1] - pars[2]).val,
                             (pars[3] * pars[6]).val,
                             pars[4] * pars[6],
                             pars[5] / pars[1],
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(1.05553491668674e-08, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.456196153046807, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.1108266934567069, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.419211642876849, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.0, 1e-7));
    }
    SECTION("Active bounds: y1 x2")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             (pars[3] * pars[6]).val,
                             (pars[4] * pars[6]).val,
                             pars[5] / pars[1],
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(1.837877829573166e-08, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.13367142357661, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.077434702465759, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.369246887944458, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.0, 1e-7));
    }
    SECTION("Active bounds: y2 x1")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[4] * (pars[1] - pars[2])).val,
                             pars[3],
                             pars[4] * pars[6],
                             (pars[5] / pars[1]).val,
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, true);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(3.85575335670611e-09, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.161296077178266, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.513226918313678, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.1, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(0.6086751941736143, 1e-7));
    }
    SECTION("Active bounds: y1 x1")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             (pars[3] * pars[6]).val,
                             pars[4] * pars[6],
                             (pars[5] / pars[1]).val,
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(4.556742521509683e-09, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.972875346712668, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.09633003605472064, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.1, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.0, 1e-7));
    }
    SECTION("Active bounds: y2 x2")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[4] * (pars[1] - pars[2])).val,
                             pars[3] * pars[6],
                             (pars[4] * pars[6]).val,
                             pars[5] / pars[1],
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(5.373563892617068e-08, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(9.405485170085401, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.5661598112606, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.403368518336621, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.0, 1e-7));
    }
    SECTION("Active bounds: y1")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             (pars[3] * pars[6]).val,
                             (pars[4] * pars[6]).val,
                             pars[5].val,
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, true);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(1.443756776956618e-07, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(8.13832625826087, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.657624293024702, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.1, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.0, 1e-7));
    }
    SECTION("Active bounds: y2")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[4] * (pars[1] - pars[2])).val,
                             pars[3] * pars[6],
                             (pars[4] * pars[6]).val,
                             (pars[5] / pars[1]).val,
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(0, 7.0, false);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, true);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(0.3053680170120716, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(7, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(6.663707134981233, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.1, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.0, 1e-7));
    }
    SECTION("Active bounds: x1")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[1] - pars[2]).val,
                             (pars[3] * pars[6]).val,
                             pars[4] * pars[6],
                             (pars[5] / pars[1]).val,
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(0, 7.0, false);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, true);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(7.119221262116694e-07, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(7, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.02430156976447609, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.1, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.0, 1e-7));
    }
    SECTION("Active bounds: x2")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             (pars[4] * (pars[1] - pars[2])).val,
                             (pars[3] * pars[6]).val,
                             (pars[4] * pars[6]).val,
                             pars[5] / pars[1],
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(0, 7.0, false);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, true);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(2.851428619095947e-06, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(7, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(3.034543683583202, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.0, 1e-7));
    }
    SECTION("No active bounds")
    {
        const auto f { [&](const std::vector<gadfit::AdVar>& pars,
                           const double x) {
            return integrate(integrand,
                             pars,
                             pars[4] * (pars[1] - pars[2]),
                             pars[3] * pars[6],
                             pars[4] * pars[6],
                             pars[5] / pars[1],
                             integration_tolerance)
                   / x;
        } };
        gadfit::LMsolver solver { f };
        setSolverStateDirect(solver);
        solver.setPar(1, 1.3, false);
        solver.setPar(2, 1.2, false);
        solver.setPar(3, 2.0, false);
        solver.setPar(4, 0.2, false);
        solver.setPar(5, 2.1, false);
        solver.setPar(6, 1.0, false);
        solver.fit(0.1);
        CHECK_THAT(solver.chi2(), WithinAbs(4.090863893678671e-05, 1e-9));
        CHECK_THAT(solver.getParValue(0), WithinRel(16.70423680614829, 1e-7));
        CHECK_THAT(solver.getParValue(1), WithinRel(1.3, 1e-7));
        CHECK_THAT(solver.getParValue(2), WithinRel(1.2, 1e-7));
        CHECK_THAT(solver.getParValue(3), WithinRel(2.0, 1e-7));
        CHECK_THAT(solver.getParValue(4), WithinRel(0.2, 1e-7));
        CHECK_THAT(solver.getParValue(5), WithinRel(2.1, 1e-7));
        CHECK_THAT(solver.getParValue(6), WithinRel(1.0, 1e-7));
    }
}

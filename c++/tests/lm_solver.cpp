#include "fixtures.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <gadfit/lm_solver.h>
#include <numeric>
#include <omp.h>

using Catch::Matchers::WithinRel;

static auto exponential(const std::vector<gadfit::AdVar>& parameters,
                        const double x) -> gadfit::AdVar
{
    using gadfit::AdVar;
    const AdVar& I0 { parameters[0] };
    const AdVar& tau { parameters[1] };
    const AdVar& bgr { parameters[2] };
    return I0 * exp(-x / tau) + bgr;
}

TEST_CASE("Indexing scheme")
{
    gadfit::LMsolver solver { exponential };
    solver.addDataset(x_data_1, y_data_1);
    solver.addDataset(x_data_2, y_data_2);
    solver.settings.iteration_limit = 4;
    solver.settings.n_threads = omp_get_max_threads();

    SECTION("Active: I0-0, bgr-0, I0-1, bgr-1, tau")
    {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(11620.0867270475, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(17.8650243622964, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(39.77705004578393, 1e-14));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(13.57729652858559, 1e-14));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(129.0275065609783, 1e-14));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(16.09079665934463, 1e-14));
    }
    SECTION("Active: bgr-0, bgr-1, tau")
    {
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], false, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(153628.8903849508, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(31.95892116514992, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0), WithinRel(fix_d[0], 1e-14));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(17.81484199806565, 1e-14));
        CHECK_THAT(solver.getParValue(0, 1), WithinRel(fix_d[4], 1e-14));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(36.73244337347508, 1e-14));
    }
    SECTION("Active: I0-0, I0-1, tau")
    {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(10810.65153981582, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(21.30228862988602, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(56.42893238415446, 1e-14));
        CHECK_THAT(solver.getParValue(2, 0), WithinRel(fix_d[1], 1e-14));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(139.4901380914605, 1e-14));
        CHECK_THAT(solver.getParValue(2, 1), WithinRel(fix_d[5], 1e-14));
    }
    SECTION("Active: tau")
    {
        solver.setPar(0, fix_d[16], false, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[17], false, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(51624.83919460665, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(10.99329301695744, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0), WithinRel(fix_d[16], 1e-14));
        CHECK_THAT(solver.getParValue(2, 0), WithinRel(fix_d[1], 1e-14));
        CHECK_THAT(solver.getParValue(0, 1), WithinRel(fix_d[17], 1e-14));
        CHECK_THAT(solver.getParValue(2, 1), WithinRel(fix_d[5], 1e-14));
    }
    SECTION("Active: bgr-0, I0-1, bgr-1, tau")
    {
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(15974.61260816282, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(20.47926391663428, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0), WithinRel(fix_d[0], 1e-14));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(18.47600900933105, 1e-14));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(143.0431252627765, 1e-14));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(9.453915929181857, 1e-14));
    }
    SECTION("Active: I0-0, bgr-0, bgr-1, tau")
    {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0, "bgr");
        solver.setPar(0, fix_d[4], false, 1, "I0");
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(145780.4588072044, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(8.408237957600141, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(45.87087327322397, 1e-14));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(16.59126759913267, 1e-14));
        CHECK_THAT(solver.getParValue(0, 1), WithinRel(fix_d[4], 1e-14));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(36.38255403506549, 1e-14));
    }
    SECTION("Active: I0-0, I0-1, bgr-1, tau")
    {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(11623.17388899667, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(20.61333132315124, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(56.5139576021328, 1e-14));
        CHECK_THAT(solver.getParValue(2, 0), WithinRel(fix_d[1], 1e-14));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(134.8973104943701, 1e-14));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(11.77612256514583, 1e-14));
    }
    SECTION("Active: I0-0, bgr-0, I0-1, tau")
    {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(30610.67204238365, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(16.54682323514368, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(29.98632400541692, 1e-14));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(12.99477135618182, 1e-14));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(124.6991105597198, 1e-14));
        CHECK_THAT(solver.getParValue(2, 1), WithinRel(fix_d[5], 1e-14));
    }
    SECTION("Active: I0-0, bgr-1, tau")
    {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[4], false, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(150672.9869101836, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(16.73368044360274, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(53.73848940201638, 1e-14));
        CHECK_THAT(solver.getParValue(2, 0), WithinRel(fix_d[1], 1e-14));
        CHECK_THAT(solver.getParValue(0, 1), WithinRel(fix_d[4], 1e-14));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(36.50405720192947, 1e-14));
    }
    SECTION("Active: bgr-0, I0-1, tau")
    {
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], true, "global parameter");
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(15348.60122706107, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(21.87456778662339, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0), WithinRel(fix_d[0], 1e-14));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(18.39176693290169, 1e-14));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(147.1783948678938, 1e-14));
        CHECK_THAT(solver.getParValue(2, 1), WithinRel(fix_d[5], 1e-14));
    }
}

static auto prepareSolver(gadfit::LMsolver& solver) -> void
{
    const auto sum { [](const std::vector<double>& x, const int limit = -1) {
        return std::accumulate(
          x.cbegin(), limit == -1 ? x.cend() : x.cbegin() + limit, 0.0);
    } };
    solver.addDataset(x_data_1, y_data_1);
    solver.addDataset(x_data_2, y_data_2);
    solver.settings.iteration_limit = 4;
    solver.settings.n_threads = omp_get_max_threads();
    solver.setPar(0, fix_d[0], true, 0);
    solver.setPar(2, fix_d[1], true, 0);
    solver.setPar(0, fix_d[4], true, 1);
    solver.setPar(2, fix_d[5], true, 1);
    solver.setPar(1, fix_d[3], true);
}

TEST_CASE("Access functions")
{
    const auto sum { [](const std::vector<double>& x, const int limit = -1) {
        return std::accumulate(
          x.cbegin(), limit == -1 ? x.cend() : x.cbegin() + limit, 0.0);
    } };
    gadfit::LMsolver solver { exponential };
    prepareSolver(solver);
    CHECK_THAT(solver.getValue(fix_d[2]), WithinRel(2.960644529912441, 1e-13));
    solver.fit(1.0);
    CHECK(solver.degreesOfFreedom() == 195);
    CHECK_THAT(solver.getValue(fix_d[2]), WithinRel(36.39905496310919, 1e-13));
    CHECK_THAT(sum(solver.getJacobian()), WithinRel(353.6485673748526, 1e-13));
    CHECK_THAT(sum(solver.getJTJ(), 5), WithinRel(580.3488115472484, 1e-13));
    CHECK_THAT(sum(solver.getDTD(), 5), WithinRel(34340.67196549198, 1e-13));
    CHECK_THAT(sum(solver.getLeftSide(), 5),
               WithinRel(614.6894835127404, 1e-13));
    CHECK_THAT(sum(solver.getRightSide(), 5),
               WithinRel(4410.585412402701, 1e-13));
    CHECK_THAT(sum(solver.getResiduals()), WithinRel(213.3530475167945, 1e-13));
}

TEST_CASE("computation of JTJ^-1")
{
    const auto sum { [](const std::vector<double>& x, const int limit = -1) {
        return std::accumulate(
          x.cbegin(), limit == -1 ? x.cend() : x.cbegin() + limit, 0.0);
    } };
    gadfit::LMsolver solver { exponential };
    prepareSolver(solver);
    solver.fit(1.0);
    const auto JTJ { solver.getJTJ() };
    const auto inv_JTJ { solver.getInvJTJ() };
    std::vector<double> result(JTJ.size(), 0.0);
    const int n_parameters { static_cast<int>(
      std::sqrt(static_cast<double>(JTJ.size()))) };
    for (int i {}; i < n_parameters; ++i) {
        for (int j {}; j < n_parameters; ++j) {
            for (int k {}; k < n_parameters; ++k) {
                result[i * n_parameters + j] +=
                  inv_JTJ[i * n_parameters + k] * JTJ[k * n_parameters + j];
            }
        }
    }
    for (int i {}; i < n_parameters; ++i) {
        for (int j {}; j < n_parameters; ++j) {
            if (i == j) {
                CHECK_THAT(result[i * n_parameters + j], WithinRel(1.0, 1e-14));
            } else {
                CHECK_THAT(result[i * n_parameters + j] + 1.0,
                           WithinRel(1.0, 1e-14));
            }
        }
    }
}

TEST_CASE("Exceptions")
{
    gadfit::LMsolver solver { exponential };
    solver.settings.iteration_limit = 4;
    solver.settings.n_threads = omp_get_max_threads();

    SECTION("Incorrect number of calls")
    {
        solver.addDataset(x_data_1, y_data_1);
        solver.setPar(0, fix_d[0], true, 0);
        CHECK_THROWS_AS(solver.addDataset(x_data_2, y_data_2),
                        gadfit::LateAddDatasetCall);
    }
    SECTION("Invalid data set index")
    {
        solver.addDataset(x_data_1, y_data_1);
        solver.addDataset(x_data_2, y_data_2);
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        CHECK_THROWS_AS(solver.setPar(1, fix_d[3], true, 2),
                        gadfit::SetParInvalidIndex);
    }
    SECTION("Uninitialized fitting parameter")
    {
        solver.addDataset(x_data_1, y_data_1);
        solver.addDataset(x_data_2, y_data_2);
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(1, fix_d[3], true);
        CHECK_THROWS_AS(solver.fit(1.0), gadfit::UninitializedParameter);
    }
    SECTION("Too few data points (or too many fitting parameters)")
    {
        solver.addDataset(2, x_data_1.data(), y_data_1.data());
        solver.addDataset(2, x_data_2.data(), y_data_2.data());
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        CHECK_THROWS_AS(solver.fit(1.0), gadfit::NegativeDegreesOfFreedom);
    }
    SECTION("No degrees of freedom")
    {
        solver.addDataset(3, x_data_1.data(), y_data_1.data());
        solver.addDataset(2, x_data_2.data(), y_data_2.data());
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        CHECK_THAT(solver.getParValue(1), WithinRel(2.945868346541778, 1e-12));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(7.351966871429338, 1e-12));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(49.68674387147222, 1e-12));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(-13.18731292934346, 1e-12));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(162.1781165060037, 1e-12));
    }
    SECTION("No active parameters")
    {
        solver.addDataset(x_data_1, y_data_1);
        solver.addDataset(x_data_2, y_data_2);
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[4], false, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], false);
        CHECK_THROWS_AS(solver.fit(1.0), gadfit::NoFittingParameters);
    }
    SECTION("Active: bgr-0, I0-1")
    {
        solver.addDataset(x_data_1, y_data_1);
        solver.addDataset(x_data_2, y_data_2);
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[12], false);
        CHECK_THROWS_AS(solver.fit(1.0), gadfit::NoGlobalParameters);
    }
}

TEST_CASE("Number of iterations")
{
    gadfit::LMsolver solver { exponential };
    solver.settings.n_threads = omp_get_max_threads();
    solver.addDataset(x_data_1, y_data_1);
    solver.addDataset(x_data_2, y_data_2);
    solver.setPar(0, fix_d[0], true, 0);
    solver.setPar(2, fix_d[1], true, 0);
    solver.setPar(0, fix_d[4], true, 1);
    solver.setPar(2, fix_d[5], true, 1);
    solver.setPar(1, fix_d[3], true);

    SECTION("No iterations")
    {
        solver.settings.iteration_limit = 0;
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(284681.4650859562, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(0.5356792380861322, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(6.13604207015635, 1e-14));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(2.960644474827888, 1e-14));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(-1.472720596147903, 1e-14));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(4.251266120087788, 1e-14));
    }
    SECTION("No iteration limit")
    {
        // Without an iteration limit the total number of iterations
        // could be sensitive to the number of threads. Only target
        // single precision accuracy here.
        solver.settings.iteration_limit = 100;
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(5640.175130917765, 1e-8));
        CHECK_THAT(solver.getParValue(1), WithinRel(20.85609539787557, 1e-8));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(46.44788540145462, 1e-8));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(10.32140443380387, 1e-8));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(152.2711588123377, 1e-8));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(5.533936911147024, 1e-8));
    }
}

TEST_CASE("Constraining the damping matrix DTD")
{
    gadfit::LMsolver solver { exponential };
    solver.settings.n_threads = omp_get_max_threads();
    solver.addDataset(x_data_1, y_data_1);
    solver.addDataset(x_data_2, y_data_2);
    solver.setPar(0, fix_d[0], true, 0);
    solver.setPar(2, fix_d[1], true, 0);
    solver.setPar(0, fix_d[4], true, 1);
    solver.setPar(2, fix_d[5], true, 1);
    solver.setPar(1, fix_d[3], true);
    solver.settings.iteration_limit = 5;

    SECTION("No constraints")
    {
        solver.settings.damp_max = false;
        // Next line should have no effect
        solver.settings.DTD_min = { 2.0, 1.0, 3.0, 1.0, 7.0 };
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(5761.320550200902, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(20.15808767822605, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(46.33952661438988, 1e-14));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(10.40696426744941, 1e-14));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(151.8449704289686, 1e-14));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(5.793008018375257, 1e-14));
    }
    SECTION("Default constraint")
    {
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(5687.451130305415, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(21.01892108898218, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(46.18357253310398, 1e-14));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(10.48386354002993, 1e-14));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(151.5283959798012, 1e-14));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(6.087406702661871, 1e-14));
    }
    SECTION("Minimal values for the diagonal elements")
    {
        solver.settings.DTD_min = { 2.0, 1.0, 3.0, 1.0, 7.0 };
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(5640.44344354764, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(20.8194135048056, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(46.44560357121117, 1e-14));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(10.33411834275264, 1e-14));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(152.2267670731584, 1e-14));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(5.560463422860977, 1e-14));
    }
}

TEST_CASE("Geodesic acceleration")
{
    gadfit::LMsolver solver { exponential };
    solver.settings.n_threads = omp_get_max_threads();
    solver.addDataset(x_data_1, y_data_1);
    solver.addDataset(x_data_2, y_data_2);
    solver.setPar(0, fix_d[0], true, 0);
    solver.setPar(2, fix_d[1], true, 0);
    solver.setPar(0, fix_d[4], true, 1);
    solver.setPar(2, fix_d[5], true, 1);
    solver.setPar(1, fix_d[3], true);
    solver.settings.iteration_limit = 5;
    solver.settings.acceleration_threshold = 0.9;
    solver.settings.verbosity =
      gadfit::io::delta1 | gadfit::io::delta2 | gadfit::io::timings;
    solver.fit(1.0);
    CHECK_THAT(solver.chi2(), WithinRel(5641.66030550462, 1e-14));
    CHECK_THAT(solver.getParValue(1), WithinRel(20.70654799943915, 1e-14));
    CHECK_THAT(solver.getParValue(0, 0), WithinRel(46.48065799723029, 1e-14));
    CHECK_THAT(solver.getParValue(2, 0), WithinRel(10.39142422387268, 1e-14));
    CHECK_THAT(solver.getParValue(0, 1), WithinRel(152.4514268293043, 1e-14));
    CHECK_THAT(solver.getParValue(2, 1), WithinRel(5.748941149916498, 1e-14));
}

TEST_CASE("Loss functions")
{
    gadfit::LMsolver solver { exponential };
    solver.settings.n_threads = omp_get_max_threads();
    solver.addDataset(x_data_1, y_data_1);
    solver.addDataset(x_data_2, y_data_2);
    solver.setPar(0, fix_d[0], true, 0);
    solver.setPar(2, fix_d[1], true, 0);
    solver.setPar(0, fix_d[4], true, 1);
    solver.setPar(2, fix_d[5], true, 1);
    solver.setPar(1, fix_d[3], true);
    solver.settings.iteration_limit = 5;

    SECTION("Linear")
    {
        solver.settings.loss = gadfit::Loss::linear;
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(5687.451130305415, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(21.01892108898218, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(46.18357253310398, 1e-14));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(10.48386354002993, 1e-14));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(151.5283959798012, 1e-14));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(6.087406702661871, 1e-14));
    }
    SECTION("Cauchy")
    {
        solver.settings.loss = gadfit::Loss::cauchy;
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(16869.67716299524, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(17.45448014750576, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(40.28201426242013, 1e-14));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(9.278480584355261, 1e-14));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(132.6242198264016, 1e-14));
        CHECK_THAT(solver.getParValue(2, 1), WithinRel(6.7051221338403, 1e-14));
    }
    SECTION("Huber")
    {
        solver.settings.iteration_limit = 2;
        solver.settings.loss = gadfit::Loss::huber;
        solver.fit(1.0);
        CHECK_THAT(solver.chi2(), WithinRel(123695.8709974329, 1e-14));
        CHECK_THAT(solver.getParValue(1), WithinRel(4.643243104460152, 1e-14));
        CHECK_THAT(solver.getParValue(0, 0),
                   WithinRel(52.6348486049053, 1e-14));
        CHECK_THAT(solver.getParValue(2, 0),
                   WithinRel(7.874003370245958, 1e-14));
        CHECK_THAT(solver.getParValue(0, 1),
                   WithinRel(166.3872296081963, 1e-14));
        CHECK_THAT(solver.getParValue(2, 1),
                   WithinRel(7.690335499679898, 1e-14));
    }
}

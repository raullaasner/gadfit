#include "testing.h"

#include "lm_solver_data.h"

#include <lm_solver.h>
#include <spdlog/spdlog.h>

static auto exponential(const std::vector<gadfit::AdVar>& parameters,
                        const double x) -> gadfit::AdVar
{
    using gadfit::AdVar;
    const AdVar& I0 { parameters[0] };
    const AdVar& tau { parameters[1] };
    const AdVar& bgr { parameters[2] };
    return I0 * exp(-x / tau) + bgr;
}

TEST_CASE( "Indexing scheme" )
{
    spdlog::set_level(spdlog::level::off);
    gadfit::LMsolver solver { exponential, MPI_COMM_WORLD };
    solver.addDataset(x_data_1, y_data_1);
    solver.addDataset(x_data_2, y_data_2);
    solver.settings.iteration_limit = 4;

    SECTION( "Active: I0-0, bgr-0, I0-1, bgr-1, tau" ) {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(11620.08672704751) );
        REQUIRE( solver.getParValue(1) == approx(17.86502436229641) );
        REQUIRE( solver.getParValue(0, 0) == approx(39.77705004578391) );
        REQUIRE( solver.getParValue(2, 0) == approx(13.57729652858559) );
        REQUIRE( solver.getParValue(0, 1) == approx(129.0275065609782) );
        REQUIRE( solver.getParValue(2, 1) == approx(16.09079665934463) );
    }
    SECTION( "Active: bgr-0, bgr-1, tau" ) {
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], false, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(153628.8903849508) );
        REQUIRE( solver.getParValue(1) == approx(31.95892116514992) );
        REQUIRE( solver.getParValue(0, 0) == approx(fix_d[0]) );
        REQUIRE( solver.getParValue(2, 0) == approx(17.81484199806565) );
        REQUIRE( solver.getParValue(0, 1) == approx(fix_d[4]) );
        REQUIRE( solver.getParValue(2, 1) == approx(36.73244337347508) );
    }
    SECTION( "Active: I0-0, I0-1, tau" ) {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(10810.65153981583) );
        REQUIRE( solver.getParValue(1) == approx(21.30228862988602) );
        REQUIRE( solver.getParValue(0, 0) == approx(56.42893238415446) );
        REQUIRE( solver.getParValue(2, 0) == approx(fix_d[1]) );
        REQUIRE( solver.getParValue(0, 1) == approx(139.4901380914605) );
        REQUIRE( solver.getParValue(2, 1) == approx(fix_d[5]) );
    }
    SECTION( "Active: tau" ) {
        solver.setPar(0, fix_d[16], false, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[17], false, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(51624.83919460662) );
        REQUIRE( solver.getParValue(1) == approx(10.99329301695744) );
        REQUIRE( solver.getParValue(0, 0) == approx(fix_d[16]) );
        REQUIRE( solver.getParValue(2, 0) == approx(fix_d[1]) );
        REQUIRE( solver.getParValue(0, 1) == approx(fix_d[17]) );
        REQUIRE( solver.getParValue(2, 1) == approx(fix_d[5]) );
    }
    SECTION( "Active: bgr-0, I0-1, bgr-1, tau" ) {
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(15974.61260816282) );
        REQUIRE( solver.getParValue(1) == approx(20.47926391663428) );
        REQUIRE( solver.getParValue(0, 0) == approx(fix_d[0]) );
        REQUIRE( solver.getParValue(2, 0) == approx(18.47600900933105) );
        REQUIRE( solver.getParValue(0, 1) == approx(143.0431252627764) );
        REQUIRE( solver.getParValue(2, 1) == approx(9.453915929181857) );
    }
    SECTION( "Active: I0-0, bgr-0, bgr-1, tau" ) {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], false, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(145780.4588072045) );
        REQUIRE( solver.getParValue(1) == approx(8.408237957600139) );
        REQUIRE( solver.getParValue(0, 0) == approx(45.87087327322399) );
        REQUIRE( solver.getParValue(2, 0) == approx(16.59126759913267) );
        REQUIRE( solver.getParValue(0, 1) == approx(fix_d[4]) );
        REQUIRE( solver.getParValue(2, 1) == approx(36.38255403506549) );
    }
    SECTION( "Active: I0-0, I0-1, bgr-1, tau" ) {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(11623.17388899667) );
        REQUIRE( solver.getParValue(1) == approx(20.61333132315124) );
        REQUIRE( solver.getParValue(0, 0) == approx(56.51395760213281) );
        REQUIRE( solver.getParValue(2, 0) == approx(fix_d[1]) );
        REQUIRE( solver.getParValue(0, 1) == approx(134.8973104943701) );
        REQUIRE( solver.getParValue(2, 1) == approx(11.77612256514583) );
    }
    SECTION( "Active: I0-0, bgr-0, I0-1, tau" ) {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(30610.67204238355) );
        REQUIRE( solver.getParValue(1) == approx(16.54682323514368) );
        REQUIRE( solver.getParValue(0, 0) == approx(29.98632400541694) );
        REQUIRE( solver.getParValue(2, 0) == approx(12.99477135618182) );
        REQUIRE( solver.getParValue(0, 1) == approx(124.6991105597199) );
        REQUIRE( solver.getParValue(2, 1) == approx(fix_d[5]) );
    }
    SECTION( "Active: I0-0, bgr-1, tau" ) {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[4], false, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(150672.9869101835) );
        REQUIRE( solver.getParValue(1) == approx(16.73368044360274) );
        REQUIRE( solver.getParValue(0, 0) == approx(53.73848940201644) );
        REQUIRE( solver.getParValue(2, 0) == approx(fix_d[1]) );
        REQUIRE( solver.getParValue(0, 1) == approx(fix_d[4]) );
        REQUIRE( solver.getParValue(2, 1) == approx(36.50405720192947) );
    }
    SECTION( "Active: bgr-0, I0-1, tau" ) {
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(15348.60122706107) );
        REQUIRE( solver.getParValue(1) == approx(21.87456778662339) );
        REQUIRE( solver.getParValue(0, 0) == approx(fix_d[0]) );
        REQUIRE( solver.getParValue(2, 0) == approx(18.39176693290169) );
        REQUIRE( solver.getParValue(0, 1) == approx(147.1783948678938) );
        REQUIRE( solver.getParValue(2, 1) == approx(fix_d[5]) );
    }
    SECTION( "No active parameters" ) {
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[4], false, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], false);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(284681.4650859555) );
        REQUIRE( solver.getParValue(1) == approx(0.5356792380861322) );
        REQUIRE( solver.getParValue(0, 0) == approx(6.13604207015635) );
        REQUIRE( solver.getParValue(2, 0) == approx(2.960644474827888) );
        REQUIRE( solver.getParValue(0, 1) == approx(-1.472720596147903) );
        REQUIRE( solver.getParValue(2, 1) == approx(4.251266120087788) );
    }
}

TEST_CASE( "Access functions" )
{
    spdlog::set_level(spdlog::level::off);
    gadfit::LMsolver solver { exponential, MPI_COMM_WORLD };
    solver.addDataset(
      std::make_shared<std::vector<double>>(x_data_1.begin(), x_data_1.end()),
      std::make_shared<std::vector<double>>(y_data_1.begin(), y_data_1.end()),
      std::make_shared<std::vector<double>>(x_data_1.size(), 1.0));
    solver.addDataset(
      std::make_shared<std::vector<double>>(x_data_2.begin(), x_data_2.end()),
      std::make_shared<std::vector<double>>(y_data_2.begin(), y_data_2.end()));
    solver.settings.iteration_limit = 4;
    solver.setPar(0, fix_d[0], true, 0);
    solver.setPar(2, fix_d[1], true, 0);
    solver.setPar(0, fix_d[4], true, 1);
    solver.setPar(2, fix_d[5], true, 1);
    solver.setPar(1, fix_d[3], true);
    REQUIRE( solver.getValue(fix_d[2]) == approx(2.960644529912441) );
    solver.fit(1.0);
    REQUIRE( solver.getValue(fix_d[2]) == approx(36.39905496310918) );
    const auto sum { [](const std::vector<double>& x, const int limit = -1) {
        return std::accumulate(
          x.cbegin(), limit == -1 ? x.cend() : x.cbegin() + limit, 0.0);
    } };
    REQUIRE( sum(solver.getJacobian()) == approx(353.6485673748525) );
    REQUIRE( sum(solver.getJTJ(), 5) == approx(580.3488115472478) );
    REQUIRE( sum(solver.getDTD(), 5) == approx(34340.67196549198) );
    REQUIRE( sum(solver.getLeftSide(), 5) == approx(614.6894835127398) );
    REQUIRE( sum(solver.getRightSide(), 5) == approx(4410.585412402702) );
    REQUIRE( sum(solver.getResiduals()) == approx(213.3530475167955) );
}

TEST_CASE( "Exceptions" )
{
    spdlog::set_level(spdlog::level::off);
    gadfit::LMsolver solver { exponential, MPI_COMM_WORLD };
    solver.settings.iteration_limit = 4;

    SECTION( "Incorrect number of calls" ) {
        try {
            solver.addDataset(x_data_1, y_data_1);
            solver.setPar(0, fix_d[0], true, 0);
            solver.addDataset(x_data_2, y_data_2);
            solver.setPar(2, fix_d[1], true, 0);
            solver.setPar(0, fix_d[4], true, 1);
            solver.setPar(2, fix_d[5], true, 1);
            solver.setPar(1, fix_d[3], true);
            solver.fit(1.0);
            REQUIRE( 1 == 0 );
        } catch (const gadfit::LateAddDatasetCall& e) {
            e.what();
            REQUIRE( 0 == 0 );
        }
    }
    SECTION( "Invalid data set index" ) {
        try {
            solver.addDataset(x_data_1, y_data_1);
            solver.addDataset(x_data_2, y_data_2);
            solver.setPar(0, fix_d[0], true, 0);
            solver.setPar(2, fix_d[1], true, 0);
            solver.setPar(0, fix_d[4], true, 1);
            solver.setPar(2, fix_d[5], true, 1);
            solver.setPar(1, fix_d[3], true, 2);
            solver.fit(1.0);
            REQUIRE( 1 == 0 );
        } catch (const gadfit::SetParInvalidIndex& e) {
            e.what();
            REQUIRE( 0 == 0 );
        }
    }
    SECTION( "Uninitialized fitting parameter" ) {
        try {
            solver.addDataset(x_data_1, y_data_1);
            solver.addDataset(x_data_2, y_data_2);
            solver.setPar(0, fix_d[0], true, 0);
            solver.setPar(2, fix_d[1], true, 0);
            solver.setPar(0, fix_d[4], true, 1);
            solver.setPar(1, fix_d[3], true);
            solver.fit(1.0);
            REQUIRE( 1 == 0 );
        } catch (const gadfit::UninitializedParameter& e) {
            e.what();
            REQUIRE( 0 == 0 );
        }
    }
    SECTION( "Too few data points (or too many fitting parameters)" ) {
        try {
            solver.addDataset(2, x_data_1.data(), y_data_1.data());
            solver.addDataset(2, x_data_2.data(), y_data_2.data());
            solver.setPar(0, fix_d[0], true, 0);
            solver.setPar(2, fix_d[1], true, 0);
            solver.setPar(0, fix_d[4], true, 1);
            solver.setPar(2, fix_d[5], true, 1);
            solver.setPar(1, fix_d[3], true);
            solver.fit(1.0);
            REQUIRE( 1 == 0 );
        } catch (const gadfit::NegativeDegreesOfFreedom& e) {
            e.what();
            REQUIRE( 0 == 0 );
        }
    }
    SECTION( "No degrees of freedom" ) {
        solver.addDataset(3, x_data_1.data(), y_data_1.data());
        solver.addDataset(2, x_data_2.data(), y_data_2.data());
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        try {
            solver.fit(1.0);
            REQUIRE( solver.chi2() == approx(12.06947119410627, 1e3) );
            REQUIRE( solver.getParValue(1) == approx(2.945868346541738) );
            REQUIRE( solver.getParValue(0, 0) == approx(7.351966871429208) );
            REQUIRE( solver.getParValue(2, 0) == approx(49.68674387147235) );
            REQUIRE( solver.getParValue(0, 1)
                     == approx(-13.18731292934496, 1e3) );
            REQUIRE( solver.getParValue(2, 1) == approx(162.1781165060048) );
        } catch (const gadfit::UnusedMPIProcess& e) {
            e.what();
        }
    }
    SECTION( "Active: bgr-0, I0-1" ) {
        solver.addDataset(x_data_1, y_data_1);
        solver.addDataset(x_data_2, y_data_2);
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[12], false);
        try {
            solver.fit(1.0);
            REQUIRE( 1 == 0 );
        } catch (const gadfit::NoGlobalParameters& e) {
            e.what();
            REQUIRE( 0 == 0 );
        }
    }
}

TEST_CASE( "Number of iterations" )
{
    spdlog::set_level(spdlog::level::off);
    gadfit::LMsolver solver { exponential, MPI_COMM_WORLD };
    solver.addDataset(x_data_1, y_data_1);
    solver.addDataset(x_data_2, y_data_2);
    solver.setPar(0, fix_d[0], true, 0);
    solver.setPar(2, fix_d[1], true, 0);
    solver.setPar(0, fix_d[4], true, 1);
    solver.setPar(2, fix_d[5], true, 1);
    solver.setPar(1, fix_d[3], true);

    SECTION( "No iterations" ) {
        solver.settings.iteration_limit = 0;
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(284681.4650859555) );
        REQUIRE( solver.getParValue(1) == approx(0.5356792380861322) );
        REQUIRE( solver.getParValue(0, 0) == approx(6.13604207015635) );
        REQUIRE( solver.getParValue(2, 0) == approx(2.960644474827888) );
        REQUIRE( solver.getParValue(0, 1) == approx(-1.472720596147903) );
        REQUIRE( solver.getParValue(2, 1) == approx(4.251266120087788) );
    }
    SECTION( "No iteration limit" ) {
        solver.settings.iteration_limit = 100;
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(5640.175130917763) );
        REQUIRE( solver.getParValue(1) == approx(20.85609539806552, 1e7) );
        REQUIRE( solver.getParValue(0, 0) == approx(46.44788540130083, 1e6) );
        REQUIRE( solver.getParValue(2, 0) == approx(10.32140443375086, 1e7) );
        REQUIRE( solver.getParValue(0, 1) == approx(152.2711588120757, 1e6) );
        REQUIRE( solver.getParValue(2, 1) == approx(5.533936910924339, 1e8) );
    }
}

TEST_CASE( "Constraining the damping matrix DTD" )
{
    spdlog::set_level(spdlog::level::off);
    gadfit::LMsolver solver { exponential, MPI_COMM_WORLD };
    solver.addDataset(x_data_1, y_data_1);
    solver.addDataset(x_data_2, y_data_2);
    solver.setPar(0, fix_d[0], true, 0);
    solver.setPar(2, fix_d[1], true, 0);
    solver.setPar(0, fix_d[4], true, 1);
    solver.setPar(2, fix_d[5], true, 1);
    solver.setPar(1, fix_d[3], true);
    solver.settings.iteration_limit = 5;

    SECTION( "No constraints" ) {
        solver.settings.damp_max = false;
        // Next line should have no effect
        solver.settings.DTD_min = { 2.0, 1.0, 3.0, 1.0, 7.0 };
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(5761.320550200893) );
        REQUIRE( solver.getParValue(1) == approx(20.15808767822607) );
        REQUIRE( solver.getParValue(0, 0) == approx(46.33952661438988) );
        REQUIRE( solver.getParValue(2, 0) == approx(10.4069642674494) );
        REQUIRE( solver.getParValue(0, 1) == approx(151.8449704289686) );
        REQUIRE( solver.getParValue(2, 1) == approx(5.793008018375243) );
    }
    SECTION( "Default constraint" ) {
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(5687.451130305411) );
        REQUIRE( solver.getParValue(1) == approx(21.01892108898218) );
        REQUIRE( solver.getParValue(0, 0) == approx(46.18357253310398) );
        REQUIRE( solver.getParValue(2, 0) == approx(10.48386354002993) );
        REQUIRE( solver.getParValue(0, 1) == approx(151.5283959798011) );
        REQUIRE( solver.getParValue(2, 1) == approx(6.087406702661866) );
    }
    SECTION( "Minimal values for the diagonal elements" ) {
        solver.settings.DTD_min = { 2.0, 1.0, 3.0, 1.0, 7.0 };
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(5640.443443547636) );
        REQUIRE( solver.getParValue(1) == approx(20.81941350480559) );
        REQUIRE( solver.getParValue(0, 0) == approx(46.44560357121117) );
        REQUIRE( solver.getParValue(2, 0) == approx(10.33411834275264) );
        REQUIRE( solver.getParValue(0, 1) == approx(152.2267670731584) );
        REQUIRE( solver.getParValue(2, 1) == approx(5.56046342286098) );
    }
}

TEST_CASE( "Geodesic acceleration" )
{
    spdlog::set_level(spdlog::level::off);
    gadfit::LMsolver solver { exponential, MPI_COMM_WORLD };
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
    REQUIRE( solver.chi2() == approx(5641.660305504621) );
    REQUIRE( solver.getParValue(1) == approx(20.70654799943915) );
    REQUIRE( solver.getParValue(0, 0) == approx(46.48065799723028) );
    REQUIRE( solver.getParValue(2, 0) == approx(10.39142422387267) );
    REQUIRE( solver.getParValue(0, 1) == approx(152.4514268293043) );
    REQUIRE( solver.getParValue(2, 1) == approx(5.748941149916492) );
}

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
    gadfit::LMsolver solver { exponential };
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
        REQUIRE( solver.chi2() == approx(13419.2324476273) );
        REQUIRE( solver.getParValue(1) == approx(22.884827826053) );
        REQUIRE( solver.getParValue(0, 0) == approx(37.25455194786) );
        REQUIRE( solver.getParValue(2, 0) == approx(12.9262098123123) );
        REQUIRE( solver.getParValue(0, 1) == approx(120.679337652733) );
        REQUIRE( solver.getParValue(2, 1) == approx(13.9172100518583) );
    }
    SECTION( "Active: bgr-0, bgr-1, tau" ) {
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], false, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(153626.6200123923) );
        REQUIRE( solver.getParValue(1) == approx(28.0465962138761) );
        REQUIRE( solver.getParValue(0, 0) == approx(fix_d[0]) );
        REQUIRE( solver.getParValue(2, 0) == approx(18.0670378013166) );
        REQUIRE( solver.getParValue(0, 1) == approx(fix_d[4]) );
        REQUIRE( solver.getParValue(2, 1) == approx(36.6704549176379) );
    }
    SECTION( "Active: I0-0, I0-1, tau" ) {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(41226.1757537588) );
        REQUIRE( solver.getParValue(1) == approx(37.4716876348651) );
        REQUIRE( solver.getParValue(0, 0) == approx(61.8928662074328) );
        REQUIRE( solver.getParValue(2, 0) == approx(fix_d[1]) );
        REQUIRE( solver.getParValue(0, 1) == approx(138.921164968188) );
        REQUIRE( solver.getParValue(2, 1) == approx(fix_d[5]) );
    }
    SECTION( "Active: I0-0, bgr-0, I0-1, bgr-1" ) {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], false);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(133479.8243379537) );
        REQUIRE( solver.getParValue(1) == approx(fix_d[3]) );
        REQUIRE( solver.getParValue(0, 0) == approx(269.304141722311) );
        REQUIRE( solver.getParValue(2, 0) == approx(19.2074379001413) );
        REQUIRE( solver.getParValue(0, 1) == approx(884.063475394066) );
        REQUIRE( solver.getParValue(2, 1) == approx(34.6630429778281) );
    }
    SECTION( "Active: tau" ) {
        solver.setPar(0, fix_d[16], false, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[17], false, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(49961.1602751974) );
        REQUIRE( solver.getParValue(1) == approx(11.2589241454522) );
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
        REQUIRE( solver.chi2() == approx(16946.5517452934) );
        REQUIRE( solver.getParValue(1) == approx(23.5087805124599) );
        REQUIRE( solver.getParValue(0, 0) == approx(fix_d[0]) );
        REQUIRE( solver.getParValue(2, 0) == approx(18.2926838650648) );
        REQUIRE( solver.getParValue(0, 1) == approx(143.247753089907) );
        REQUIRE( solver.getParValue(2, 1) == approx(7.41966861364184) );
    }
    SECTION( "Active: I0-0, bgr-0, bgr-1, tau" ) {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], false, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(145969.6460672426) );
        REQUIRE( solver.getParValue(1) == approx(9.6493001540062) );
        REQUIRE( solver.getParValue(0, 0) == approx(38.8125594831746) );
        REQUIRE( solver.getParValue(2, 0) == approx(16.3892170419466) );
        REQUIRE( solver.getParValue(0, 1) == approx(fix_d[4]) );
        REQUIRE( solver.getParValue(2, 1) == approx(36.4005406565312) );
    }
    SECTION( "Active: I0-0, I0-1, bgr-1, tau" ) {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(20328.9274178601) );
        REQUIRE( solver.getParValue(1) == approx(29.8006091499258) );
        REQUIRE( solver.getParValue(0, 0) == approx(58.7510164156725) );
        REQUIRE( solver.getParValue(2, 0) == approx(fix_d[1]) );
        REQUIRE( solver.getParValue(0, 1) == approx(134.596382423157) );
        REQUIRE( solver.getParValue(2, 1) == approx(8.18765519687475) );
    }
    SECTION( "Active: I0-0, bgr-0, I0-1, tau" ) {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(19110.8917974984) );
        REQUIRE( solver.getParValue(1) == approx(31.644340100719) );
        REQUIRE( solver.getParValue(0, 0) == approx(26.5687296086344) );
        REQUIRE( solver.getParValue(2, 0) == approx(10.314778623254) );
        REQUIRE( solver.getParValue(0, 1) == approx(136.057802762212) );
        REQUIRE( solver.getParValue(2, 1) == approx(fix_d[5]) );
    }
    SECTION( "Active: I0-0, bgr-1, tau" ) {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[4], false, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(152198.4612377039) );
        REQUIRE( solver.getParValue(1) == approx(20.647406018572) );
        REQUIRE( solver.getParValue(0, 0) == approx(39.5446097854515) );
        REQUIRE( solver.getParValue(2, 0) == approx(fix_d[1]) );
        REQUIRE( solver.getParValue(0, 1) == approx(fix_d[4]) );
        REQUIRE( solver.getParValue(2, 1) == approx(36.5609695432337) );
    }
    SECTION( "Active: bgr-0, I0-1, tau" ) {
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(17084.6857044287) );
        REQUIRE( solver.getParValue(1) == approx(25.2416940612762) );
        REQUIRE( solver.getParValue(0, 0) == approx(fix_d[0]) );
        REQUIRE( solver.getParValue(2, 0) == approx(18.1899933794028) );
        REQUIRE( solver.getParValue(0, 1) == approx(146.208847896439) );
        REQUIRE( solver.getParValue(2, 1) == approx(fix_d[5]) );
    }
    SECTION( "Active: bgr-0, I0-1" ) {
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[12], false);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(80628.4089204265) );
        REQUIRE( solver.getParValue(1) == approx(6.76382444846188) );
        REQUIRE( solver.getParValue(0, 0) == approx(fix_d[0]) );
        REQUIRE( solver.getParValue(2, 0) == approx(19.3148861774975) );
        REQUIRE( solver.getParValue(0, 1) == approx(244.850164021374) );
        REQUIRE( solver.getParValue(2, 1) == approx(fix_d[5]) );
    }
}

TEST_CASE( "Access functions" )
{
    spdlog::set_level(spdlog::level::off);
    gadfit::LMsolver solver { exponential };
    solver.addDataset(x_data_1, y_data_1);
    solver.addDataset(x_data_2, y_data_2);
    solver.settings.iteration_limit = 4;
    solver.setPar(0, fix_d[0], true, 0);
    solver.setPar(2, fix_d[1], true, 0);
    solver.setPar(0, fix_d[4], true, 1);
    solver.setPar(2, fix_d[5], true, 1);
    solver.setPar(1, fix_d[3], true);
    REQUIRE( solver.getValue(fix_d[2]) == approx(2.960644529912441) );
    solver.fit(1.0);
    REQUIRE( solver.getValue(fix_d[2]) == approx(37.07089678775328) );
    {
        double sum {};
        for (const auto& el : solver.getJacobian()) {
            sum += el;
        }
        REQUIRE( sum == approx(331.8214061379837) );
    }
    {
        double sum {};
        for (const auto& el : solver.getJTJ()) {
            sum += el;
        }
        REQUIRE( sum == approx(786.5700259460133) );
    }
    {
        double sum {};
        for (const auto& el : solver.getDTD()) {
            sum += el;
        }
        REQUIRE( sum == approx(465.1656068124079) );
    }
    {
        double sum {};
        for (const auto& el : solver.getLeftSide()) {
            sum += el;
        }
         REQUIRE( sum == approx(791.2216820141374) );
    }
    {
        double sum {};
        for (const auto& el : solver.getRightSide()) {
            sum += el;
        }
        REQUIRE( sum == approx(4096.69145240406) );
    }
    {
        double sum {};
        for (const auto& el : solver.getResiduals()) {
            sum += el;
        }
        REQUIRE( sum == approx(353.1929743320203) );
    }
}

TEST_CASE( "Exceptions" )
{
    spdlog::set_level(spdlog::level::off);
    gadfit::LMsolver solver { exponential };
    solver.settings.iteration_limit = 4;

    SECTION( "Incorrect or of calls" ) {
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
    const auto reduceVector { [](const auto& in, const int N) {
        return std::vector<double>(in.cbegin(), in.cbegin() + N);
    } };
    SECTION( "Too few data points (or too many fitting parameters)" ) {
        try {

            solver.addDataset(
              reduceVector(x_data_1, 2), reduceVector(y_data_1, 2));
            solver.addDataset(
              reduceVector(x_data_2, 2), reduceVector(y_data_2, 2));
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
        solver.addDataset(reduceVector(x_data_1, 3), reduceVector(y_data_1, 3));
        solver.addDataset(reduceVector(x_data_2, 2), reduceVector(y_data_2, 2));
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(409.5387799144992) );
        REQUIRE( solver.getParValue(1) == approx(4.224317491642794) );
        REQUIRE( solver.getParValue(0, 0) == approx(24.67898685099651) );
        REQUIRE( solver.getParValue(2, 0) == approx(35.84110743027947) );
        REQUIRE( solver.getParValue(0, 1) == approx(108.2684165067531) );
        REQUIRE( solver.getParValue(2, 1) == approx(70.48326918227681) );
    }
}

TEST_CASE( "Other" )
{
    spdlog::set_level(spdlog::level::off);
    gadfit::LMsolver solver { exponential };

    SECTION( "No iterations" ) {
        solver.addDataset(x_data_1, y_data_1);
        solver.addDataset(x_data_2, y_data_2);
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.settings.iteration_limit = 0;
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(284681.4650859555) );
        REQUIRE( solver.getParValue(1) == approx(0.5356792380861322) );
        REQUIRE( solver.getParValue(0, 0) == approx(6.13604207015635) );
        REQUIRE( solver.getParValue(2, 0) == approx(2.960644474827888) );
        REQUIRE( solver.getParValue(0, 1) == approx(-1.472720596147903) );
        REQUIRE( solver.getParValue(2, 1) == approx(4.251266120087788) );
    }
    SECTION( "No active parameters" ) {
        solver.addDataset(x_data_1, y_data_1);
        solver.addDataset(x_data_2, y_data_2);
        solver.setPar(0, fix_d[0], false, 0);
        solver.setPar(2, fix_d[1], false, 0);
        solver.setPar(0, fix_d[4], false, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], false);
        spdlog::set_level(spdlog::level::info);
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(284681.4650859555) );
        REQUIRE( solver.getParValue(1) == approx(0.5356792380861322) );
        REQUIRE( solver.getParValue(0, 0) == approx(6.13604207015635) );
        REQUIRE( solver.getParValue(2, 0) == approx(2.960644474827888) );
        REQUIRE( solver.getParValue(0, 1) == approx(-1.472720596147903) );
        REQUIRE( solver.getParValue(2, 1) == approx(4.251266120087788) );
    }
    SECTION( "No iteration limit" ) {
        solver.addDataset(x_data_1, y_data_1);
        solver.addDataset(x_data_2, y_data_2);
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], true, 1);
        solver.setPar(1, fix_d[3], true);
        solver.settings.iteration_limit = 100;
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(5640.175130917764) );
        REQUIRE( solver.getParValue(1) == approx(20.85609539369426) );
        REQUIRE( solver.getParValue(0, 0) == approx(46.44788540483412) );
        REQUIRE( solver.getParValue(2, 0) == approx(10.32140443497198) );
        REQUIRE( solver.getParValue(0, 1) == approx(152.2711588181074) );
        REQUIRE( solver.getParValue(2, 1) == approx(5.533936916048498) );
    }
}

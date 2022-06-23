#include "testing.h"

#include "lm_solver_data.h"

#include <gadfit/lm_solver.h>
#include <spdlog/spdlog.h>

// If compiled with Scalapack, its functionality is already
// extensively tested by other test cases. Here we test a few cases
// relevant to Scalapack only.

static auto exponential(const std::vector<gadfit::AdVar>& parameters,
                        const double x) -> gadfit::AdVar
{
    using gadfit::AdVar;
    const AdVar& I0 { parameters[0] };
    const AdVar& tau { parameters[1] };
    const AdVar& bgr { parameters[2] };
    return I0 * exp(-x / tau) + bgr;
}

TEST_CASE( "Memory layouts" )
{
    spdlog::set_level(spdlog::level::off);
    gadfit::LMsolver solver { exponential, MPI_COMM_WORLD };
    solver.addDataset(x_data_1, y_data_1);
    solver.addDataset(x_data_2, y_data_2);
    solver.settings.iteration_limit = 4;
    SECTION( "General rectangular BLACS grid; large blocks" ) {
        solver.setPar(0, fix_d[0], true, 0);
        solver.setPar(2, fix_d[1], true, 0);
        solver.setPar(0, fix_d[4], true, 1);
        solver.setPar(2, fix_d[5], false, 1);
        solver.setPar(1, fix_d[3], true);
        solver.settings.blacs_single_column = false;
        solver.settings.min_n_blocks = 1;
        solver.fit(1.0);
        REQUIRE( solver.chi2() == approx(30610.67204238355) );
        REQUIRE( solver.getParValue(1) == approx(16.54682323514368) );
        REQUIRE( solver.getParValue(0, 0) == approx(29.98632400541694) );
        REQUIRE( solver.getParValue(2, 0) == approx(12.99477135618182) );
        REQUIRE( solver.getParValue(0, 1) == approx(124.6991105597199) );
        REQUIRE( solver.getParValue(2, 1) == approx(fix_d[5]) );
    }
}

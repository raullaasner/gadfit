#include "testing.h"

#include "mpi_basic_data.h"

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

// Helper function for the case of three data sets
static auto setSolverState(gadfit::LMsolver& solver) -> void {
    solver.addDataset(x_data_1, y_data_1);
    solver.addDataset(x_data_3, y_data_3);
    solver.addDataset(97, x_data_2.data(), y_data_2.data());
    solver.setPar(0, fix_d[0], true, 0);
    solver.setPar(2, fix_d[2], true, 0);
    solver.setPar(0, fix_d[17], true, 1);
    solver.setPar(2, fix_d[5], true, 1);
    solver.setPar(0, fix_d[16], true, 2);
    solver.setPar(2, fix_d[12], true, 2);
    solver.setPar(1, fix_d[1], true);
}

TEST_CASE( "Basic tests" )
{
    static constexpr double ref_chi2 { 13083.80579217767 };
    static constexpr double ref_1    { 20.78511330961546 };
    static constexpr double ref_0_0  { 46.47983491330410 };
    static constexpr double ref_2_0  { 10.34639934873397 };
    static constexpr double ref_0_1  { 217.5845947842725 };
    static constexpr double ref_2_1  { 18.03661200603774 };
    static constexpr double ref_0_2  { 152.2129184602429 };
    static constexpr double ref_2_2  { 5.698662376973664 };

    spdlog::set_level(spdlog::level::off);

    int my_id {};
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    int num_procs {};
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm mpi_sub_comm {};
    const int color { my_id < num_procs / 2 ? 0 : 1 };
    MPI_Comm_split(MPI_COMM_WORLD, color, my_id, &mpi_sub_comm);

    SECTION( "Different communicators" ) {
        // Whether it's serial, parallel, or with sub-communicators,
        // we should always get the same results.
        gadfit::LMsolver solver_1 { exponential };
        setSolverState(solver_1);
        solver_1.settings.iteration_limit = 8;
        solver_1.fit();
        CHECK( solver_1.chi2() == approx(ref_chi2) );
        CHECK( solver_1.getParValue(1) == approx(ref_1) );
        CHECK( solver_1.getParValue(0, 0) == approx(ref_0_0) );
        CHECK( solver_1.getParValue(2, 0) == approx(ref_2_0) );
        CHECK( solver_1.getParValue(0, 1) == approx(ref_0_1) );
        CHECK( solver_1.getParValue(2, 1) == approx(ref_2_1) );
        CHECK( solver_1.getParValue(0, 2) == approx(ref_0_2) );
        CHECK( solver_1.getParValue(2, 2) == approx(ref_2_2) );
        gadfit::LMsolver solver_2 { exponential, MPI_COMM_WORLD };
        setSolverState(solver_2);
        solver_2.settings.iteration_limit = 8;
        solver_2.fit();
        CHECK( solver_2.chi2() == approx(ref_chi2) );
        CHECK( solver_2.getParValue(1) == approx(ref_1) );
        CHECK( solver_2.getParValue(0, 0) == approx(ref_0_0) );
        CHECK( solver_2.getParValue(2, 0) == approx(ref_2_0) );
        CHECK( solver_2.getParValue(0, 1) == approx(ref_0_1) );
        CHECK( solver_2.getParValue(2, 1) == approx(ref_2_1) );
        CHECK( solver_2.getParValue(0, 2) == approx(ref_0_2) );
        CHECK( solver_2.getParValue(2, 2) == approx(ref_2_2) );
        gadfit::LMsolver solver_3 { exponential, mpi_sub_comm };
        setSolverState(solver_3);
        solver_3.settings.iteration_limit = 8;
        solver_3.fit();
        CHECK( solver_3.chi2() == approx(ref_chi2) );
        CHECK( solver_3.getParValue(1) == approx(ref_1) );
        CHECK( solver_3.getParValue(0, 0) == approx(ref_0_0) );
        CHECK( solver_3.getParValue(2, 0) == approx(ref_2_0) );
        CHECK( solver_3.getParValue(0, 1) == approx(ref_0_1) );
        CHECK( solver_3.getParValue(2, 1) == approx(ref_2_1) );
        CHECK( solver_3.getParValue(0, 2) == approx(ref_0_2) );
        CHECK( solver_3.getParValue(2, 2) == approx(ref_2_2) );
    }

    SECTION( "Split communicator" ) {
        gadfit::LMsolver solver_1 { exponential, mpi_sub_comm };
        setSolverState(solver_1);
        gadfit::LMsolver solver_2 { exponential, mpi_sub_comm };
        solver_2.addDataset(x_data_1, y_data_1);
        solver_2.addDataset(x_data_3, y_data_3);
        solver_2.setPar(0, fix_d[0], true, 0);
        solver_2.setPar(2, fix_d[2], true, 0);
        solver_2.setPar(0, fix_d[17], true, 1);
        solver_2.setPar(2, fix_d[5], true, 1);
        solver_2.setPar(1, fix_d[1], true);
        if (color == 0) {
            solver_1.settings.iteration_limit = 8;
            solver_1.fit();
        } else {
            solver_2.settings.iteration_limit = 8;
            solver_2.fit(1.0);
        }
        if (color == 0) {
            // First group of MPI processes performed the fitting
            // procedure with solver_1 whereas solver_2 retained its
            // initial state. Now check that the first solver yields
            // the converged parameters and the second solver outputs
            // the initial values.
            CHECK( solver_1.chi2() == approx(ref_chi2) );
            CHECK( solver_1.getParValue(1) == approx(20.78511330961546) );
            CHECK( solver_1.getParValue(0, 0) == approx(46.4798349133041) );
            CHECK( solver_1.getParValue(2, 0) == approx(10.34639934873397) );
            CHECK( solver_1.getParValue(0, 1) == approx(217.5845947842725) );
            CHECK( solver_1.getParValue(2, 1) == approx(18.03661200603774) );
            CHECK( solver_1.getParValue(0, 2) == approx(152.2129184602429) );
            CHECK( solver_1.getParValue(2, 2) == approx(5.698662376973664) );
            CHECK( solver_2.chi2() == approx(445170.488068111) );
            CHECK( solver_2.getParValue(1) == approx(fix_d[1]) );
            CHECK( solver_2.getParValue(0, 0) == approx(fix_d[0]) );
            CHECK( solver_2.getParValue(2, 0) == approx(fix_d[2]) );
            CHECK( solver_2.getParValue(0, 1) == approx(fix_d[17]) );
            CHECK( solver_2.getParValue(2, 1) == approx(fix_d[5]) );
        } else {
            // Same as above but vice versa
            CHECK( solver_1.chi2() == approx(651849.451623239) );
            CHECK( solver_1.getParValue(1) == approx(fix_d[1]) );
            CHECK( solver_1.getParValue(0, 0) == approx(fix_d[0]) );
            CHECK( solver_1.getParValue(2, 0) == approx(fix_d[2]) );
            CHECK( solver_1.getParValue(0, 1) == approx(fix_d[17]) );
            CHECK( solver_1.getParValue(2, 1) == approx(fix_d[5]) );
            CHECK( solver_1.getParValue(0, 2) == approx(fix_d[16]) );
            CHECK( solver_1.getParValue(2, 2) == approx(fix_d[12]) );
            CHECK( solver_2.chi2() == approx(9258.726275249157) );
            CHECK( solver_2.getParValue(1) == approx(20.80254628279121) );
            CHECK( solver_2.getParValue(0, 0) == approx(46.4719634330144) );
            CHECK( solver_2.getParValue(2, 0) == approx(10.34025972482301) );
            CHECK( solver_2.getParValue(0, 1) == approx(217.5480332306478) );
            CHECK( solver_2.getParValue(2, 1) == approx(18.00867601264374) );
        }
    }
}

TEST_CASE( "Exceptions" )
{
    MPI_Finalize();
    try {
        gadfit::LMsolver solver { exponential, MPI_COMM_WORLD };
        REQUIRE( 1 == 0 );
    } catch (const gadfit::MPIUninitialized& e) {
        e.what();
        REQUIRE( 0 == 0 );
    }
}

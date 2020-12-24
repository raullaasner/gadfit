#ifdef USE_MPI

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    int mpi_finalized {};
    MPI_Finalized(&mpi_finalized);
    if (!static_cast<bool>(mpi_finalized)) {
        MPI_Finalize();
    }
    return result;
}

#else

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

TEST_CASE( "All test cases reside in other .cpp files (empty)" ,
           "[multi-file:1]")
{
}

#endif // USE_MPI

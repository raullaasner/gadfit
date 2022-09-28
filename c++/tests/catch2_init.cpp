#include <catch2/catch_session.hpp>

#ifdef USE_MPI

#include <mpi.h>

int main(int argc, char* argv[])
{
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

int main(int argc, char* argv[])
{
    int result = Catch::Session().run(argc, argv);
    return result;
}

#endif // USE_MPI

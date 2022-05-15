#include <mpi.h>

extern "C"
{
    auto pdsyr2k_(const char* uplo,
                  const char* trans,
                  const int* n,
                  const int* k,
                  const double* alpha,
                  double* a,
                  const int* ia,
                  const int* ja,
                  const int* desca,
                  double* b,
                  const int* ib,
                  const int* jb,
                  const int* descb,
                  const double* beta,
                  double* c,
                  const int* ic,
                  const int* jc,
                  const int* descc) -> void;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    constexpr char c {};
    constexpr int n {};
    double d {};
    pdsyr2k_(
      &c, &c, &n, &n, &d, &d, &n, &n, &n, &d, &n, &n, &n, &d, &d, &n, &n, &n);
    return 0;
}

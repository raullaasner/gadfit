#include <mkl_cblas.h>
#include <mkl_lapack.h>

int main()
{
    static constexpr char uplo { 'l' };
    static constexpr int* N {};
    static constexpr double* A {};
    dpptrs(&uplo, N, N, A, A, N, N);
}

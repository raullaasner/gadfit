#include <cblas.h>

extern "C" {
    auto dpptrs_(const char*,
                const int*,
                const int*,
                double*,
                double*,
                const int*,
                int*) -> void;
}

int main()
{
    static constexpr char uplo { 'l' };
    static constexpr int* N {};
    static constexpr double* A {};
    dpptrs_(&uplo, N, N, A, A, N, N);
}

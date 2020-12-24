// Licensed under the Apache License, Version 2.0 (the "License"); you
// may not use this file except in compliance with the License.  You
// may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied.  See the License for the specific language governing
// permissions and limitations under the License.

#include "parallel.h"

#include <cstring>

#ifndef USE_MPI

auto MPI_Comm_rank(MPI_Comm, int*) -> void {}

auto MPI_Comm_size(MPI_Comm, int*) -> void {}

auto MPI_Initialized(int*) -> void {}

// This should never be called
static auto MPI_Alltoallv(const double* const,
                          const int* const,
                          const int* const,
                          int,
                          double* const,
                          const int* const,
                          const int* const,
                          int,
                          const MPI_Comm)
{}

#endif // USE_MPI

auto fragmentToGlobal(const double* const fragment,
                      const int* const sendcounts,
                      const int* const sdispls,
                      double* const result,
                      const int* const recvcounts,
                      const int* const rdispls,
                      MPI_Comm mpi_comm) -> void
{
    if (mpi_comm == MPI_COMM_NULL) {
        memcpy(result, fragment, *sendcounts * sizeof(double));
    } else {
        MPI_Alltoallv(fragment,
                      sendcounts,
                      sdispls,
                      MPI_DOUBLE,
                      result,
                      recvcounts,
                      rdispls,
                      MPI_DOUBLE,
                      mpi_comm);
    }
}

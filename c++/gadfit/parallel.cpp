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

auto MPI_Finalized(int*) -> void {}

auto MPI_Allreduce(int, double*, int, int, int, MPI_Comm) -> void {}

auto MPI_Comm_split(MPI_Comm, int, int, MPI_Comm*) -> void {}

auto MPI_Recv(void*, int, int, int, int, MPI_Comm, int) -> void {}

auto MPI_Send(const void*, int, int, int, int, MPI_Comm) -> void {}

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

auto gatherTimes(const MPI_Comm mpi_comm,
                 const int my_rank,
                 const int num_procs,
                 const gadfit::Timer& timer,
                 std::vector<gadfit::Times>& times) -> void
{
    if (my_rank == 0) {
        times.front().wall = timer.totalWallTime();
        times.front().cpu = timer.totalCPUTime();
        if (timer.totalNumberOfCalls() > 0) {
            times.front().wall_ave =
              times.front().wall / timer.totalNumberOfCalls();
            times.front().cpu_ave =
              times.front().cpu / timer.totalNumberOfCalls();
        }
        // The number of calls has to be the same on all processes and
        // thus will not be synced.
        for (int i_proc { 1 }; i_proc < num_procs; ++i_proc) {
            MPI_Recv(&times.at(i_proc).wall,
                     1,
                     MPI_DOUBLE,
                     i_proc,
                     i_proc,
                     mpi_comm,
                     MPI_STATUS_IGNORE);
            MPI_Recv(&times.at(i_proc).cpu,
                     1,
                     MPI_DOUBLE,
                     i_proc,
                     i_proc,
                     mpi_comm,
                     MPI_STATUS_IGNORE);
            if (timer.totalNumberOfCalls() > 0) {
                times.at(i_proc).wall_ave =
                  times.at(i_proc).wall / timer.totalNumberOfCalls();
                times.at(i_proc).cpu_ave =
                  times.at(i_proc).cpu / timer.totalNumberOfCalls();
            }
        }
    } else {
        const double wall_time { timer.totalWallTime() };
        MPI_Send(&wall_time, 1, MPI_DOUBLE, 0, my_rank, mpi_comm);
        const double cpu_time { timer.totalCPUTime() };
        MPI_Send(&cpu_time, 1, MPI_DOUBLE, 0, my_rank, mpi_comm);
    }
}

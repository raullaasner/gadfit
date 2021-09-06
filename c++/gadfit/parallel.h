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

// Provides variables and functions that deal with parallelization
// only.

#pragma once

#include "timer.h"

#include <vector>

#ifdef USE_MPI

#include <mpi.h>

#else

using MPI_Comm = int;

constexpr MPI_Comm MPI_COMM_NULL {};
constexpr MPI_Comm MPI_COMM_WORLD { MPI_COMM_NULL };
constexpr int MPI_DOUBLE {};
constexpr int MPI_IN_PLACE {};
constexpr int MPI_SUM {};
constexpr int MPI_STATUS_IGNORE {};

auto MPI_Comm_rank(MPI_Comm, int*) -> void;
auto MPI_Comm_size(MPI_Comm, int*) -> void;
auto MPI_Initialized(int*) -> void;
auto MPI_Finalized(int*) -> void;
auto MPI_Allreduce(int, double*, int, int, int, MPI_Comm) -> void;
auto MPI_Comm_split(MPI_Comm, int, int, MPI_Comm*) -> void;
auto MPI_Recv(void*, int, int, int, int, MPI_Comm, int) -> void;
auto MPI_Send(const void*, int, int, int, int, MPI_Comm) -> void;

#endif // USE_MPI

// Each MPI process broadcasts their copy of a fragment into a global
// array. The fragments should not be overlapping but otherwise no
// assumptions are made about the memory layout.
auto fragmentToGlobal(const double* const fragment,
                      const int* const sendcounts,
                      const int* const sdispls,
                      double* const result,
                      const int* const recvcounts,
                      const int* const rdispls,
                      MPI_Comm mpi_comm) -> void;

// A utility function for LMsolver::printTimings. The first process
// collects timing data from all other processes.
auto gatherTimes(const MPI_Comm mpi_comm,
                 const int my_rank,
                 const int num_procs,
                 const gadfit::Timer& timer,
                 std::vector<gadfit::Times>& times) -> void;

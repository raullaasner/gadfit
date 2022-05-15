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

#include "gadfit_mpi.h"

#include <cstring>

#ifndef USE_MPI

auto MPI_Comm_rank(MPI_Comm, int*) -> void {}

auto MPI_Comm_size(MPI_Comm, int*) -> void {}

auto MPI_Initialized(int*) -> void {}

auto MPI_Finalized(int*) -> void {}

auto MPI_Comm_split(MPI_Comm, int, int, MPI_Comm*) -> void {}

auto MPI_Allreduce(int, double*, int, int, int, MPI_Comm) -> void {}

#endif // USE_MPI

namespace gadfit {

MPIVars::MPIVars(MPI_Comm comm) : comm { comm }
{
    if (comm == MPI_COMM_NULL) {
        return;
    }
    int mpi_finalized {};
    MPI_Finalized(&mpi_finalized);
    if (!static_cast<bool>(mpi_finalized)) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &n_procs);
    }
}

SharedArray::SharedArray(const MPIVars& mpi, const int n_values)
{
    if (mpi.comm == MPI_COMM_NULL) {
        storage_no_mpi.resize(n_values, 0.0);
        local = std::span { storage_no_mpi };
        global = std::span { storage_no_mpi };
        return;
    }
#ifdef USE_MPI
    double* local_ptr {};
    MPI_Win_allocate_shared(static_cast<MPI_Aint>(sizeof(double) * n_values),
                            sizeof(double),
                            MPI_INFO_NULL,
                            mpi.comm,
                            &local_ptr,
                            &win);
    local = std::span { local_ptr, static_cast<size_t>(n_values) };
    std::fill(local.begin(), local.end(), 0.0);
    MPI_Aint size {};
    int disp_unit {};
    double* global_ptr {};
    MPI_Win_shared_query(win, 0, &size, &disp_unit, &global_ptr);
    MPI_Win_shared_query(win, mpi.n_procs - 1, &size, &disp_unit, &local_ptr);
    global =
      std::span { global_ptr, local_ptr - global_ptr + size / sizeof(double) };
#endif
}

auto SharedArray::operator=(SharedArray&& shared_array) noexcept -> SharedArray&
{
    std::swap(win, shared_array.win);
    std::swap(storage_no_mpi, shared_array.storage_no_mpi);
    local = shared_array.local;
    global = shared_array.global;
    return *this;
}

[[nodiscard]] auto SharedArray::getWin() const -> const MPI_Win&
{
    return win;
}

SharedArray::~SharedArray()
{
    if (static_cast<bool>(win)) {
#ifdef USE_MPI
        MPI_Win_free(&win);
#endif
    }
}

auto gatherTimes(const MPIVars& mpi,
                 const Timer& timer,
                 std::vector<Times>& times) -> void
{
    std::vector<double> wall_times {};
    std::vector<double> cpu_times {};
    gatherScalar(mpi, timer.totalWallTime(), wall_times);
    gatherScalar(mpi, timer.totalCPUTime(), cpu_times);
    if (mpi.rank != 0) {
        return;
    }
    times.front().wall = wall_times.front();
    times.front().cpu = cpu_times.front();
    if (timer.totalNumberOfCalls() > 0) {
        times.front().wall_ave =
          times.front().wall / timer.totalNumberOfCalls();
        times.front().cpu_ave = times.front().cpu / timer.totalNumberOfCalls();
    }
    for (int i_proc {}; i_proc < mpi.n_procs; ++i_proc) {
        times.at(i_proc).wall = wall_times.at(i_proc);
        times.at(i_proc).cpu = cpu_times.at(i_proc);
        if (timer.totalNumberOfCalls() > 0) {
            times.at(i_proc).wall_ave =
              times.at(i_proc).wall / timer.totalNumberOfCalls();
            times.at(i_proc).cpu_ave =
              times.at(i_proc).cpu / timer.totalNumberOfCalls();
        }
    }
    // The number of calls has to be the same on all processes and
    // thus will not be synced.
}

} // namespace gadfit

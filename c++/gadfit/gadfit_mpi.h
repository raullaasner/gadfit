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

// Provides MPI related variables and functions. If not compiling with
// MPI, also defines MPI stubs.

#pragma once

#include "timer.h"

#include <span>
#include <vector>

#ifdef USE_MPI

#include <mpi.h>

#else

using MPI_Comm = int;
using MPI_Win = int;

constexpr MPI_Comm MPI_COMM_NULL {};
constexpr MPI_Comm MPI_COMM_WORLD { MPI_COMM_NULL };
constexpr int MPI_INT {};
constexpr int MPI_DOUBLE {};
constexpr int MPI_IN_PLACE {};
constexpr int MPI_SUM {};
constexpr int MPI_STATUS_IGNORE {};
constexpr int MPI_INFO_NULL {};

#endif // USE_MPI

namespace gadfit {

// A struct for basic MPI variables. The rank and number of processes
// are only set if comm is not null.
struct MPIVars
{
    MPI_Comm comm {};
    int rank {};
    int n_procs { 1 };
    MPIVars(MPI_Comm comm);
};

// Defines are shared array using the MPI RMA model
class SharedArray
{
private:
    MPI_Win win {};
    // Variable for storing data if no MPI is used
    std::vector<double> storage_no_mpi {};

public:
    // Points to the beginning of data of the current process
    std::span<double> local {};
    // Points to the beginning of data of the root process. This one
    // is typically used for the remote access.
    std::span<double> global {};
    SharedArray() = default;
    SharedArray(const SharedArray&) = delete;
    SharedArray(SharedArray&&) = delete;
    // Allocate a shared array with n_values elements of type double
    SharedArray(const MPIVars& mpi, const int n_values);
    // Need this explicitly, otherwise the MPI window will be
    // destroyed as shared_array goes out of scope
    auto operator=(SharedArray&& shared_array) noexcept -> SharedArray&;
    auto operator=(const SharedArray&) -> SharedArray& = delete;
    [[nodiscard]] auto getWin() const -> const MPI_Win&;
    ~SharedArray();
};

// Wrapper around MPI_Gather that deals with different types
template <typename T>
auto gatherScalar(const MPIVars& mpi, const T value, std::vector<T>& values)
  -> void
{
    if (mpi.rank == 0) {
        values.resize(mpi.n_procs);
        values.front() = value;
    }
#ifdef USE_MPI
    if constexpr (std::is_same_v<T, int>) {
        MPI_Gather(&value, 1, MPI_INT, values.data(), 1, MPI_INT, 0, mpi.comm);
    } else if constexpr (std::is_same_v<T, double>) {
        MPI_Gather(
          &value, 1, MPI_DOUBLE, values.data(), 1, MPI_DOUBLE, 0, mpi.comm);
    } else {
        throw std::domain_error { "type not supported by gatherScalar" };
    }
#endif
}

// Gather timer contents from all processes to root. It is like
// gatherScalar but also contains some quantities averaged over the
// processes.
auto gatherTimes(const MPIVars& mpi,
                 const Timer& timer,
                 std::vector<Times>& times) -> void;

} // namespace gadfit

// Licensed under the Apache License, Version 2.0 (the "License"); you
// may not use this file except in compliance with the License. You
// may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

// Provides Scalapack related variables and functions. If not
// compiling with Scalapack, defines Scalapack stubs.

#pragma once

#include "gadfit_mpi.h"

#include <array>
#include <vector>

#ifdef USE_SCALAPACK

extern "C"
{
    auto Cblacs_gridexit(int) -> void;
}

#else

auto Cblacs_gridexit(int) -> void;

#endif

namespace gadfit {

// A struct for basic BLACS variables. The BLACS process grid is
// created if mpi.comm is not null.
struct BLACSVars
{
private:
    bool initialized { false };

public:
    int blacs_context {};
    // Number of rows in the BLACS process grid
    int n_procs_r { 1 };
    // Number of columns in the BLACS process grid
    int n_procs_c { 1 };
    // Row location of this MPI process in the process grid
    int row {};
    // Column location of this MPI process in the process grid
    int col {};
    // Create a (rectangular) BLACS process grid. If single_column
    // then create an n x 1 grid where n is the number of MPI
    // processes.
    auto init(const MPIVars& mpi, const bool single_column = false) -> void;
    [[nodiscard]] auto isInitialized() const -> bool;
};

// Block cyclic array object used with Scalapack routines. It contains
// the Scalapack array descriptor (desc) and other parameters helpful
// when manipulating 2D block cyclic arrays.
class BCArray
{
private:
    static constexpr int scalapack_descriptor_length { 9 };
    bool initialized { false };
    auto initNoScalapack(const int data_rows, const int data_cols) -> void;

public:
    // Block sizes across rows and columns
    int block_size_r { 1 };
    int block_size_c { 1 };
    // Number of rows and columns assigned to this MPI process
    int n_rows {};
    int n_cols {};
    // Scalapack array descriptor:
    // 0 - Descriptor type (=1 for dense matrices)
    // 1 - BLACS context handle for the process grid
    // 2 - Number of rows in the global matrix
    // 3 - Number of columns in the global matrix
    // 4 - Row blocking factor
    // 5 - Column blocking factor
    // 6 - Process row over which the first row of the global matrix
    //     is distributed
    // 7 - Process column over which the first column of the global
    //     matrix is distributed
    // 8 - Leading dimension of the local matrix
    std::array<int, scalapack_descriptor_length> desc {};
    // Descriptor for when the full array is stored on root. Useful
    // for distributing or gathering an array.
    std::array<int, scalapack_descriptor_length> desc_local {};
    // Same as desc_local but the distributed array is transposed with
    // the respect to the non-distributed one. This comes into play
    // occasionally because of different array storage formats of
    // Fortran and C++.
    std::array<int, scalapack_descriptor_length> desc_local_T {};
    // Mapping of global row and column indices to the local
    // array. Values equal or greater to zero correspond to indices in
    // the local array. Negative values correspond to elements in the
    // global array not referenced by this MPI process.
    std::vector<int> global_rows {};
    std::vector<int> global_cols {};
    // Lists of row and columns indices of the global matrix assigned
    // to this MPI process
    std::vector<int> rows {};
    std::vector<int> cols {};
    // Local contents of the block cyclic array
    std::vector<double> local {};
    // Full array stored on every MPI process. Typically not
    // allocated.
    std::vector<double> global {};
    // Set blocking factors, all descriptors, and the various index
    // arrays.
    // data_rows, data_cols - are the dimensions of the global matrix
    // equal_blocks - require blocking factors to across rows and
    //                columns to be the same
    auto init(const BLACSVars& sca,
              const int data_rows,
              const int data_cols,
              const int min_n_blocks = 1,
              const bool equal_blocks = false) -> void;
    [[nodiscard]] auto isInitialized() const -> bool;
    // Use local to populate global
    auto populateGlobal(const MPIVars& mpi) -> void;
    // Same but transpose global. But since global still follows
    // Fortran memory layout this is actually not transposed in C++.
    auto populateGlobalTranspose(const MPIVars& mpi) -> void;
    // Print block-cyclic array dimensions
    auto printInfo(const MPIVars& mpi) const -> void;
};

// If compiled with Scalapack, copy contents of a shared array into a
// 2D block cyclic array. Otherwise, make a copy the shared array as a
// local array.
auto sharedToBC(const SharedArray& src, BCArray& dest) -> void;

auto copyBCArray(const BCArray& A, BCArray& B) -> void;

auto pdgemv(const char trans, const BCArray& A, const BCArray& B, BCArray& C)
  -> void;

auto pdsyr2k(const char trans, const BCArray& A, const BCArray& B, BCArray& C)
  -> void;

auto pdpotrf(BCArray& A) -> void;

auto pdpotrs(const BCArray& A, BCArray& B) -> void;

} // namespace gadfit

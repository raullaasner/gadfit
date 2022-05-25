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

#include "gadfit_scalapack.h"

#include "automatic_differentiation.h"
#include "lapack.h"

#include <numeric>
#include <spdlog/spdlog.h>

#include <spdlog/spdlog.h>

#ifdef USE_SCALAPACK

extern "C"
{
    auto Csys2blacs_handle(MPI_Comm) -> int;
    auto Cblacs_gridinit(int*, const char*, int, int) -> void;
    auto Cblacs_gridinfo(int, int*, int*, int*, int*) -> void;
    auto numroc_(const int*, const int*, const int*, const int*, const int*)
      -> int;
    auto descinit_(int* desc,
                   const int* data_rows,
                   const int* data_cols,
                   const int*,
                   const int*,
                   const int*,
                   const int*,
                   const int*,
                   const int*,
                   int*) -> void;
    auto pdgeadd_(const char*,
                  const int*,
                  const int*,
                  const double*,
                  const double*,
                  const int*,
                  const int*,
                  const int*,
                  const double*,
                  double*,
                  const int*,
                  const int*,
                  const int*) -> void;
    auto pdgemv_(const char*,
                 const int*,
                 const int*,
                 const double*,
                 const double*,
                 const int*,
                 const int*,
                 const int*,
                 const double*,
                 const int*,
                 const int*,
                 const int*,
                 const int*,
                 const double*,
                 double*,
                 const int*,
                 const int*,
                 const int*,
                 const int*) -> void;
    auto pdsyr2k_(const char*,
                  const char*,
                  const int*,
                  const int*,
                  const double*,
                  const double*,
                  const int*,
                  const int*,
                  const int*,
                  const double*,
                  const int*,
                  const int*,
                  const int*,
                  const double*,
                  double*,
                  const int*,
                  const int*,
                  const int*) -> void;
    auto pdpotrf_(const char*,
                  const int*,
                  double*,
                  const int*,
                  const int*,
                  const int*,
                  int*) -> void;
    auto pdpotrs_(const char*,
                  const int*,
                  const int*,
                  const double*,
                  const int*,
                  const int*,
                  const int*,
                  double*,
                  const int*,
                  const int*,
                  const int*,
                  int*) -> void;
}

#endif

namespace gadfit {

constexpr char trans_n { 'n' };
constexpr char trans_t { 't' };
constexpr char uplo { 'u' };
constexpr double zero {};
constexpr int i_zero {};
constexpr double half { 0.5 };
constexpr double one { 1.0 };
constexpr int i_one { 1 };
// Indices corresponding to elements of a Scalapack array descriptor
// that store the number or rows and columns of the global array
constexpr int idx_sca_r { 2 };
constexpr int idx_sca_c { 3 };

auto BLACSVars::init(const MPIVars& mpi, const bool single_column) -> void
{
    if (mpi.comm == MPI_COMM_NULL) {
        return;
    }
#ifdef USE_SCALAPACK
    blacs_context = Csys2blacs_handle(mpi.comm);
    if (single_column) {
        n_procs_r = mpi.n_procs;
        n_procs_c = 1;
    } else {
        std::array<int, 2> blacs_dims {};
        MPI_Dims_create(mpi.n_procs, 2, blacs_dims.data());
        n_procs_r = blacs_dims[0];
        n_procs_c = blacs_dims[1];
    }
    constexpr char layout { 'r' };
    Cblacs_gridinit(&blacs_context, &layout, n_procs_r, n_procs_c);
    Cblacs_gridinfo(blacs_context, &n_procs_r, &n_procs_c, &row, &col);
    if (mpi.rank == 0) {
        spdlog::debug("Initializing Scalapack:");
        spdlog::debug("  Process grid: {} x {}", n_procs_r, n_procs_c);
    }
    initialized = true;
#endif
}

auto BLACSVars::isInitialized() const -> bool
{
    return initialized;
}

// Block size is a power of two such that each process has at least
// one block - its product with the number of processes in a row or
// column of the BLACS process grid is not larger than the data
// size. Furthermore, we include an additional factor min_n_blocks
// which ensures that each process has at least that many blocks. This
// is good for load balancing. Call this function separately for block
// sizes across rows and columns.
static auto computeBlockSize(const int nproc,
                             const int data_size,
                             const int min_n_blocks = 1) -> int
{
    int block_size { 1 };
    while (2 * block_size * nproc * min_n_blocks <= data_size) {
        block_size *= 2;
    }
    return block_size;
}

auto BCArray::init(const BLACSVars& sca,
                   const int data_rows,
                   const int data_cols,
                   const int min_n_blocks,
                   const bool equal_blocks) -> void
{
    if (!sca.isInitialized()) {
        initNoScalapack(data_rows, data_cols);
        return;
    }
#ifdef USE_SCALAPACK
    // Set blocking factors
    block_size_r = computeBlockSize(sca.n_procs_r, data_rows, min_n_blocks);
    block_size_c = computeBlockSize(sca.n_procs_c, data_cols, min_n_blocks);
    if (equal_blocks) {
        const int block_size_small { std::min(block_size_r, block_size_c) };
        block_size_r = block_size_small;
        block_size_c = block_size_small;
    }
    n_rows = std::max(
      numroc_(&data_rows, &block_size_r, &sca.row, &i_zero, &sca.n_procs_r), 1);
    n_cols = std::max(
      numroc_(&data_cols, &block_size_c, &sca.col, &i_zero, &sca.n_procs_c), 1);
    // Set the main descriptor
    int info {};
    descinit_(desc.data(),
              &data_rows,
              &data_cols,
              &block_size_r,
              &block_size_c,
              &i_zero,
              &i_zero,
              &sca.blacs_context,
              &n_rows,
              &info);
    local.resize(n_rows * n_cols);
    // Set descriptor for the non-distributed matrix
    const int n_rows_local { std::max(
      numroc_(&data_rows, &data_rows, &sca.row, &i_zero, &sca.n_procs_r), 1) };
    descinit_(desc_local.data(),
              &data_rows,
              &data_cols,
              &data_rows,
              &data_cols,
              &i_zero,
              &i_zero,
              &sca.blacs_context,
              &n_rows_local,
              &info);
    // Set descriptor for the non-distributed transposed matrix
    const int n_rows_local_T { std::max(
      numroc_(&data_cols, &data_cols, &sca.row, &i_zero, &sca.n_procs_r), 1) };
    descinit_(desc_local_T.data(),
              &data_cols,
              &data_rows,
              &data_cols,
              &data_rows,
              &i_zero,
              &i_zero,
              &sca.blacs_context,
              &n_rows_local_T,
              &info);
    // Generate index arrays
    global_rows.resize(data_rows, passive_idx);
    global_cols.resize(data_cols, passive_idx);
    rows.resize(n_rows, 0);
    cols.resize(n_cols, 0);
    int local_idx {};
    for (int global_idx {}; global_idx < data_rows; ++global_idx) {
        if (global_idx / block_size_r % sca.n_procs_r == sca.row) {
            global_rows.at(global_idx) = local_idx++;
        }
    }
    local_idx = 0;
    for (int global_idx {}; global_idx < data_cols; ++global_idx) {
        if (global_idx / block_size_c % sca.n_procs_c == sca.col) {
            global_cols.at(global_idx) = local_idx++;
        }
    }
    local_idx = 0;
    for (int global_idx {}; global_idx < data_rows; ++global_idx) {
        if (global_rows.at(global_idx) != passive_idx) {
            rows.at(local_idx++) = global_idx;
        }
    }
    local_idx = 0;
    for (int global_idx {}; global_idx < data_cols; ++global_idx) {
        if (global_cols.at(global_idx) != passive_idx) {
            cols.at(local_idx++) = global_idx;
        }
    }
    with_scalapack = true;
#endif
}

auto BCArray::initNoScalapack(const int data_rows, const int data_cols) -> void
{
    block_size_r = data_rows;
    block_size_c = data_cols;
    n_rows = data_rows;
    n_cols = data_cols;
    desc[idx_sca_r] = data_rows;
    desc[idx_sca_c] = data_cols;
    global_rows.resize(n_rows);
    global_cols.resize(n_cols);
    std::iota(global_rows.begin(), global_rows.end(), 0);
    std::iota(global_cols.begin(), global_cols.end(), 0);
    rows = global_rows;
    cols = global_cols;
    local.resize(data_rows * data_cols);
}

auto BCArray::isWithScalapack() const -> bool
{
    return with_scalapack;
}

auto BCArray::populateGlobal(const MPIVars& mpi) -> void
{
    const int global_size { desc[idx_sca_r] * desc[idx_sca_c] };
    if (with_scalapack) {
#ifdef USE_SCALAPACK
        global.resize(global_size);
        pdgeadd_(&trans_n,
                 &desc[idx_sca_r],
                 &desc[idx_sca_c],
                 &one,
                 local.data(),
                 &i_one,
                 &i_one,
                 desc.data(),
                 &zero,
                 global.data(),
                 &i_one,
                 &i_one,
                 desc_local.data());
        MPI_Bcast(global.data(), global_size, MPI_DOUBLE, 0, mpi.comm);
#endif
    } else {
        global = local;
    }
}

auto BCArray::populateGlobalTranspose(const MPIVars& mpi) -> void
{
    const int global_size { desc[idx_sca_r] * desc[idx_sca_c] };
    if (with_scalapack) {
#ifdef USE_SCALAPACK
        global.resize(global_size);
        pdgeadd_(&trans_t,
                 &desc[idx_sca_c],
                 &desc[idx_sca_r],
                 &one,
                 local.data(),
                 &i_one,
                 &i_one,
                 desc.data(),
                 &zero,
                 global.data(),
                 &i_one,
                 &i_one,
                 desc_local_T.data());
        MPI_Bcast(global.data(), global_size, MPI_DOUBLE, 0, mpi.comm);
#endif
    } else {
        global = local;
    }
}

auto BCArray::printInfo(const MPIVars& mpi) const -> void
{
    std::vector<int> block_sizes_r {};
    std::vector<int> block_sizes_c {};
    std::vector<int> row_sizes {};
    std::vector<int> col_sizes {};
    gatherScalar(mpi, block_size_r, block_sizes_r);
    gatherScalar(mpi, block_size_c, block_sizes_c);
    gatherScalar(mpi, n_rows, row_sizes);
    gatherScalar(mpi, n_cols, col_sizes);
    if (mpi.rank == 0) {
        spdlog::debug(
          "      Global dimensions: {} x {}", desc[idx_sca_r], desc[idx_sca_c]);
        spdlog::debug("      Block sizes: {} x {}", block_size_r, block_size_c);
        for (int i_proc {}; i_proc < mpi.n_procs; ++i_proc) {
            spdlog::debug("      Local dimensions on MPI rank {:3}: {} x {}",
                          i_proc,
                          row_sizes.at(i_proc),
                          col_sizes.at(i_proc));
        }
    }
}

auto sharedToBC(const SharedArray& src, BCArray& dest) -> void
{
#ifdef USE_MPI
    if (src.getWin() != nullptr) {
        MPI_Win_fence(0, src.getWin());
    }
#endif
    if (dest.isWithScalapack()) {
        for (int i {}; i < dest.n_rows; ++i) {
            for (int j {}; j < dest.n_cols; ++j) {
                // The block-cyclic array will be used in Scalapck
                // routines which is a Fortran library. Hence, save the
                // transpose of dest here because Fortran uses
                // column-major order.
                dest.local[j * dest.n_rows + i] =
                  src
                    .global[dest.rows[i] * dest.desc[idx_sca_c] + dest.cols[j]];
            }
        }
    } else {
        for (int i {}; i < static_cast<int>(src.global.size()); ++i) {
            dest.local[i] = src.global[i];
        }
    }
}

auto copyBCArray(const BCArray& A, BCArray& B) -> void
{
    if (A.isWithScalapack()) {
#ifdef USE_SCALAPACK
        pdgeadd_(&trans_n,
                 &A.desc[idx_sca_r],
                 &A.desc[idx_sca_c],
                 &one,
                 A.local.data(),
                 &i_one,
                 &i_one,
                 A.desc.data(),
                 &zero,
                 B.local.data(),
                 &i_one,
                 &i_one,
                 B.desc.data());
#endif
    } else {
        B = A;
    }
}

auto pdgemv(const char trans, const BCArray& A, const BCArray& B, BCArray& C)
  -> void
{
    if (A.isWithScalapack()) {
#ifdef USE_SCALAPACK
        pdgemv_(&trans,
                &A.desc[idx_sca_r],
                &A.desc[idx_sca_c],
                &one,
                A.local.data(),
                &i_one,
                &i_one,
                A.desc.data(),
                B.local.data(),
                &i_one,
                &i_one,
                B.desc.data(),
                &i_one,
                &zero,
                C.local.data(),
                &i_one,
                &i_one,
                C.desc.data(),
                &i_one);
#endif
    } else {
        dgemv(trans,
              A.desc[idx_sca_r],
              A.desc[idx_sca_c],
              A.local,
              B.local,
              C.local);
    }
}

auto pdsyr2k(const char trans, const BCArray& A, const BCArray& B, BCArray& C)
  -> void
{
    const int K { trans == 'n' ? A.desc[idx_sca_c] : A.desc[idx_sca_r] };
    if (A.isWithScalapack()) {
#ifdef USE_SCALAPACK
        pdsyr2k_(&uplo,
                 &trans,
                 &C.desc[idx_sca_r],
                 &K,
                 &half,
                 A.local.data(),
                 &i_one,
                 &i_one,
                 A.desc.data(),
                 B.local.data(),
                 &i_one,
                 &i_one,
                 B.desc.data(),
                 &zero,
                 C.local.data(),
                 &i_one,
                 &i_one,
                 C.desc.data());
#endif
    } else {
        dsyrk(trans, C.desc[idx_sca_r], K, A.local, C.local);
    }
}

auto pdpotrf(BCArray& A) -> void
{
    if (A.isWithScalapack()) {
#ifdef USE_SCALAPACK
        int info {};
        pdpotrf_(&uplo,
                 &A.desc[idx_sca_r],
                 A.local.data(),
                 &i_one,
                 &i_one,
                 A.desc.data(),
                 &info);
#endif
    } else {
        dpptrf(A.desc[idx_sca_r], A.local);
    }
}

auto pdpotrs(const BCArray& A, BCArray& B) -> void
{
    if (A.isWithScalapack()) {
#ifdef USE_SCALAPACK
        int info {};
        pdpotrs_(&uplo,
                 &A.desc[idx_sca_r],
                 &i_one,
                 A.local.data(),
                 &i_one,
                 &i_one,
                 A.desc.data(),
                 B.local.data(),
                 &i_one,
                 &i_one,
                 B.desc.data(),
                 &info);
#endif
    } else {
        dpptrs(A.desc[idx_sca_r], A.local, B.local);
    }
}

} // namespace gadfit

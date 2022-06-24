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

// Provides the main class of the solver. This is what the user
// directly interacts with for performing a fitting procedure.

#pragma once

#include "exceptions.h"
#include "fit_function.h"
#include "scalapack.h"

#include <bitset>
#include <cassert>
#include <memory>
#include <set>

namespace gadfit {

namespace io {

constexpr int n_io_flags { 6 };
using flag = std::bitset<n_io_flags>;
static constexpr flag all { 0b0000'0001 };
static constexpr flag delta1 { 0b0000'0010 };
static constexpr flag delta2 { 0b0000'0100 };
static constexpr flag timings { 0b0000'1000 };
static constexpr flag hide_local { 0b0001'0000 };
static constexpr flag hide_global { 0b0010'0000 };

} // namespace io

class LMsolver
{
private:
    // Defaults for settings
    static constexpr double default_lambda { 10.0 };
    static constexpr double default_lambda_down { 10.0 };
    static constexpr double default_lambda_up { 10.0 };
    static constexpr int default_lambda_incs { 3 };
    static constexpr bool default_damp_max { true };
    static constexpr double default_DTD_min {};
    static constexpr double default_acceleration_threshold { -1.0 };
    static constexpr bool default_blacs_single_column { true };
    static constexpr int default_min_n_blocks { 100 };
    static constexpr bool default_prepare_getters { false };

    static constexpr int default_iteration_limit { 1000 };

    // If setPar is called with this value, denotes a global fitting parameter
    static constexpr int global_dataset_idx { passive_idx };

    struct
    {
        // Indices of all active parameters. For example, if
        // active[1].find(3) != active[1].end() is true then the
        // fourth parameter of the second data set is an active
        // fitting parameter, otherwise it is passive.
        std::vector<std::set<int>> active {};
        // Sum of active fitting parameters over all data sets. Global
        // parameters should not be double counted here.
        int n_active {};
        // Indices of all global active parameters.
        std::set<int> global {};
        // Number of all data points
        int n_datapoints {};
        // n_datapoints - n_active
        int degrees_of_freedom {};
        // Determines which parameters correspond to which columns in
        // the Jacobian and some other global arrays. Say we have a
        // fitting function with 3 parameters but only the first and
        // last one are active. Then jacobian[0] = {0, 1}, i.e. the
        // counting skips all passive parameters. Now say the first
        // parameter is global and the other one is local. Then for a
        // second data set jacobian[1] = {0, 2}. Global indices always
        // come before local ones in this indexing scheme. If we only
        // have a single data set then the indexing scheme is always
        // trivial, i.e. iota(jacobian.begin(), jacobian.end(), 0);
        std::vector<std::vector<int>> jacobian {};
        // When looping over all data points, specifies the range of
        // data attributed to the given MPI process. If, for example,
        // this process processes data points 13-18 from data set 1,
        // 0-71 from set 2, and no data points from data set 3, then
        // data_ranges = {{13, 18}, {0, 71}, {0, -1}}. This scheme
        // handles any number of data sets, each with a different
        // size, and any number of MPI processes.
        std::vector<std::array<int, 2>> data_ranges {};
        // Total number of data points processed by different MPI
        // processes
        std::vector<int> workloads {};
        // If all data sets were stacked into a single global array,
        // marks the starting index for different MPI processes.
        std::vector<int> global_starts {};
    } indices;
    // Basic MPI and Scalapack variables
    const MPIVars mpi;
    BLACSVars sca {};
    // All data sets must be added before any calls to setPar. This
    // variable keeps track of that.
    bool set_par_called { false };

    // Input data are stored in x_data and y_data. x_data stores the
    // independent values (typically x-values in a plot).
    std::vector<std::span<const double>> x_data {};
    // y_data stores the dependent values
    std::vector<std::span<const double>> y_data {};
    // If given, these are the data point errors which determine the
    // weights when constructing the Jacobian. Default is to use the
    // same weight for all data points.
    std::vector<std::span<const double>> errors {};
    // Model functions used for fitting. The user only specifies a
    // single fitting function but it is duplicated over all the data
    // sets (curves). This is because the fitting parameters
    // associated with the model function can differ between data
    // sets.
    std::vector<FitFunction> fit_functions {};
    // Parameter names can be optionally displayed during the fitting procedure
    std::vector<std::string> parameter_names {};
    // Main work arrays. These are all publicly available via getters.
    BCArray Jacobian {};
    BCArray JTJ {};
    BCArray DTD {};
    // JTJ + lambda * diag(JTJ), i.e. everything except delta on the left side
    BCArray left_side {};
    BCArray left_side_chol {};
    // J^T * residuals, where residuals = (y - f(beta))/sigma
    BCArray right_side {};
    BCArray residuals {};
    BCArray delta1 {};
    // Second directional derivative
    BCArray omega {};
    BCArray delta2 {};
    BCArray D_delta {};
    // Performance
    Timer Jacobian_timer {};
    Timer chi2_timer {};
    Timer linalg_timer {};
    Timer omega_timer {};
    Timer comm_timer {};
    Timer main_timer {};

public:
    LMsolver() = delete;
    // With an MPI communicator, runs in parallel. Default is to run
    // in serial.
    LMsolver(const fitSignature& function_body,
             const MPI_Comm& mpi_comm = MPI_COMM_NULL);
    LMsolver(const LMsolver&) = default;
    LMsolver(LMsolver&&) = default;
    auto operator=(const LMsolver&) -> LMsolver& = delete;
    auto operator=(LMsolver&&) -> LMsolver& = delete;

    // Data (x- and y-values and the corresponding errors if present)
    // are captured as std::span in order to avoid copies. The data
    // array type T needs to be compatible with std::span,
    // e.g. std::vector or std::array.
    template <typename T>
    auto addDataset(const T& x_data,
                    const T& y_data,
                    const std::vector<double>& errors = {}) -> void;
    // If the data go out of scope before LMsolver::fit is called, it
    // is better to call addDataset with shared pointers.
    auto addDataset(const std::shared_ptr<std::vector<double>>& x_data,
                    const std::shared_ptr<std::vector<double>>& y_data,
                    const std::shared_ptr<std::vector<double>>& errors = {})
      -> void;
    // Low level version of addDataset that deals directly with
    // pointers. Use this if pointing to a segment of an array or when
    // the template type T above is incompatible with std::span.
    auto addDataset(const int n_datapoints,
                    const double* x_data,
                    const double* y_data,
                    const double* errors = nullptr) -> void;
    // Set a fitting parameter as active/passive and local/global and
    // initialize it as a passive AD variable. The AD variables are
    // only activated when calculating the Jacobian.
    auto setPar(const int i_par,
                const double val,
                const bool active = false,
                const int i_dataset = global_dataset_idx,
                const std::string& parameter_name = "") -> void;
    // Same but for setting a global parameter with a name
    auto setPar(const int i_par,
                const double val,
                const bool active,
                const std::string& parameter_name) -> void;
    auto fit(double lambda = default_lambda) -> void;
    [[nodiscard]] auto getParValue(const int i_par,
                                   const int i_dataset = 0) const -> double;
    [[nodiscard]] auto getValue(const double arg, const int i_dataset = 0) const
      -> double;

    [[nodiscard]] auto getJacobian() const -> const std::vector<double>&;
    [[nodiscard]] auto getJTJ() const -> const std::vector<double>&;
    [[nodiscard]] auto getDTD() const -> const std::vector<double>&;
    [[nodiscard]] auto getLeftSide() const -> const std::vector<double>&;
    [[nodiscard]] auto getRightSide() const -> const std::vector<double>&;
    [[nodiscard]] auto getResiduals() const -> const std::vector<double>&;
    [[nodiscard]] auto chi2() -> double;
    ~LMsolver();

    struct
    {
        int iteration_limit { default_iteration_limit };
        int lambda_incs { default_lambda_incs };
        double lambda_down { default_lambda_down };
        double lambda_up { default_lambda_up };
        bool damp_max { default_damp_max };
        std::vector<double> DTD_min { default_DTD_min };
        double acceleration_threshold { default_acceleration_threshold };
        bool blacs_single_column { default_blacs_single_column };
        int min_n_blocks { default_min_n_blocks };
        bool prepare_getters { default_prepare_getters };
        io::flag verbosity {};
    } settings; // NOLINT: exposing this is harmless and convenient
                // for the user

private:
    // If addDataset was called with shared pointers as arguments,
    // store those in these variables. This ensures the data do not go
    // out of scope until the fitting procedure has finished and
    // LMsolver has gone out of scope.
    std::vector<std::shared_ptr<std::vector<double>>> x_data_shared {};
    std::vector<std::shared_ptr<std::vector<double>>> y_data_shared {};
    std::vector<std::shared_ptr<std::vector<double>>> errors_shared {};
    // Determines all relevant dimensions and parameter positions in
    // the Jacobian. Also, if using MPI, distributes all data among
    // the processes. This is called before every fitting procedure in
    // case the user activates or deactivates some parameters between
    // multiple calls to fit.
    auto prepareIndexing() -> void;
    // Detailed info about 2D block-cyclic arrays for debugging
    auto printMatrixSizes() const -> void;
    auto resetTimers() -> void;
    // J^T x J + lambda diag(D^T x D)
    auto computeLeftHandSide(const double lambda,
                             SharedArray& Jacobian_shared,
                             SharedArray& residuals_shared) -> void;
    // J^T x (y - f(beta))
    auto computeRightHandSide() -> void;
    // Compute the update vector delta1 and optionally delta2 (the
    // acceleration term)
    auto computeDeltas(SharedArray& omega_shared) -> void;
    auto deactivateParameters(const int i_set) -> void;
    auto saveParameters(std::vector<std::vector<double>>& old_parameters)
      -> void;
    auto updateParameters() -> void;
    auto revertParameters(const std::vector<std::vector<double>>& parameters)
      -> void;
    auto printIterationResults(const int i_iteration,
                               const double lambda,
                               const double new_chi2) const -> void;
    auto printTimings() const -> void;
    // Test whether an IO flag is enabled
    [[nodiscard]] auto ioTest(io::flag flag) const -> bool;
};

// LCOV_EXCL_START
template <typename T>
auto LMsolver::addDataset(const T& x_data,
                          const T& y_data,
                          const std::vector<double>& errors) -> void
{
    assert(x_data.size() == y_data.size());
    assert(errors.empty() || x_data.size() == errors.size());
    addDataset(static_cast<int>(x_data.size()),
               x_data.data(),
               y_data.data(),
               errors.data());
}
// LCOV_EXCL_STOP

} // namespace gadfit

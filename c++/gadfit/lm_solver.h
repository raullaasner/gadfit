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
#include "parallel.h"

#include <bitset>
#include <memory>
#include <set>
#include <vector>

namespace gadfit {

namespace io {

using flag = std::bitset<4>;
static constexpr flag all { 0b0000'0001 };
static constexpr flag delta1 { 0b0000'0010 };
static constexpr flag delta2 { 0b0000'0100 };
static constexpr flag timings { 0b0000'1000 };

} // namespace io

class LMsolver
{
private:
    // Defaults for options
    static constexpr double default_lambda { 10.0 };
    static constexpr double default_lambda_down { 10.0 };
    static constexpr double default_lambda_up { 10.0 };
    static constexpr int default_lambda_incs { 3 };
    static constexpr bool default_damp_max { true };
    static constexpr double default_DTD_min {};
    static constexpr double default_acceleration_threshold { -1.0 };

    static constexpr int default_iteration_limit { 1000 };

    // If setPar is called with this value, denotes a global fitting parameter
    static constexpr int global_dataset_idx { -1 };
    // If data_ranges[i].back() equals this value, where i is a data
    // set, then no data points from that data set are processed by
    // the given MPI process.
    static constexpr int passive_data_range_idx { -1 };

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

    // MPI
    MPI_Comm mpi_comm {};
    int my_rank {};
    int num_procs { 1 };

    // All data sets must be added before any calls to setPar. This
    // variable keeps track of that.
    bool set_par_called { false };

    std::vector<std::vector<double>> x_data {};
    std::vector<std::vector<double>> y_data {};
    std::vector<std::vector<double>> errors {};
    std::vector<FitFunction> fit_functions {};
    // Main work arrays. These are all publicly available via getters.
    std::vector<double> Jacobian {};
    std::vector<double> JTJ {};
    std::vector<double> DTD {};
    // JTJ + lambda * diag(JTJ), i.e. everything except delta on the left side
    std::vector<double> left_side {};
    // J^T * residuals, where residuals = (y - f(beta))/sigma
    std::vector<double> right_side {};
    std::vector<double> residuals {};
    std::vector<double> delta1 {};
    // Second directional derivative
    std::vector<double> omega {};
    std::vector<double> delta2 {};

    // Performance
    Timer Jacobian_timer {};
    Timer chi2_timer {};
    Timer linalg_timer {};
    Timer omega_timer {};
    Timer main_timer {};

public:
    LMsolver() = delete;
    // With an MPI communicator, runs in parallel. Default is to run
    // in serial.
    LMsolver(const fitSignature& function_body,
             MPI_Comm mpi_comm = MPI_COMM_NULL);
    LMsolver(const LMsolver&) = default;
    LMsolver(LMsolver&&) = default;
    auto operator=(const LMsolver&) -> LMsolver& = default;
    auto operator=(LMsolver&&) -> LMsolver& = default;

    template <typename T>
    auto addDataset(const T& x_data,
                    const T& y_data,
                    const std::vector<double>& errors = {}) -> void;
    // Set a fitting parameter as active/passive and local/global and
    // initialize it as a passive AD variable. The AD variables are
    // only activated when calculating the Jacobian.
    auto setPar(const int i_par,
                const double val,
                const bool active = false,
                const int i_dataset = global_dataset_idx) -> void;
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
        io::flag verbosity {};
    } settings; // NOLINT: exposing this is harmless and convenient
                // for the user

private:
    // General purpose work array for storing intermediate
    // results. Size can vary throughout the code.
    std::vector<double> work_tmp {};

    // Determines all relevant dimensions and parameter positions in
    // the Jacobian. Also, if using MPI, distributes all data among
    // the processes. This is called before every fitting procedure in
    // case the user activates or deactivates some parameters between
    // multiple calls to fit.
    auto prepareIndexing() -> void;
    auto resetTimers() -> void;
    auto computeLeftHandSide(const double lambda,
                             std::vector<double>& Jacobian_fragment,
                             std::vector<double>& residuals_fragment) -> void;
    auto computeRightHandSide() -> void;
    // Compute the update vector delta1 and optionally delta2 (the
    // acceleration term)
    auto computeDeltas(std::vector<double>& omega_fragment) -> void;
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

template <typename T>
auto LMsolver::addDataset(const T& x_data,
                          const T& y_data,
                          const std::vector<double>& errors) -> void
{
    if (set_par_called) {
        throw LateAddDatasetCall {};
    }
    this->x_data.push_back(std::vector<double>(x_data.size()));
    this->y_data.push_back(std::vector<double>(y_data.size()));
    for (int i {}; i < static_cast<int>(x_data.size()); ++i) {
        this->x_data.back()[i] = x_data[i];
        this->y_data.back()[i] = y_data[i];
    }
    if (!errors.empty()) {
        this->errors.emplace_back(std::vector<double>(errors.size()));
        for (int i {}; i < static_cast<int>(errors.size()); ++i) {
            this->errors.back()[i] = errors[i];
        }
    } else {
        this->errors.emplace_back(std::vector<double>(x_data.size(), 1.0));
    }
    if (fit_functions.size() < this->x_data.size()) {
        fit_functions.push_back(fit_functions.back());
    }
    indices.active.emplace_back(std::set<int> {});
}

} // namespace gadfit

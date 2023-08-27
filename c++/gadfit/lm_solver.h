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
// directly interacts with for performing a fitting procedure. In
// order to run the fitting procedure in parallel set
// LMsolver::settings.n_threads to the number of OpenMP threads.

#pragma once

#include "exceptions.h"
#include "fit_function.h"
#include "timer.h"

#include <bitset>
#include <cassert>
#include <set>
#include <span>

namespace gadfit {

struct Indices
{
    // Indices of all active parameters. For example, if
    // active[1].find(3) != active[1].end() is true then the fourth
    // parameter of the second data set is an active fitting
    // parameter, otherwise it is passive.
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
    // Determines which parameters correspond to which columns in the
    // Jacobian and some other global arrays. Say we have a fitting
    // function with 3 parameters but only the first and last one are
    // active. Then jacobian[0] = {0, 1}, i.e. the counting skips all
    // passive parameters. Now say the first parameter is global and
    // the other one is local. Then for the second data set
    // jacobian[1] = {0, 2}. Global indices always come before local
    // ones. If we only have a single data set then the indexing
    // scheme is always trivial, i.e. std::iota(jacobian.begin(),
    // jacobian.end(), 0);
    std::vector<std::vector<int>> jacobian {};
};

namespace io {

// IO flags for the LMsolver::verbosity settings
constexpr int n_io_flags { 7 };
using flag = std::bitset<n_io_flags>;
static constexpr flag all { 0b0000'0001 };
static constexpr flag delta1 { 0b0000'0010 };
static constexpr flag delta2 { 0b0000'0100 };
static constexpr flag timings { 0b0000'1000 };
static constexpr flag hide_local { 0b0001'0000 };
static constexpr flag hide_global { 0b0010'0000 };
static constexpr flag final_only { 0b0100'0000 };

} // namespace io

// Loss functions
enum class Loss
{
    linear,
    cauchy,
    huber,
};

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
    static constexpr int default_iteration_limit { 1000 };
    static constexpr int default_ad_sweep_size { default_sweep_size };
    static constexpr int default_n_threads { 1 };

    // If setPar is called with this value, denotes a global fitting parameter
    static constexpr int global_dataset_idx { passive_idx };

    Indices indices {};
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
    // If no error vector was provided we create a dummy one with all ones
    std::vector<std::vector<double>> errors_dummy {};
    // Model functions used for fitting. The user only specifies a
    // single fitting function but it is duplicated over all the data
    // sets (curves). This is because the fitting parameters
    // associated with the model function can differ between data
    // sets.
    std::vector<FitFunction> fit_functions {};
    // Parameter names can be optionally displayed during the fitting procedure
    std::vector<std::string> parameter_names {};
    // Main work arrays. These are all publicly available via getters.
    std::vector<double> jacobian {};
    std::vector<double> JTJ {};
    std::vector<double> DTD {};
    // JTJ + lambda * diag(JTJ), i.e. everything except delta on the left side
    std::vector<double> left_side {};
    std::vector<double> left_side_chol {};
    // J^T * residuals, where residuals = (y - f(beta))/sigma
    std::vector<double> right_side {};
    std::vector<double> residuals {};
    std::vector<double> delta1 {};
    // Second directional derivative
    std::vector<double> omega {};
    std::vector<double> delta2 {};
    std::vector<double> D_delta {};
    // Performance
    Timer Jacobian_timer {};
    Timer chi2_timer {};
    Timer linalg_timer {};
    Timer omega_timer {};
    Timer main_timer {};

public:
    LMsolver() = delete;
    LMsolver(const fitSignature& function_body);
    LMsolver(const LMsolver&) = default;
    LMsolver(LMsolver&&) = default;
    auto operator=(const LMsolver&) -> LMsolver& = delete;
    auto operator=(LMsolver&&) -> LMsolver& = delete;

    // Data (x- and y-values and the corresponding errors if present)
    // are captured as std::span in order to avoid copies. The data
    // array type T needs to be compatible with std::span,
    // e.g. std::vector or std::array.
    auto addDataset(const auto& x_data, const auto& y_data, const auto& errors)
      -> void;
    auto addDataset(const auto& x_data, const auto& y_data) -> void;
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

    [[nodiscard]] auto degreesOfFreedom() const -> int;
    [[nodiscard]] auto getJacobian() -> const std::vector<double>&;
    [[nodiscard]] auto getJTJ() -> const std::vector<double>&;
    [[nodiscard]] auto getDTD() -> const std::vector<double>&;
    [[nodiscard]] auto getLeftSide() -> const std::vector<double>&;
    [[nodiscard]] auto getRightSide() -> const std::vector<double>&;
    [[nodiscard]] auto getResiduals() -> const std::vector<double>&;
    [[nodiscard]] auto getInvJTJ() -> const std::vector<double>&;
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
        Loss loss { Loss::linear };
        int ad_sweep_size { default_ad_sweep_size };
        int n_threads { default_n_threads };
    } settings; // NOLINT: exposing this is harmless and convenient
                // for the user

private:
    // J^T x J + lambda diag(D^T x D)
    auto computeLeftHandSide(const double lambda) -> void;
    // J^T x (y - f(beta))
    auto computeRightHandSide() -> void;
    // Compute the update vector delta1 and optionally delta2 (the
    // acceleration term)
    auto computeDeltas(std::vector<double>& omega_shared) -> void;
    auto printIterationResults(const int i_iteration,
                               const double lambda,
                               const double new_chi2) const -> void;
    auto printTimings() const -> void;
    // Test whether an IO flag is enabled
    [[nodiscard]] auto ioTest(io::flag flag) const -> bool;
};

// LCOV_EXCL_START
auto LMsolver::addDataset(const auto& x_data,
                          const auto& y_data,
                          const auto& errors) -> void
{
    assert(x_data.size() == y_data.size());
    assert(x_data.size() == errors.size());
    addDataset(static_cast<int>(x_data.size()),
               x_data.data(),
               y_data.data(),
               errors.data());
}
auto LMsolver::addDataset(const auto& x_data, const auto& y_data) -> void
{
    assert(x_data.size() == y_data.size());
    addDataset(
      static_cast<int>(x_data.size()), x_data.data(), y_data.data(), nullptr);
}
// LCOV_EXCL_STOP

} // namespace gadfit

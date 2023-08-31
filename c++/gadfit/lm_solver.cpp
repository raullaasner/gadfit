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

#include "lm_solver.h"

#include "fit_function.h"
#include "lapack.h"
#include "numerical_integration.h"

#include <numeric>
#include <spdlog/spdlog.h>

namespace gadfit {

LMsolver::LMsolver(const fitSignature& function_body)
{
    fit_functions.emplace_back(function_body);
}

auto LMsolver::addDataset(const int n_datapoints,
                          const double* x_data,
                          const double* y_data,
                          const double* errors) -> void
{
    if (set_par_called) {
        throw LateAddDatasetCall {};
    }
    this->x_data.emplace_back(x_data, static_cast<size_t>(n_datapoints));
    this->y_data.emplace_back(y_data, static_cast<size_t>(n_datapoints));
    if (static_cast<bool>(errors)) {
        this->errors.emplace_back(errors, static_cast<size_t>(n_datapoints));
    } else {
        if (errors_dummy.size() < this->x_data.size()) {
            errors_dummy.resize(this->x_data.size());
        }
        // By default all data points to have equal weights
        errors_dummy.back() = std::vector<double>(n_datapoints, 1.0);
        this->errors.emplace_back(errors_dummy.back());
    }
    // Make a new copy of the fitting function corresponding to the
    // last data set.
    if (fit_functions.size() < this->x_data.size()) {
        fit_functions.push_back(fit_functions.back());
    }
    indices.active.emplace_back();
}

auto LMsolver::setPar(const int i_par,
                      const double val,
                      const bool active,
                      const int i_dataset,
                      const std::string& parameter_name) -> void
{
    // Make sure addDataset was called enough times.
    if (i_dataset >= static_cast<int>(x_data.size())
        || static_cast<int>(x_data.size()) == 0) {
        throw SetParInvalidIndex { i_dataset };
    }
    set_par_called = true;
    // When setting a global fitting parameter, update the indexing
    // array keeping track of all global parameters and edit each
    // instance of fit_functions accordingly.
    if (i_dataset == global_dataset_idx) {
        if (active) {
            indices.global.insert(i_par);
        } else {
            indices.global.erase(i_par);
        }
        for (int i_group {}; i_group < static_cast<int>(fit_functions.size());
             ++i_group) {
            fit_functions.at(i_group).par(i_par) = AdVar { val };
            fit_functions.at(i_group).deactivatePar(i_par);
            if (active) {
                indices.active.at(i_group).insert(i_par);
            } else {
                indices.active.at(i_group).erase(i_par);
            }
        }
    } else {
        indices.global.erase(i_par);
        fit_functions.at(i_dataset).par(i_par) = AdVar { val };
        fit_functions.at(i_dataset).deactivatePar(i_par);
        if (active) {
            indices.active.at(i_dataset).insert(i_par);
        } else {
            indices.active.at(i_dataset).erase(i_par);
        }
    }
    if (i_par >= static_cast<int>(parameter_names.size())) {
        parameter_names.resize(i_par + 1, "");
    }
    if (!parameter_name.empty()) {
        parameter_names.at(i_par) = parameter_name;
    }
}

auto LMsolver::setPar(const int i_par,
                      const double val,
                      const bool active,
                      const std::string& parameter_name) -> void
{
    setPar(i_par, val, active, global_dataset_idx, parameter_name);
}

// Determine all relevant dimensions and parameter positions in the
// Jacobian. In case the user activates or deactivates some parameters
// between fitting procedures, this is called before every call to
// LMsolver::fit.
static auto prepareIndexing(const std::vector<std::span<const double>>& x_data,
                            const std::vector<FitFunction>& fit_functions,
                            Indices& indices) -> void
{
    indices.n_active = 0;
    for (const auto& active : indices.active) {
        indices.n_active += static_cast<int>(active.size());
    }
    indices.n_active -=
      static_cast<int>((indices.active.size() - 1) * indices.global.size());
    if (indices.n_active > 0 && x_data.size() > 1 && indices.global.empty()) {
        throw NoGlobalParameters {};
    }
    indices.n_datapoints = 0;
    for (const auto& dataset : x_data) {
        indices.n_datapoints += static_cast<int>(dataset.size());
    }
    indices.degrees_of_freedom = indices.n_datapoints - indices.n_active;
    if (indices.degrees_of_freedom < 0) {
        throw NegativeDegreesOfFreedom {};
    }
    if (indices.degrees_of_freedom == 0) {
        spdlog::warn("Warning: there are no degrees of freedom - chi2/DOF has "
                     "no meaning.");
        indices.degrees_of_freedom = 1;
    }
    for (const auto& func : fit_functions) {
        if (func.getNumPars() != fit_functions.front().getNumPars()) {
            throw UninitializedParameter {};
        }
    }
    // The way indices.jacobian is built up tries to be as flexible as
    // possible. The global indices come before local indices, meaning
    // that the first columns in the Jacobian always correspond to
    // global fitting parameters. If a parameter is local, it doesn't
    // have to be active for each data set.
    indices.jacobian.resize(x_data.size());
    int next_idx {};
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        auto& cur_jacobian { indices.jacobian.at(i_set) };
        cur_jacobian.resize(indices.active.at(i_set).size());
        int next_global_pos {};
        if (i_set == 0) {
            auto active_index { indices.active.at(i_set).cbegin() };
            for (int i_par {}; i_par < static_cast<int>(cur_jacobian.size());
                 ++i_par) {
                if (indices.global.find(*active_index++)
                    != indices.global.cend()) {
                    cur_jacobian.at(i_par) = next_global_pos++;
                } else {
                    cur_jacobian.at(i_par) =
                      static_cast<int>(indices.global.size()) + next_idx++;
                }
            }
            next_idx = static_cast<int>(cur_jacobian.size());
        } else {
            auto active_index { indices.active.at(i_set).cbegin() };
            for (int i_par {}; i_par < static_cast<int>(cur_jacobian.size());
                 ++i_par) {
                if (indices.global.find(*active_index++)
                    != indices.global.cend()) {
                    cur_jacobian.at(i_par) = next_global_pos++;
                } else {
                    cur_jacobian.at(i_par) = next_idx++;
                }
            }
        }
    }
    if (indices.n_active == 0) {
        throw NoFittingParameters {};
    }
}

static auto saveParameters(const Indices& indices,
                           const std::vector<FitFunction>& fit_functions,
                           std::vector<std::vector<double>>& old_parameters)
  -> void
{
    for (int i_set {}; i_set < static_cast<int>(fit_functions.size());
         ++i_set) {
        for (const auto& idx : indices.active.at(i_set)) {
            old_parameters.at(i_set).at(idx) =
              fit_functions.at(i_set).par(idx).val;
        }
    }
}

static auto revertParameters(
  const Indices& indices,
  const std::vector<std::vector<double>>& old_parameters,
  std::vector<FitFunction>& fit_functions) -> void
{
    for (int i_set {}; i_set < static_cast<int>(fit_functions.size());
         ++i_set) {
        for (const auto& idx : indices.active.at(i_set)) {
            fit_functions.at(i_set).par(idx).val =
              old_parameters.at(i_set).at(idx);
        }
    }
}

static auto deactivateParameters(const Indices& indices,
                                 const int i_set,
                                 std::vector<FitFunction>& fit_functions)
  -> void
{
    for (const auto& idx : indices.active.at(i_set)) {
        fit_functions.at(i_set).deactivatePar(idx);
    }
}

static auto updateParameters(const Indices& indices,
                             const double acceleration_threshold,
                             const std::vector<double>& delta1,
                             const std::vector<double>& delta2,
                             std::vector<FitFunction>& fit_functions) -> void
{
    const int n_datasets { static_cast<int>(fit_functions.size()) };
    for (int i_set {}; i_set < n_datasets; ++i_set) {
        auto idx_jacobian { indices.jacobian.at(i_set).cbegin() };
        for (const auto& idx : indices.active.at(i_set)) {
            fit_functions[i_set].par(idx).val += delta1[*idx_jacobian++];
        }
    }
    if (acceleration_threshold > 0.0) {
        for (int i_set {}; i_set < n_datasets; ++i_set) {
            auto idx_jacobian { indices.jacobian.at(i_set).cbegin() };
            for (const auto& idx : indices.active.at(i_set)) {
                constexpr double half { 0.5 };
                fit_functions.at(i_set).par(idx).val -=
                  half * delta2[*idx_jacobian++];
            }
        }
    }
}

// Return the square root of the derivative of the loss function. This
// value is 1 unless using a robust cost function.
[[nodiscard]] static auto loss(const Loss loss_type, const double res) -> double
{
    // In the formulas, z is the square or the residual
    switch (loss_type) {
    case Loss::cauchy:
        // rho(z) = ln(1 + z)
        // sqrt(d rho(z) / dz) = sqrt(1 / (1 + z))
        return std::sqrt(1.0 / (1.0 + res * res));
    case Loss::huber:
        // If res <= 1
        //   rho(z) = z
        //   sqrt(d rho(z) / dz) = 1
        // else
        //   rho(z) = 2 z^0.5 - 1
        //   sqrt(d rho(z) / dz) = sqrt(1 / sqrt(z))
        if (res * res > 1.0) {
            return std::sqrt(1.0 / std::abs(res));
        } else {
            return 1.0;
        }
    case Loss::linear:
        // rho(z) = z
        // sqrt(d rho(z) / dz) = 1
        return 1.0;
    default:
        return 0.0;
    }
}

auto LMsolver::computeLeftHandSide(const double lambda) -> void
{
    Jacobian_timer.start();
    int i_start {}; // Starting index for data sets
    std::vector<double> adjoints {};
#pragma omp parallel num_threads(settings.n_threads)                           \
  firstprivate(i_start, fit_functions) private(adjoints)
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        int cur_idx { -1 };
        for (const auto& idx : indices.active[i_set]) {
            fit_functions[i_set].activateParReverse(idx, ++cur_idx);
            addADSeed(fit_functions[i_set].par(idx));
        }
#pragma omp for nowait
        for (int i_point = 0; i_point < static_cast<int>(x_data[i_set].size());
             ++i_point) {
            // One residual
            const double res {
                (y_data[i_set][i_point]
                 - fit_functions[i_set](x_data[i_set][i_point]).val)
                / errors[i_set][i_point]
            };
            // Square root of the derivative of the loss function
            const double drho_sqrt { loss(settings.loss, res) };
            residuals[i_start + i_point] = drho_sqrt * res;
            gadfit::returnSweep(*indices.active[i_set].rbegin(), adjoints);
            for (int i_par {};
                 i_par < static_cast<int>(indices.jacobian[i_set].size());
                 ++i_par) {
                jacobian[(i_start + i_point) * indices.n_active
                         + indices.jacobian[i_set][i_par]] =
                  drho_sqrt * adjoints[i_par] / errors[i_set][i_point];
            }
        }
        deactivateParameters(indices, i_set, fit_functions);
        i_start += static_cast<int>(x_data[i_set].size());
    }
    Jacobian_timer.stop();
    linalg_timer.start();
    dsyrk('t', indices.n_active, indices.n_datapoints, jacobian, JTJ);
    for (int i_par {}; i_par < indices.n_active; ++i_par) {
        const int idx { i_par * indices.n_active + i_par };
        DTD[idx] = settings.damp_max ? std::max(DTD[idx], JTJ[idx]) : JTJ[idx];
    }
    for (int i {}; i < static_cast<int>(left_side.size()); ++i) {
        left_side[i] = JTJ[i] + lambda * DTD[i];
    }
    linalg_timer.stop();
}

auto LMsolver::computeRightHandSide() -> void
{
    linalg_timer.start();
    dgemv('t',
          indices.n_datapoints,
          indices.n_active,
          jacobian,
          residuals,
          right_side);
    linalg_timer.stop();
}

auto LMsolver::computeDeltas(std::vector<double>& omega) -> void
{
    linalg_timer.start();
    delta1 = right_side;
    left_side_chol = left_side;
    dpptrf(indices.n_active, left_side_chol);
    dpptrs(indices.n_active, left_side_chol, delta1);
    linalg_timer.stop();
    if (!(settings.acceleration_threshold > 0.0)) {
        return;
    }
    // This loop is similar to the residual or Jacobian calculation
    // except only the second directional derivatives are computed.
    omega_timer.start();
    int i_start {};
#pragma omp parallel num_threads(settings.n_threads)                           \
  firstprivate(i_start, fit_functions)
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        auto idx_jacobian { indices.jacobian.at(i_set).cbegin() };
        for (const auto& idx : indices.active.at(i_set)) {
            fit_functions.at(i_set).activateParForward(idx,
                                                       delta1[*idx_jacobian++]);
        }
#pragma omp for nowait
        for (int i_point = 0; i_point < static_cast<int>(x_data[i_set].size());
             ++i_point) {
            omega[i_start + i_point] =
              fit_functions[i_set](x_data[i_set][i_point]).dd
              / errors[i_set][i_point];
        }
        deactivateParameters(indices, i_set, fit_functions);
        i_start += static_cast<int>(x_data[i_set].size());
    }
    omega_timer.stop();
    linalg_timer.start();
    dgemv('t', indices.n_datapoints, indices.n_active, jacobian, omega, delta2);
    dpptrs(indices.n_active, left_side_chol, delta2);
    // Use acceleration only if sqrt((delta2 D delta2)/(delta1 D
    // delta1)) < alpha, where alpha is the acceleration ratio
    // threshold.
    dgemv('n', indices.n_active, indices.n_active, DTD, delta2, D_delta);
    const double D_delta2 { std::inner_product(
      delta2.cbegin(), delta2.cend(), D_delta.cbegin(), 0.0) };
    dgemv('n', indices.n_active, indices.n_active, DTD, delta1, D_delta);
    const double D_delta1 { std::inner_product(
      delta1.cbegin(), delta1.cend(), D_delta.cbegin(), 0.0) };
    const double acc_ratio { std::sqrt(D_delta2 / D_delta1) };
    if (acc_ratio > settings.acceleration_threshold) {
        std::ranges::fill(delta2, 0.0);
    }
    linalg_timer.stop();
}

auto LMsolver::fit(double lambda) -> void
{
    prepareIndexing(x_data, fit_functions, indices);
    // Reset all timers
    Jacobian_timer.reset();
    chi2_timer.reset();
    linalg_timer.reset();
    omega_timer.reset();
    main_timer.reset();
    // Initialize main arrays
    jacobian.resize(indices.n_datapoints * indices.n_active);
    residuals.resize(indices.n_datapoints);
    JTJ.resize(indices.n_active * indices.n_active);
    DTD.assign(indices.n_active * indices.n_active, 0.0);
    left_side.resize(indices.n_active * indices.n_active);
    left_side_chol.resize(indices.n_active * indices.n_active);
    delta1.resize(indices.n_active);
    right_side.resize(indices.n_active);
    if (settings.acceleration_threshold > 0.0) {
        omega.resize(indices.n_datapoints);
        delta2.resize(indices.n_active);
        D_delta.resize(indices.n_active);
    }
    if (settings.DTD_min.size() > 1) {
        for (int i {}; i < indices.n_active; ++i) {
            DTD.at(i * indices.n_active + i) = settings.DTD_min.at(i);
        }
    }
    std::vector<std::vector<double>> old_parameters(x_data.size());
    for (auto& parameters : old_parameters) {
        parameters.resize(fit_functions.front().getNumPars());
    }
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        deactivateParameters(indices, i_set, fit_functions);
    }
    // Start the fitting process
    main_timer.start();
    double old_chi2 { chi2() };
    int i_iteration {};
    bool finished { settings.iteration_limit == 0 };
    while (!finished) {
        ++i_iteration;
        computeLeftHandSide(lambda);
        computeRightHandSide();
        computeDeltas(omega);
        saveParameters(indices, fit_functions, old_parameters);
        updateParameters(indices,
                         settings.acceleration_threshold,
                         delta1,
                         delta2,
                         fit_functions);
        // Update lambda which is allowed to increase no more than
        // lambda_incs times consecutively.
        for (int i_lambda {}; i_lambda <= settings.lambda_incs; ++i_lambda) {
            const double new_chi2 { chi2() };
            if (!ioTest(io::hide_all) && !ioTest(io::final_only)) {
                spdlog::debug("Current lambda: {}, new chi2 = {}, old chi2: {}",
                              lambda,
                              new_chi2,
                              old_chi2);
            }
            if (new_chi2 < old_chi2) {
                lambda /= settings.lambda_down;
                old_chi2 = new_chi2;
                if (!ioTest(io::hide_all) && !ioTest(io::final_only)) {
                    printIterationResults(i_iteration, lambda, new_chi2);
                }
                break;
            }
            if (i_lambda < settings.lambda_incs) {
                lambda *= settings.lambda_up;
                revertParameters(indices, old_parameters, fit_functions);
                for (int i {}; i < static_cast<int>(JTJ.size()); ++i) {
                    left_side[i] = JTJ[i] + lambda * DTD[i];
                }
                computeDeltas(omega);
                updateParameters(indices,
                                 settings.acceleration_threshold,
                                 delta1,
                                 delta2,
                                 fit_functions);
            } else {
                revertParameters(indices, old_parameters, fit_functions);
                if (!ioTest(io::hide_all) && !ioTest(io::final_only)) {
                    spdlog::info("Lambda increased {} times in a row",
                                 settings.lambda_incs);
                    spdlog::info("");
                }
                // This iteration was unsuccessful. Decrease
                // i_iteration so it shows the total number of
                // successful iterations.
                --i_iteration;
                finished = true;
            }
        }
        if (i_iteration == settings.iteration_limit) {
            if (!ioTest(io::hide_all)) {
                spdlog::info("Iteration limit reached");
            }
            finished = true;
        }
    }
    main_timer.stop();
    if (!ioTest(io::hide_all) && ioTest(io::final_only)
        || settings.iteration_limit == 0) {
        printIterationResults(i_iteration, lambda, old_chi2);
    }
    if (!ioTest(io::hide_all) && ioTest(io::timings)) {
        printTimings();
    }
}

auto LMsolver::chi2() -> double
{
    chi2_timer.start();
    double sum {};
#pragma omp parallel num_threads(settings.n_threads)
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
#pragma omp for reduction(+ : sum) nowait
        for (int i = 0; i < static_cast<int>(x_data[i_set].size()); ++i) {
            const double diff { (y_data[i_set][i]
                                 - fit_functions[i_set](x_data[i_set][i]).val)
                                / errors[i_set][i] };
            sum += diff * diff;
        }
    }
    chi2_timer.stop();
    return sum;
}

[[nodiscard]] auto LMsolver::degreesOfFreedom() const -> int
{
    return indices.degrees_of_freedom;
}

[[nodiscard]] auto LMsolver::getJacobian() -> const std::vector<double>&
{
    return jacobian;
}

static auto populateLowerTriange(const int dim, std::vector<double>& data)
  -> void
{
    for (int i {}; i < dim; ++i) {
        for (int j {}; j < i; ++j) {
            data[i * dim + j] = data[j * dim + i];
        }
    }
}

[[nodiscard]] auto LMsolver::getJTJ() -> const std::vector<double>&
{
    populateLowerTriange(indices.n_active, JTJ);
    return JTJ;
}

[[nodiscard]] auto LMsolver::getDTD() -> const std::vector<double>&
{
    return DTD;
}

[[nodiscard]] auto LMsolver::getLeftSide() -> const std::vector<double>&
{
    populateLowerTriange(indices.n_active, left_side);
    return left_side;
}

[[nodiscard]] auto LMsolver::getRightSide() -> const std::vector<double>&
{
    return right_side;
}

[[nodiscard]] auto LMsolver::getResiduals() -> const std::vector<double>&
{
    return residuals;
}

[[nodiscard]] auto LMsolver::getInvJTJ() -> const std::vector<double>&
{
    left_side_chol = JTJ;
    dpptrf(indices.n_active, left_side_chol);
    dpotri(indices.n_active, left_side_chol);
    populateLowerTriange(indices.n_active, left_side_chol);
    return left_side_chol;
}

auto LMsolver::printIterationResults(const int i_iteration,
                                     const double lambda,
                                     const double new_chi2) const -> void
{
    spdlog::info("Iteration: {}", i_iteration);
    spdlog::info("Lambda: {}", lambda);
    spdlog::info("Chi2/DOF: {}", new_chi2 / indices.degrees_of_freedom);
    // Constructs the line that shows the parameter value and other
    // quantities if requested
    const auto parameterLine { [&](const int i_set, const int i_par) {
        const auto active_pos { indices.active.at(i_set).find(i_par) };
        const bool is_active { active_pos != indices.active.at(i_set).cend() };
        std::ostringstream line {};
        line << std::setprecision(std::numeric_limits<double>::digits10);
        if (!parameter_names.at(i_par).empty()) {
            constexpr int max_expected_par_name_length { 15 };
            line << std::setw(max_expected_par_name_length)
                 << parameter_names.at(i_par) << ": ";
        } else {
            line << "    Parameter " << i_par << ": ";
        }
        line << fit_functions.at(i_set).getParValue(i_par);
        if (is_active) {
            // Parameter index if all inactive parameters were skipped
            const auto idx { std::distance(indices.active.at(i_set).cbegin(),
                                           active_pos) };
            if (ioTest(io::delta1)) {
                line << " (" << delta1[indices.jacobian.at(i_set).at(idx)]
                     << ')';
            }
            if (ioTest(io::delta2) && !delta2.empty()) {
                line << " (" << delta2[indices.jacobian.at(i_set).at(idx)]
                     << ')';
            }
        } else {
            line << " (fixed)";
        }
        spdlog::info(line.str());
    } };
    // When fitting more than one curve, print all global and local
    // active fitting parameters separately. Otherwise, there is no
    // distinction.
    const bool single_dataset { x_data.size() == 1 };
    if (!single_dataset && !ioTest(io::hide_global)) {
        spdlog::info("  Global parameters");
        for (int i_par {}; i_par < fit_functions.front().getNumPars();
             ++i_par) {
            if (indices.global.find(i_par) != indices.global.cend()) {
                parameterLine(0, i_par);
            }
        }
    }
    if (!ioTest(io::hide_local)) {
        for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
            if (static_cast<int>(x_data.size()) > 1) {
                spdlog::info("  Data set: {}", i_set);
            }
            for (int i_par {}; i_par < fit_functions.front().getNumPars();
                 ++i_par) {
                if (single_dataset
                    || indices.global.find(i_par) == indices.global.cend()) {
                    parameterLine(i_set, i_par);
                }
            }
        }
    }
    spdlog::info("");
}

auto LMsolver::printTimings() const -> void
{
    const auto timings { [this](const std::string_view term,
                                const Timer& timer) {
        constexpr int total_percent { 100 };
        spdlog::info("{:14} {:10.2f} {:10.2f}  {:6.2f}% {:5}",
                     term,
                     timer.totalWallTime(),
                     timer.totalCPUTime(),
                     total_percent * timer.totalCPUTime()
                       / main_timer.totalCPUTime(),
                     timer.totalNumberOfCalls());
    } };
    spdlog::info("");
    spdlog::info("Timings          Wall (s)    CPU (s)  CPU rel  Calls");
    spdlog::info("====================================================");
    timings("Jacobian", Jacobian_timer);
    timings("Chi2", chi2_timer);
    timings("Linear algebra", linalg_timer);
    timings("Omega", omega_timer);
    spdlog::info("----------------------------------------------------");
    timings("Main loop", main_timer);
    spdlog::info("====================================================");
    spdlog::info("");
}

auto LMsolver::ioTest(io::flag flag) const -> bool
{
    return (settings.verbosity & io::all).any()
           || (settings.verbosity & flag).any();
}

auto LMsolver::getParValue(const int i_par, const int i_dataset) const -> double
{
    return fit_functions.at(i_dataset).getParValue(i_par);
}

auto LMsolver::getValue(const double arg, const int i_dataset) const -> double
{
    return fit_functions.at(i_dataset)(arg).val;
}

} // namespace gadfit

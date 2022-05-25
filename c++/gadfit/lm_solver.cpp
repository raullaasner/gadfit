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
#include "numerical_integration.h"

#include <iomanip>
#include <numeric>
#include <spdlog/spdlog.h>
#include <sstream>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace gadfit {

LMsolver::LMsolver(const fitSignature& function_body, const MPI_Comm& mpi_comm)
  : mpi { mpi_comm }
{
#ifdef USE_MPI
    if (mpi_comm != MPI_COMM_NULL) {
        int mpi_initialized {};
        MPI_Initialized(&mpi_initialized);
        int mpi_finalized {};
        MPI_Finalized(&mpi_finalized);
        if (!static_cast<bool>(mpi_initialized)
            || static_cast<bool>(mpi_finalized)) {
            throw MPIUninitialized {};
        }
    }
#endif
    initializeADReverse();
    fit_functions.emplace_back(FitFunction { function_body });
}

auto LMsolver::addDataset(const int n_datapoints,
                          const double* x_data,
                          const double* y_data,
                          const double* errors) -> void
{
    if (set_par_called) {
        throw LateAddDatasetCall {};
    }
    this->x_data.emplace_back(
      std::span<const double> { x_data, static_cast<size_t>(n_datapoints) });
    this->y_data.emplace_back(
      std::span<const double> { y_data, static_cast<size_t>(n_datapoints) });
    if (static_cast<bool>(errors)) {
        this->errors.emplace_back(std::span<const double> {
          errors, static_cast<size_t>(n_datapoints) });
    } else {
        if (errors_shared.size() < this->x_data.size()) {
            errors_shared.resize(this->x_data.size());
        }
        // By default all data points to have equal weights
        errors_shared.back() = std::make_shared<std::vector<double>>(
          std::vector<double>(n_datapoints, 1.0));
        this->errors.emplace_back(
          std::span<const double> { *errors_shared.back() });
    }
    // Make a new copy of the fitting function corresponding to the
    // last data set.
    if (fit_functions.size() < this->x_data.size()) {
        fit_functions.push_back(fit_functions.back());
    }
    indices.active.emplace_back(std::set<int> {});
}

auto LMsolver::addDataset(const std::shared_ptr<std::vector<double>>& x_data,
                          const std::shared_ptr<std::vector<double>>& y_data,
                          const std::shared_ptr<std::vector<double>>& errors)
  -> void
{
    x_data_shared.push_back(x_data);
    y_data_shared.push_back(y_data);
    if (errors) {
        errors_shared.push_back(errors);
        addDataset(*x_data, *y_data, *errors);
    } else {
        addDataset(*x_data, *y_data, {});
    }
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

auto LMsolver::fit(double lambda) -> void
{
    prepareIndexing();
    if (indices.n_active == 0) {
        throw NoFittingParameters {};
    }
    resetTimers();
    sca.init(mpi, settings.blacs_single_column);
    // Initialize the shared arrays
    SharedArray Jacobian_shared {
        mpi, indices.workloads.at(mpi.rank) * indices.n_active
    };
    SharedArray residuals_shared { mpi, indices.workloads.at(mpi.rank) };
    // Initialize the block-cyclic arrays
    assert(settings.min_n_blocks >= 1
           && "Each process needs to hold at least one block if possible");
    const int& nb { settings.min_n_blocks };
    Jacobian.init(sca, indices.n_datapoints, indices.n_active, nb);
    JTJ.init(sca, indices.n_active, indices.n_active, nb);
    DTD.init(sca, indices.n_active, indices.n_active, nb);
    std::fill(DTD.local.begin(), DTD.local.end(), 0.0);
    // Need equal block sizes for the left hand side because pdpotrf
    // (Cholesky) requires square block decomposition
    left_side.init(sca, indices.n_active, indices.n_active, nb);
    left_side_chol.init(sca, indices.n_active, indices.n_active, nb, true);
    residuals.init(sca, indices.n_datapoints, 1, nb);
    delta1.init(sca, indices.n_active, 1, nb);
    right_side.init(sca, indices.n_active, 1, nb);
    if (sca.isInitialized()) {
        printMatrixSizes();
    }
    SharedArray omega_shared {};
    if (settings.acceleration_threshold > 0.0) {
        omega_shared = SharedArray { mpi, indices.workloads.at(mpi.rank) };
        omega.init(sca, indices.n_datapoints, 1, nb);
        delta2.init(sca, indices.n_active, 1, nb);
        D_delta.init(sca, indices.n_active, 1, nb);
    }
    if (static_cast<int>(settings.DTD_min.size()) > 1) {
        for (int i {}; i < indices.n_active; ++i) {
            if (JTJ.global_rows.at(i) != passive_idx
                && JTJ.global_cols.at(i) != passive_idx) {
                DTD.local.at(JTJ.global_cols.at(i) * JTJ.n_rows
                             + JTJ.global_rows.at(i)) = settings.DTD_min.at(i);
            }
        }
    }
    std::vector<std::vector<double>> old_parameters(x_data.size());
    for (auto& parameters : old_parameters) {
        parameters.resize(fit_functions.front().getNumPars());
    }
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        deactivateParameters(i_set);
    }
    main_timer.start();
    double old_chi2 { chi2() };
    int i_iteration {};
    bool finished { settings.iteration_limit == 0 };
    while (!finished) {
        ++i_iteration;
        computeLeftHandSide(lambda, Jacobian_shared, residuals_shared);
        computeRightHandSide();
        computeDeltas(omega_shared);
        saveParameters(old_parameters);
        updateParameters();
        // Update lambda which is allowed to increase no more than
        // lambda_incs times consecutively.
        for (int i_lambda {}; i_lambda <= settings.lambda_incs; ++i_lambda) {
            const double new_chi2 { chi2() };
            if (mpi.rank == 0) {
                spdlog::debug("Current lambda: {}, new chi2 = {}, old chi2: {}",
                              lambda,
                              new_chi2,
                              old_chi2);
            }
            if (new_chi2 < old_chi2) {
                lambda /= settings.lambda_down;
                old_chi2 = new_chi2;
                printIterationResults(i_iteration, lambda, new_chi2);
                break;
            }
            if (i_lambda < settings.lambda_incs) {
                lambda *= settings.lambda_up;
                revertParameters(old_parameters);
                for (int i {}; i < static_cast<int>(left_side.local.size());
                     ++i) {
                    left_side.local[i] = JTJ.local[i] + lambda * DTD.local[i];
                }
                computeDeltas(omega_shared);
                updateParameters();
            } else {
                revertParameters(old_parameters);
                if (mpi.rank == 0) {
                    spdlog::info("Lambda increased {} times in a row",
                                 settings.lambda_incs);
                    spdlog::info("");
                }
                finished = true;
            }
        }
        if (i_iteration == settings.iteration_limit) {
            if (mpi.rank == 0) {
                spdlog::info("Iteration limit reached");
            }
            finished = true;
        }
    }
    main_timer.stop();
    if (settings.iteration_limit == 0) {
        printIterationResults(0, lambda, old_chi2);
    }
    if (ioTest(io::timings)) {
        printTimings();
    }
    if (settings.prepare_getters) {
        Jacobian.populateGlobalTranspose(mpi);
        JTJ.populateGlobalTranspose(mpi);
        DTD.populateGlobalTranspose(mpi);
        left_side.populateGlobalTranspose(mpi);
        right_side.populateGlobalTranspose(mpi);
        residuals.populateGlobalTranspose(mpi);
    }
#ifdef USE_SCALAPACK
    if (sca.isInitialized()) {
        Cblacs_gridexit(sca.blacs_context);
    }
#endif
}

auto LMsolver::prepareIndexing() -> void
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
        if (mpi.rank == 0) {
            spdlog::warn(
              "Warning: there are no degrees of freedom - chi2/DOF has "
              "no meaning.");
        }
        indices.degrees_of_freedom = 1;
    }
    for (auto& func : fit_functions) {
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
    if (indices.n_datapoints < mpi.n_procs) {
        throw UnusedMPIProcesses {};
    }
    // Distribute data points over all data sets among the remaining
    // MPI processes
    indices.workloads =
      std::vector<int>(mpi.n_procs, indices.n_datapoints / mpi.n_procs);
    for (int i_rank {}; i_rank < indices.n_datapoints % mpi.n_procs; ++i_rank) {
        ++indices.workloads.at(i_rank);
    }
    indices.global_starts.resize(mpi.n_procs);
    indices.global_starts.front() = 0;
    for (int i_rank { 1 }; i_rank < mpi.n_procs; ++i_rank) {
        indices.global_starts.at(i_rank) = indices.global_starts.at(i_rank - 1)
                                           + indices.workloads.at(i_rank - 1);
    }
    const std::array<int, 2> my_global_range {
        indices.global_starts.at(mpi.rank),
        indices.global_starts.at(mpi.rank) + indices.workloads.at(mpi.rank) - 1
    };
    // If data_ranges[i].back() equals passive_idx, where i is a data
    // set, then no data points from that data set are processed by
    // the given MPI process.
    indices.data_ranges =
      std::vector<std::array<int, 2>>(x_data.size(), { 0, passive_idx });
    // Auxiliary variable that maps the current data set range onto a
    // global range. For instance, with two data sets of sizes 79 and
    // 93, cur_range is initialized to { 0, 78 } and updated to { 79, 171 }
    // in the loop below.
    std::array<int, 2> cur_range {
        0, static_cast<int>(x_data.front().size()) - 1
    };
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        if (my_global_range.front() >= cur_range.front()
            && my_global_range.front() <= cur_range.back()) {
            indices.data_ranges.at(i_set).front() =
              my_global_range.front() - cur_range.front();
        }
        if (my_global_range.back() >= cur_range.front()) {
            if (my_global_range.back() <= cur_range.back()) {
                indices.data_ranges.at(i_set).back() =
                  my_global_range.back() - cur_range.front();
            } else if (my_global_range.front() <= cur_range.back()) {
                indices.data_ranges.at(i_set).back() =
                  static_cast<int>(x_data.at(i_set).size()) - 1;
            }
        }
        if (i_set < static_cast<int>(x_data.size()) - 1) {
            cur_range.front() += static_cast<int>(x_data.at(i_set).size());
            cur_range.back() += static_cast<int>(x_data.at(i_set + 1).size());
        }
    }
}

auto LMsolver::printMatrixSizes() const -> void
{
    if (mpi.rank == 0) {
        spdlog::debug("  Matrix sizes:");
        spdlog::debug("    Jacobian:");
    }
    Jacobian.printInfo(mpi);
    if (mpi.rank == 0) {
        spdlog::debug("    JTJ:");
    }
    JTJ.printInfo(mpi);
    if (mpi.rank == 0) {
        spdlog::debug("    residuals:");
    }
    residuals.printInfo(mpi);
    if (mpi.rank == 0) {
        spdlog::debug("    delta1:");
    }
    delta1.printInfo(mpi);
    if (mpi.rank == 0) {
        spdlog::debug("    chol:");
    }
    left_side_chol.printInfo(mpi);
}
auto LMsolver::resetTimers() -> void
{
    Jacobian_timer.reset();
    chi2_timer.reset();
    linalg_timer.reset();
    omega_timer.reset();
    comm_timer.reset();
    main_timer.reset();
}

auto LMsolver::computeLeftHandSide(const double lambda,
                                   SharedArray& Jacobian_shared,
                                   SharedArray& residuals_shared) -> void
{
    auto cur_residual { residuals_shared.local.begin() };
    auto cur_Jacobian { Jacobian_shared.local.begin() };
    Jacobian_timer.start();
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        adResetSweep();
        int cur_idx { -1 };
        for (const auto& idx : indices.active.at(i_set)) {
            fit_functions.at(i_set).activateParReverse(idx, ++cur_idx);
            addADSeed(fit_functions.at(i_set).par(idx));
        }
        for (int i_point { indices.data_ranges.at(i_set).front() };
             i_point <= indices.data_ranges.at(i_set).back();
             ++i_point) {
            *cur_residual++ =
              (y_data[i_set][i_point]
               - fit_functions[i_set](x_data[i_set][i_point]).val)
              / errors[i_set][i_point];
            gadfit::returnSweep();
            for (int i_par {};
                 i_par < static_cast<int>(indices.jacobian[i_set].size());
                 ++i_par) {
                *(cur_Jacobian + indices.jacobian[i_set][i_par]) =
                  gadfit::reverse::adjoints[i_par] / errors[i_set][i_point];
            }
            cur_Jacobian += indices.n_active;
        }
        deactivateParameters(i_set);
    }
    Jacobian_timer.stop();
    comm_timer.start();
    sharedToBC(residuals_shared, residuals);
    sharedToBC(Jacobian_shared, Jacobian);
    comm_timer.stop();
    linalg_timer.start();
    pdsyr2k('t', Jacobian, Jacobian, JTJ);
    for (int i_par {}; i_par < indices.n_active; ++i_par) {
        if (JTJ.global_rows.at(i_par) != passive_idx
            && JTJ.global_cols.at(i_par) != passive_idx) {
            const int idx { JTJ.global_cols.at(i_par) * JTJ.n_rows
                            + JTJ.global_rows.at(i_par) };
            DTD.local.at(idx) = settings.damp_max
                                  ? std::max(DTD.local[idx], JTJ.local[idx])
                                  : JTJ.local[idx];
        }
    }
    for (int i {}; i < static_cast<int>(left_side.local.size()); ++i) {
        left_side.local[i] = JTJ.local[i] + lambda * DTD.local[i];
    }
    linalg_timer.stop();
}

auto LMsolver::computeRightHandSide() -> void
{
    linalg_timer.start();
    pdgemv('t', Jacobian, residuals, right_side);
    linalg_timer.stop();
}

auto LMsolver::computeDeltas(SharedArray& omega_shared) -> void
{
    linalg_timer.start();
    delta1 = right_side;
    copyBCArray(left_side, left_side_chol);
    pdpotrf(left_side_chol);
    pdpotrs(left_side_chol, delta1);
    delta1.populateGlobal(mpi);
    linalg_timer.stop();
    if (!(settings.acceleration_threshold > 0.0)) {
        return;
    }
    // This is a similar loop as for the residuals or the Jacobian
    // except only the second directional derivatives are computed.
    omega_timer.start();
    auto cur_omega { omega_shared.local.begin() };
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        auto idx_jacobian { indices.jacobian.at(i_set).cbegin() };
        for (const auto& idx : indices.active.at(i_set)) {
            fit_functions.at(i_set).activateParForward(
              idx, delta1.global.at(*idx_jacobian++));
        }
        for (int i_point { indices.data_ranges.at(i_set).front() };
             i_point <= indices.data_ranges.at(i_set).back();
             ++i_point) {
            *cur_omega++ = fit_functions[i_set](x_data[i_set][i_point]).dd
                           / errors[i_set][i_point];
        }
        deactivateParameters(i_set);
    }
    omega_timer.stop();
    comm_timer.start();
    sharedToBC(omega_shared, omega);
    comm_timer.stop();
    linalg_timer.start();
    pdgemv('t', Jacobian, omega, delta2);
    pdpotrs(left_side_chol, delta2);
    // Use acceleration only if sqrt((delta2 D delta2)/(delta1 D
    // delta1)) < alpha, where alpha is the acceleration ratio
    // threshold.
    pdgemv('n', DTD, delta2, D_delta);
    double D_delta2 { std::inner_product(delta2.local.cbegin(),
                                         delta2.local.cend(),
                                         D_delta.local.cbegin(),
                                         0.0) };
#ifdef USE_MPI
    if (mpi.comm != MPI_COMM_NULL) {
        MPI_Allreduce(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
          MPI_IN_PLACE, &D_delta2, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    }
#endif
    pdgemv('n', DTD, delta1, D_delta);
    double D_delta1 { std::inner_product(delta1.local.cbegin(),
                                         delta1.local.cend(),
                                         D_delta.local.cbegin(),
                                         0.0) };
#ifdef USE_MPI
    if (mpi.comm != MPI_COMM_NULL) {
        MPI_Allreduce(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
          MPI_IN_PLACE, &D_delta1, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    }
#endif
    double acc_ratio { std::sqrt(D_delta2 / D_delta1) };
    delta2.populateGlobal(mpi);
    if (acc_ratio > settings.acceleration_threshold) {
        std::fill(delta2.global.begin(), delta2.global.end(), 0.0);
    }
    linalg_timer.stop();
}

auto LMsolver::deactivateParameters(const int i_set) -> void
{
    for (const auto& idx : indices.active.at(i_set)) {
        fit_functions.at(i_set).deactivatePar(idx);
    }
}

auto LMsolver::saveParameters(std::vector<std::vector<double>>& old_parameters)
  -> void
{
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        for (const auto& idx : indices.active.at(i_set)) {
            old_parameters.at(i_set).at(idx) =
              fit_functions.at(i_set).par(idx).val;
        }
    }
}

auto LMsolver::updateParameters() -> void
{
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        auto idx_jacobian { indices.jacobian.at(i_set).cbegin() };
        for (const auto& idx : indices.active.at(i_set)) {
            fit_functions.at(i_set).par(idx).val +=
              delta1.global.at(*idx_jacobian++);
        }
    }
    if (settings.acceleration_threshold > 0.0) {
        for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
            auto idx_jacobian { indices.jacobian.at(i_set).cbegin() };
            for (const auto& idx : indices.active.at(i_set)) {
                fit_functions.at(i_set).par(idx).val -=
                  // NOLINTNEXTLINE
                  0.5 * delta2.global.at(*idx_jacobian++);
            }
        }
    }
}

auto LMsolver::revertParameters(
  const std::vector<std::vector<double>>& old_parameters) -> void
{
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        for (const auto& idx : indices.active.at(i_set)) {
            fit_functions.at(i_set).par(idx).val =
              old_parameters.at(i_set).at(idx);
        }
    }
}

auto LMsolver::chi2() -> double
{
    chi2_timer.start();
    double sum {};
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        // Normally we would loop over indices.data_ranges if MPI has
        // been initialized. However, if chi2() is called before fit()
        // then indices.data_ranges is not yet initialized so we do
        // this operation in serial.
        int i_point_beg {};
        int i_point_end {};
        if (indices.data_ranges.empty()) {
            i_point_beg = 0;
            i_point_end = static_cast<int>(x_data.at(i_set).size()) - 1;
        } else {
            i_point_beg = indices.data_ranges.at(i_set).front();
            i_point_end = indices.data_ranges.at(i_set).back();
        }
        for (int i_point { i_point_beg }; i_point <= i_point_end; ++i_point) {
            const double diff {
                (y_data[i_set][i_point]
                 - fit_functions[i_set](x_data[i_set][i_point]).val)
                / errors[i_set][i_point]
            };
            sum += diff * diff;
        }
    }
    chi2_timer.stop();
    comm_timer.start();
#ifdef USE_MPI
    if (mpi.comm != MPI_COMM_NULL && !indices.data_ranges.empty()) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
        MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    }
#endif
    comm_timer.stop();
    return sum;
}

auto LMsolver::getJacobian() const -> const std::vector<double>&
{
    return Jacobian.global;
}

auto LMsolver::getJTJ() const -> const std::vector<double>&
{
    return JTJ.global;
}

auto LMsolver::getDTD() const -> const std::vector<double>&
{
    return DTD.global;
}

auto LMsolver::getLeftSide() const -> const std::vector<double>&
{
    return left_side.global;
}

auto LMsolver::getRightSide() const -> const std::vector<double>&
{
    return right_side.global;
}

auto LMsolver::getResiduals() const -> const std::vector<double>&
{
    return residuals.global;
}

auto LMsolver::printIterationResults(const int i_iteration,
                                     const double lambda,
                                     const double new_chi2) const -> void
{
    if (mpi.rank != 0) {
        return;
    }
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
                line << " ("
                     << delta1.global.at(indices.jacobian.at(i_set).at(idx))
                     << ')';
            }
            if (ioTest(io::delta2) && !delta2.global.empty()) {
                line << " ("
                     << delta2.global.at(indices.jacobian.at(i_set).at(idx))
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
    std::vector<Times> Jacobian_times(mpi.n_procs);
    std::vector<Times> chi2_times(mpi.n_procs);
    std::vector<Times> linalg_times(mpi.n_procs);
    std::vector<Times> omega_times(mpi.n_procs);
    std::vector<Times> comm_times(mpi.n_procs);
    std::vector<Times> main_times(mpi.n_procs);
    gatherTimes(mpi, Jacobian_timer, Jacobian_times);
    gatherTimes(mpi, chi2_timer, chi2_times);
    gatherTimes(mpi, linalg_timer, linalg_times);
    gatherTimes(mpi, omega_timer, omega_times);
    gatherTimes(mpi, comm_timer, comm_times);
    gatherTimes(mpi, main_timer, main_times);
    if (mpi.rank == 0) {
        spdlog::info("");
        spdlog::info("Cost relative to the main loop on root process");
        spdlog::info("==============================================");
        const auto relTime { [this](const double time) {
            constexpr int total_percent { 100 };
            return total_percent * time / main_timer.totalCPUTime();
        } };
        spdlog::info("      Jacobian:{:6.2f}%",
                     relTime(Jacobian_timer.totalCPUTime()));
        spdlog::info("          chi2:{:6.2f}%",
                     relTime(chi2_timer.totalCPUTime()));
        spdlog::info("Linear algebra:{:6.2f}%",
                     relTime(linalg_timer.totalCPUTime()));
        spdlog::info("         omega:{:6.2f}%",
                     relTime(omega_timer.totalCPUTime()));
        spdlog::info(" Communication:{:6.2f}%",
                     relTime(comm_timer.totalCPUTime()));
        spdlog::info(
          "         Total:{:6.2f}%",
          relTime(Jacobian_timer.totalCPUTime() + chi2_timer.totalCPUTime()
                  + linalg_timer.totalCPUTime() + omega_timer.totalCPUTime()
                  + comm_timer.totalCPUTime()));
        spdlog::info("");
        spdlog::info(
          "Timings          CPU, s    Wall, s    CPU*, s   Wall*, s Calls");
        spdlog::info(
          "==============================================================");
        const auto timings { [this](const std::vector<Times>& times,
                                    const Timer& timer) {
            for (int i_proc {}; i_proc < mpi.n_procs; ++i_proc) {
                spdlog::info(
                  "  Proc {:5} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:5}",
                  i_proc,
                  times.at(i_proc).cpu,
                  times.at(i_proc).wall,
                  times.at(i_proc).cpu_ave,
                  times.at(i_proc).wall_ave,
                  timer.totalNumberOfCalls());
            }
        } };
        spdlog::info(" Jacobian");
        timings(Jacobian_times, Jacobian_timer);
        spdlog::info(" Chi2");
        timings(chi2_times, chi2_timer);
        spdlog::info(" Linear algebra");
        timings(linalg_times, linalg_timer);
        if (settings.acceleration_threshold > 0.0) {
            spdlog::info(" Omega");
            timings(omega_times, omega_timer);
        }
        spdlog::info(
          "--------------------------------------------------------------");
        spdlog::info(" Main loop");
        for (int i_proc {}; i_proc < mpi.n_procs; ++i_proc) {
            spdlog::info("  Proc {:5} {:10.2f} {:10.2f}",
                         i_proc,
                         main_times.at(i_proc).cpu,
                         main_times.at(i_proc).wall);
        }
        spdlog::info(
          "==============================================================");
        spdlog::info("* - per call");
    }
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

LMsolver::~LMsolver()
{
    freeIntegration();
    freeAdReverse();
}

} // namespace gadfit

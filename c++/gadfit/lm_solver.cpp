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

#include "cblas_wrapper.h"
#include "fit_function.h"
#include "numerical_integration.h"

#include <iomanip>
#include <numeric>
#include <spdlog/spdlog.h>
#include <sstream>

namespace gadfit {

LMsolver::LMsolver(const fitSignature& function_body, MPI_Comm mpi_comm)
  : mpi_comm { mpi_comm }
{
    if (mpi_comm != MPI_COMM_NULL) {
        int mpi_initialized {};
        MPI_Initialized(&mpi_initialized);
        int mpi_finalized {};
        MPI_Finalized(&mpi_finalized);
        if (!static_cast<bool>(mpi_initialized)
            || static_cast<bool>(mpi_finalized)) {
            throw MPIUninitialized {};
        }
        MPI_Comm_rank(mpi_comm, &my_rank);
        MPI_Comm_size(mpi_comm, &num_procs);
    }
    initializeADReverse();
    fit_functions.emplace_back(FitFunction { function_body });
}

auto LMsolver::setPar(const int i_par,
                      const double val,
                      const bool active,
                      const int i_dataset) -> void
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
}

auto LMsolver::fit(double lambda) -> void
{
    prepareIndexing();
    resetTimers();
    if (indices.n_active == 0 && my_rank == 0) {
        spdlog::warn("No active fitting parameters.");
    }
    // Prepare the main work arrays
    Jacobian =
      std::vector<double>(indices.n_datapoints * indices.n_active, 0.0);
    residuals.resize(indices.n_datapoints);
    if (settings.acceleration_threshold > 0.0) {
        omega.resize(indices.n_datapoints);
    }
    // Each MPI process is assigned a segment of some of work arrays
    // such as the Jacobian.
    std::vector<double> Jacobian_fragment(
      indices.workloads.at(my_rank) * indices.n_active, 0.0);
    std::vector<double> residuals_fragment(indices.workloads.at(my_rank));
    std::vector<double> omega_fragment;
    if (settings.acceleration_threshold > 0.0) {
        omega_fragment.resize(indices.workloads.at(my_rank));
    }
    JTJ.resize(indices.n_active * indices.n_active);
    DTD = std::vector<double>(JTJ.size(), 0.0);
    if (static_cast<int>(settings.DTD_min.size()) > 1) {
        for (int i {}; i < indices.n_active; ++i) {
            DTD.at(i * indices.n_active + i) = settings.DTD_min.at(i);
        }
    }
    left_side.resize(JTJ.size());
    right_side.resize(indices.n_active);
    delta1.resize(right_side.size());
    if (settings.acceleration_threshold > 0.0) {
        delta2.resize(delta1.size());
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
    bool finished { settings.iteration_limit == 0 || indices.n_active == 0 };
    while (!finished) {
        ++i_iteration;
        computeLeftHandSide(lambda, Jacobian_fragment, residuals_fragment);
        computeRightHandSide();
        computeDeltas(omega_fragment);
        saveParameters(old_parameters);
        updateParameters();
        // Update lambda, which is allowed to increase no more than
        // lambda_incs times consecutively.
        for (int i_lambda {}; i_lambda <= settings.lambda_incs; ++i_lambda) {
            const double new_chi2 { chi2() };
            if (my_rank == 0) {
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
                for (int i {}; i < static_cast<int>(left_side.size()); ++i) {
                    left_side[i] = JTJ[i] + lambda * DTD[i];
                }
                computeDeltas(omega_fragment);
                updateParameters();
            } else {
                revertParameters(old_parameters);
                if (my_rank == 0) {
                    spdlog::info("Lambda increased {} times in a row\n",
                                 settings.lambda_incs);
                }
                finished = true;
            }
        }
        if (i_iteration == settings.iteration_limit) {
            if (my_rank == 0) {
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
        if (my_rank == 0) {
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
    if (indices.n_datapoints < num_procs) {
        if (my_rank == 0) {
            spdlog::warn(
              "Warning: less data points than MPI processes. Splitting the "
              "communicator and continuing with a subset of the MPI "
              "processes.");
        }
        const int color { my_rank < indices.n_datapoints ? 1 : 0 };
        MPI_Comm_split(mpi_comm, color, my_rank, &mpi_comm);
        if (color == 1) {
            MPI_Comm_rank(mpi_comm, &my_rank);
            MPI_Comm_size(mpi_comm, &num_procs);
        } else {
            throw UnusedMPIProcess {};
        }
    }
    // Distribute data points over all data sets among the remaining
    // MPI processes
    indices.workloads =
      std::vector<int>(num_procs, indices.n_datapoints / num_procs);
    for (int i_rank {}; i_rank < indices.n_datapoints % num_procs; ++i_rank) {
        ++indices.workloads.at(i_rank);
    }
    indices.global_starts.resize(num_procs);
    indices.global_starts.front() = 0;
    for (int i_rank { 1 }; i_rank < num_procs; ++i_rank) {
        indices.global_starts.at(i_rank) = indices.global_starts.at(i_rank - 1)
                                           + indices.workloads.at(i_rank - 1);
    }
    const std::array<int, 2> my_global_range {
        indices.global_starts.at(my_rank),
        indices.global_starts.at(my_rank) + indices.workloads.at(my_rank) - 1
    };
    indices.data_ranges = std::vector<std::array<int, 2>>(
      x_data.size(), { 0, passive_data_range_idx });
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

auto LMsolver::resetTimers() -> void
{
    Jacobian_timer.reset();
    chi2_timer.reset();
    linalg_timer.reset();
    omega_timer.reset();
    main_timer.reset();
}

auto LMsolver::computeLeftHandSide(const double lambda,
                                   std::vector<double>& Jacobian_fragment,
                                   std::vector<double>& residuals_fragment)
  -> void
{
    auto cur_residual { residuals_fragment.begin() };
    auto cur_Jacobian { Jacobian_fragment.begin() };
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
    std::vector<int> sendcounts(num_procs, indices.workloads.at(my_rank));
    std::vector<int> sdispls(num_procs, 0);
    fragmentToGlobal(residuals_fragment.data(),
                     sendcounts.data(),
                     sdispls.data(),
                     residuals.data(),
                     indices.workloads.data(),
                     indices.global_starts.data(),
                     mpi_comm);
    std::vector<int> recvcounts(num_procs);
    std::vector<int> rdispls(num_procs);
    for (int i_rank {}; i_rank < num_procs; ++i_rank) {
        sendcounts.at(i_rank) *= indices.n_active;
        rdispls.at(i_rank) =
          indices.global_starts.at(i_rank) * indices.n_active;
        recvcounts.at(i_rank) = indices.workloads.at(i_rank) * indices.n_active;
    }
    fragmentToGlobal(Jacobian_fragment.data(),
                     sendcounts.data(),
                     sdispls.data(),
                     Jacobian.data(),
                     recvcounts.data(),
                     rdispls.data(),
                     mpi_comm);
    linalg_timer.start();
    dsyrk_wrapper(indices.n_active, indices.n_datapoints, Jacobian, JTJ);
    for (int i_par {}; i_par < indices.n_active; ++i_par) {
        DTD[i_par * indices.n_active + i_par] =
          settings.damp_max ? std::max(DTD[i_par * indices.n_active + i_par],
                                       JTJ[i_par * indices.n_active + i_par])
                            : JTJ[i_par * indices.n_active + i_par];
    }
    for (int i {}; i < static_cast<int>(left_side.size()); ++i) {
        left_side[i] = JTJ[i] + lambda * DTD[i];
    }
    linalg_timer.stop();
}

auto LMsolver::computeRightHandSide() -> void
{
    linalg_timer.start();
    dgemv_tr_wrapper(
      indices.n_datapoints, indices.n_active, Jacobian, residuals, right_side);
    linalg_timer.stop();
}

auto LMsolver::computeDeltas(std::vector<double>& omega_fragment) -> void
{
    delta1 = right_side;
    linalg_timer.start();
    dpptrs_wrapper(indices.n_active, left_side, delta1);
    linalg_timer.stop();
    if (settings.acceleration_threshold > 0.0) {
        // This is a similar loop as for the residuals or the Jacobian
        // above, except only the second directional derivatives are
        // computed.
        omega_timer.start();
        auto cur_omega { omega_fragment.begin() };
        for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
            auto idx_jacobian { indices.jacobian.at(i_set).cbegin() };
            for (const auto& idx : indices.active.at(i_set)) {
                fit_functions.at(i_set).activateParForward(
                  idx, delta1.at(*idx_jacobian++));
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
        std::vector<int> sendcounts(num_procs, indices.workloads.at(my_rank));
        std::vector<int> sdispls(num_procs, 0);
        fragmentToGlobal(omega_fragment.data(),
                         sendcounts.data(),
                         sdispls.data(),
                         omega.data(),
                         indices.workloads.data(),
                         indices.global_starts.data(),
                         mpi_comm);
        linalg_timer.start();
        dgemv_tr_wrapper(
          indices.n_datapoints, indices.n_active, Jacobian, omega, delta2);
        dpptrs_wrapper(indices.n_active, left_side, delta2);
        // Use acceleration only if
        // sqrt((delta2 D delta2)/(delta1 D delta1)) < alpha, where
        // alpha is the acceleration ratio threshold.
        if (static_cast<int>(work_tmp.size()) < indices.n_active) {
            work_tmp.resize(indices.n_active);
        }
        dgemv_tr_wrapper(
          indices.n_active, indices.n_active, DTD, delta2, work_tmp);
        double acc_ratio { std::inner_product(
          delta2.cbegin(), delta2.cend(), work_tmp.cbegin(), 0.0) };
        dgemv_tr_wrapper(
          indices.n_active, indices.n_active, DTD, delta1, work_tmp);
        acc_ratio /= std::inner_product(
          delta1.cbegin(), delta1.cend(), work_tmp.cbegin(), 0.0);
        acc_ratio = std::sqrt(acc_ratio);
        linalg_timer.stop();
        if (acc_ratio > settings.acceleration_threshold) {
            std::fill(delta2.begin(), delta2.end(), 0.0);
        }
    }
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
            fit_functions.at(i_set).par(idx).val += delta1.at(*idx_jacobian++);
        }
    }
    if (settings.acceleration_threshold > 0.0) {
        for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
            auto idx_jacobian { indices.jacobian.at(i_set).cbegin() };
            for (const auto& idx : indices.active.at(i_set)) {
                fit_functions.at(i_set).par(idx).val -=
                  // NOLINTNEXTLINE
                  0.5 * delta2.at(*idx_jacobian++);
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
        // been initialized. However, if chi2() is call before fit(),
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
    if (mpi_comm != MPI_COMM_NULL && !indices.data_ranges.empty()) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
        MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    }
    return sum;
}

auto LMsolver::getJacobian() const -> const std::vector<double>&
{
    return Jacobian;
}

auto LMsolver::getJTJ() const -> const std::vector<double>&
{
    return JTJ;
}

auto LMsolver::getDTD() const -> const std::vector<double>&
{
    return DTD;
}

auto LMsolver::getLeftSide() const -> const std::vector<double>&
{
    return left_side;
}

auto LMsolver::getRightSide() const -> const std::vector<double>&
{
    return right_side;
}

auto LMsolver::getResiduals() const -> const std::vector<double>&
{
    return residuals;
}

auto LMsolver::printIterationResults(const int i_iteration,
                                     const double lambda,
                                     const double new_chi2) const -> void
{
    if (my_rank != 0) {
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
        line << std::setprecision(std::numeric_limits<double>::digits10)
             << "    Parameter " << i_par << ": "
             << fit_functions.at(i_set).getParValue(i_par);
        if (is_active) {
            // Parameter index if all inactive parameters were skipped
            const auto idx { std::distance(indices.active.at(i_set).cbegin(),
                                           active_pos) };
            if (ioTest(io::delta1)) {
                line << " (" << delta1.at(indices.jacobian.at(i_set).at(idx))
                     << ')';
            }
            if (ioTest(io::delta2) && !delta2.empty()) {
                line << " (" << delta2.at(indices.jacobian.at(i_set).at(idx))
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
    if (!single_dataset) {
        spdlog::info("  Global parameters");
        for (int i_par {}; i_par < fit_functions.front().getNumPars();
             ++i_par) {
            if (indices.global.find(i_par) != indices.global.cend()) {
                parameterLine(0, i_par);
            }
        }
    }
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
    spdlog::info("");
}

auto LMsolver::printTimings() const -> void
{
    std::vector<Times> Jacobian_times(num_procs);
    std::vector<Times> chi2_times(num_procs);
    std::vector<Times> linalg_times(num_procs);
    std::vector<Times> omega_times(num_procs);
    std::vector<Times> main_times(num_procs);
    gatherTimes(mpi_comm, my_rank, num_procs, Jacobian_timer, Jacobian_times);
    gatherTimes(mpi_comm, my_rank, num_procs, chi2_timer, chi2_times);
    gatherTimes(mpi_comm, my_rank, num_procs, linalg_timer, linalg_times);
    gatherTimes(mpi_comm, my_rank, num_procs, omega_timer, omega_times);
    gatherTimes(mpi_comm, my_rank, num_procs, main_timer, main_times);
    if (my_rank == 0) {
        spdlog::info("");
        spdlog::info("Relative cost compared to the main loop on process 0");
        spdlog::info("====================================================");
        const auto relTime { [this](const double time) {
            return 100 * time / main_timer.totalCPUTime();
        } };
        spdlog::info(" Jacobian:{:6.2f}%",
                     relTime(Jacobian_timer.totalCPUTime()));
        spdlog::info("     chi2:{:6.2f}%", relTime(chi2_timer.totalCPUTime()));
        spdlog::info("   linalg:{:6.2f}%",
                     relTime(linalg_timer.totalCPUTime()));
        spdlog::info("    omega:{:6.2f}%", relTime(omega_timer.totalCPUTime()));
        spdlog::info(
          "    Total:{:6.2f}%",
          relTime(Jacobian_timer.totalCPUTime() + chi2_timer.totalCPUTime()
                  + linalg_timer.totalCPUTime() + omega_timer.totalCPUTime()));
        spdlog::info("");
        spdlog::info(
          "Timings          CPU, s    Wall, s    CPU*, s   Wall*, s Calls");
        spdlog::info(
          "==============================================================");
        const auto timings { [this](const std::vector<Times>& times,
                                    const Timer& timer) {
            for (int i_proc {}; i_proc < num_procs; ++i_proc) {
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
        if (settings.acceleration_threshold > 0.0) {
            spdlog::info(" Omega");
            timings(omega_times, omega_timer);
        }
        spdlog::info(
          "--------------------------------------------------------------");
        spdlog::info(" Main loop");
        for (int i_proc {}; i_proc < num_procs; ++i_proc) {
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

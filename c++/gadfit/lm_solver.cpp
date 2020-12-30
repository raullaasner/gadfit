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

#include <numeric>
#include <spdlog/spdlog.h>

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
    if (i_dataset >= static_cast<int>(size(x_data))
        || static_cast<int>(size(x_data)) == 0) {
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
            fit_functions.at(i_group).par(i_par) =
              AdVar { val, 0.0, 0.0, passive_idx };
            if (active) {
                indices.active.at(i_group).insert(i_par);
            } else {
                indices.active.at(i_group).erase(i_par);
            }
        }
    } else {
        indices.global.erase(i_par);
        fit_functions.at(i_dataset).par(i_par) =
          AdVar { val, 0.0, 0.0, passive_idx };
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
    if (indices.n_active == 0 && my_rank == 0) {
        spdlog::warn("No active fitting parameters.");
    }
    // Prepare the main work arrays
    Jacobian =
      std::vector<double>(indices.n_datapoints * indices.n_active, 0.0);
    residuals.resize(indices.n_datapoints);
    // Each MPI process is assigned a segment of the Jacobian and the residuals.
    std::vector<double> Jacobian_fragment(
      indices.workloads.at(my_rank) * indices.n_active, 0.0);
    std::vector<double> residuals_fragment(indices.workloads.at(my_rank));
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
    std::vector<std::vector<double>> old_parameters(x_data.size());
    for (auto& parameters : old_parameters) {
        parameters.resize(fit_functions.front().getNumPars());
    }
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        deactivateParameters(i_set);
    }
    double old_chi2 { chi2() };
    int i_iteration {};
    bool finished { settings.iteration_limit == 0 || indices.n_active == 0 };
    while (!finished) {
        ++i_iteration;
        computeLeftHandSide(lambda, Jacobian_fragment, residuals_fragment);
        computeRightHandSide();
        computeDeltas();
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
                computeDeltas();
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
    if (settings.iteration_limit == 0) {
        printIterationResults(0, lambda, old_chi2);
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
    indices.n_datapoints = 0;
    for (const auto& dataset : x_data) {
        indices.n_datapoints += static_cast<int>(dataset.size());
    }
    indices.degrees_of_freedom = indices.n_datapoints - indices.n_active;
    if (indices.degrees_of_freedom < 0) {
        throw NegativeDegreesOfFreedom {};
    }
    if (indices.degrees_of_freedom == 0 && my_rank == 0) {
        spdlog::warn(
          "There are no degrees of freedom - chi2/DOF has no meaning.");
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
    // Distribute data points over all data sets among the MPI processes
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

auto LMsolver::computeLeftHandSide(const double lambda,
                                   std::vector<double>& Jacobian_fragment,
                                   std::vector<double>& residuals_fragment)
  -> void
{
    auto cur_residual { residuals_fragment.begin() };
    auto cur_Jacobian { Jacobian_fragment.begin() };
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
}

auto LMsolver::computeRightHandSide() -> void
{
    dgemv_tr_wrapper(
      indices.n_datapoints, indices.n_active, Jacobian, residuals, right_side);
}

auto LMsolver::computeDeltas() -> void
{
    delta1 = right_side;
    dpotrs_wrapper(indices.n_active, left_side, delta1);
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

auto LMsolver::chi2() const -> double
{
    double sum {};
    for (int i_set {}; i_set < static_cast<int>(x_data.size()); ++i_set) {
        for (int i_point {}; i_point < static_cast<int>(x_data[i_set].size());
             ++i_point) {
            double diff { (y_data[i_set][i_point]
                           - fit_functions[i_set](x_data[i_set][i_point]).val)
                          / errors[i_set][i_point] };
            sum += diff * diff;
        }
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
    // When fitting more than one curve, print all global and local
    // active fitting parameters separately. Otherwise, there is no
    // distinction.
    const bool single_dataset { x_data.size() == 1 };
    if (!single_dataset) {
        spdlog::info("  Global parameters");
        for (int i_par {}; i_par < fit_functions.front().getNumPars();
             ++i_par) {
            if (indices.global.find(i_par) != indices.global.cend()) {
                spdlog::info("    Parameter {}: {}",
                             i_par,
                             fit_functions.front().getParValue(i_par));
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
                spdlog::info("    Parameter {}: {}",
                             i_par,
                             fit_functions.at(i_set).getParValue(i_par));
            }
        }
    }
    spdlog::info("");
}

auto LMsolver::getParValue(const int i_par, const int i_dataset) const -> double
{
    return fit_functions.at(i_dataset).getParValue(i_par);
}

auto LMsolver::getValue(const double arg, const int i_dataset) const -> double
{
    return fit_functions.at(i_dataset)(arg);
}

LMsolver::~LMsolver()
{
    freeAdReverse();
}

} // namespace gadfit
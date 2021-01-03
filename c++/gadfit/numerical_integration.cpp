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

#include "numerical_integration.h"

#include "exceptions.h"
#include "fit_function.h"
#include "gauss_kronrod_parameters.h"

#include <numeric>

#include <iomanip>
#include <iostream>
namespace gadfit {

// Gauss-Kronrod parameters
namespace gk {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<double> roots {};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<double> weights_gauss {};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<double> weights_kronrod {};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<double> weights_kronrod_2D {};

} // gk namespace

struct Workspace
{
    std::vector<std::array<double, 2>> bounds {};
    std::vector<double> sums {};
    std::vector<double> abs_errors {};
    std::vector<AdVar> inactive_parameters {};
    std::vector<AdVar> integrand_inactive_parameters {};
    auto resize(const int size) -> void
    {
        bounds.resize(size + 1);
        sums.resize(size);
        abs_errors.resize(size);
    }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<Workspace> workspaces {};

auto initIntegration(const int workspace_size, const int n_workspaces) -> void
{
    gk::roots = std::vector<double> { roots_15p.cbegin(), roots_15p.cend() };
    gk::weights_gauss = std::vector<double> { weights_gauss_7p.cbegin(),
                                              weights_gauss_7p.cend() };
    gk::weights_kronrod = std::vector<double> { weights_kronrod_15p.cbegin(),
                                                weights_kronrod_15p.cend() };
    const int kronrod_size { static_cast<int>(gk::weights_kronrod.size()) };
    gk::weights_kronrod_2D.resize(kronrod_size * kronrod_size);
    for (int i {}; i < kronrod_size; ++i) {
        for (int j {}; j < kronrod_size; ++j) {
            gk::weights_kronrod_2D[i * kronrod_size + j] =
              gk::weights_kronrod[i] * gk::weights_kronrod[j];
        }
    }
    workspaces.resize(n_workspaces);
    for (auto& work : workspaces) {
        work.resize(workspace_size);
    }
}

auto gaussKronrod(const integrandSignature& function,
                  const std::vector<AdVar>& parameters,
                  const double lower,
                  const double upper,
                  double& abs_error) -> double
{
    const double scale { (upper - lower) / 2 };
    const double shift { (lower + upper) / 2 };
    double sum_kronrod {};
    double sum_gauss {};
    AdVar arg {};
    arg.idx = passive_idx;
    for (int i {}; i < static_cast<int>(gk::roots.size()); ++i) {
        arg.val = scale * gk::roots[i] + shift;
        const double val { function(parameters, arg).val };
        sum_kronrod += gk::weights_kronrod[i] * val;
        if (i % 2 == 1) {
            sum_gauss += gk::weights_gauss[i / 2] * val;
        }
    }
    sum_kronrod *= scale;
    abs_error = std::abs(sum_kronrod - scale * sum_gauss);
    return sum_kronrod;
}

auto kronrod(const integrandSignature& function,
             const std::vector<AdVar>& parameters,
             const double lower,
             const double upper) -> AdVar
{
    const double scale { (upper - lower) / 2 };
    const double shift { (lower + upper) / 2 };
    AdVar sum { 0.0, passive_idx };
    AdVar arg {};
    arg.idx = passive_idx;
    for (int i {}; i < static_cast<int>(gk::roots.size()); ++i) {
        arg.val = scale * gk::roots[i] + shift;
        const AdVar val { function(parameters, arg) };
        sum += gk::weights_kronrod[i] * val;
    }
    return scale * sum;
}

auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const double lower,
               const double upper,
               const double rel_error,
               const double abs_error) -> AdVar
{
    static int int_order { -1 };
    ++int_order;
    if (workspaces.empty()) {
        initIntegration();
    }
    auto& work { workspaces[int_order] };
    if (work.inactive_parameters.empty()) {
        work.inactive_parameters.resize(parameters.size());
        for (int i {}; i < static_cast<int>(parameters.size()); ++i) {
            work.inactive_parameters[i].val = parameters[i].val;
            work.inactive_parameters[i].idx = -1;
        }
    }
    work.bounds.front() = { lower, upper };
    work.sums.front() = { gaussKronrod(function,
                                       work.inactive_parameters,
                                       lower,
                                       upper,
                                       work.abs_errors.front()) };
    for (int iter {}; iter < static_cast<int>(work.sums.size()); ++iter) {
        const int max_error_idx { static_cast<int>(std::distance(
          work.abs_errors.cbegin(),
          std::max_element(work.abs_errors.cbegin(),
                           work.abs_errors.cbegin() + iter + 1))) };
        const std::array<double, 2> bounds_cur { work.bounds[max_error_idx] };
        const double middle { (bounds_cur.back() + bounds_cur.front()) / 2 };
        work.sums[max_error_idx] = gaussKronrod(function,
                                                work.inactive_parameters,
                                                bounds_cur.front(),
                                                middle,
                                                work.abs_errors[max_error_idx]);
        work.sums[iter + 1] = gaussKronrod(function,
                                           work.inactive_parameters,
                                           middle,
                                           bounds_cur.back(),
                                           work.abs_errors[iter + 1]);
        work.bounds[max_error_idx].back() = middle;
        work.bounds[iter + 1] = { middle, bounds_cur.back() };
        const double errors_sum { std::accumulate(
          work.abs_errors.cbegin(), work.abs_errors.cbegin() + iter + 2, 0.0) };
        const double sums_sum { std::accumulate(
          work.sums.cbegin(), work.sums.cbegin() + iter + 2, 0.0) };
        if (errors_sum < abs_error || errors_sum / sums_sum < rel_error) {
            AdVar result { 0.0, passive_idx };
            for (int i {}; i < iter + 2; ++i) {
                result += kronrod(function,
                                  parameters,
                                  work.bounds[i].front(),
                                  work.bounds[i].back());
            }
            --int_order;
            return result;
        }
    }
    throw InsufficientIntegrationWorkspace {};
}

// integrand_inactive_parameters is a work array with the same values
// as 'parameters' but all the indices are inactive. It is used for
// function evaluation when AD is not required.
static auto setIntegrandInactiveParameters(const std::vector<AdVar>& parameters)
  -> void
{
    if (workspaces.front().integrand_inactive_parameters.size()
        < parameters.size()) {
        workspaces.front().integrand_inactive_parameters.resize(
          parameters.size());
        for (auto& par : workspaces.front().integrand_inactive_parameters) {
            par.idx = passive_idx;
        }
    }
    for (int i {}; i < static_cast<int>(parameters.size()); ++i) {
        workspaces.front().integrand_inactive_parameters[i].val =
          parameters[i].val;
    }
}

auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const AdVar& lower,
               const double upper,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(
      function, parameters, lower.val, upper, rel_error, abs_error) };
    if (lower.idx > passive_idx) {
        if (result.idx == passive_idx) {
            result.idx = ++reverse::last_index;
            reverse::forwards[reverse::last_index] = result.val;
        }
        setIntegrandInactiveParameters(parameters);
        reverse::constants[++reverse::const_count] =
          function(workspaces.front().integrand_inactive_parameters,
                   AdVar { lower.val, passive_idx })
            .val;
        reverse::trace[++reverse::trace_count] = lower.idx;
        reverse::trace[++reverse::trace_count] = result.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::int_lower_bound);
    } else if (lower.idx == forward_active_idx) {
        const AdVar tmp =
          function(parameters, AdVar { lower.val, passive_idx });
        result.d -= lower.d * tmp.val;
        result.dd -= lower.dd * tmp.val
                     + lower.d * (function(parameters, lower).d + tmp.d);
        if (result.idx == passive_idx) {
            result.idx = forward_active_idx;
        }
    }
    return result;
}

auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const double lower,
               const AdVar& upper,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(
      function, parameters, lower, upper.val, rel_error, abs_error) };
    if (upper.idx > passive_idx) {
        if (result.idx == passive_idx) {
            result.idx = ++reverse::last_index;
            reverse::forwards[reverse::last_index] = result.val;
        }
        setIntegrandInactiveParameters(parameters);
        reverse::constants[++reverse::const_count] =
          function(workspaces.front().integrand_inactive_parameters,
                   AdVar { upper.val, passive_idx })
            .val;
        reverse::trace[++reverse::trace_count] = upper.idx;
        reverse::trace[++reverse::trace_count] = result.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::int_upper_bound);
    } else if (upper.idx == forward_active_idx) {
        const AdVar tmp =
          function(parameters, AdVar { upper.val, passive_idx });
        result.d += upper.d * tmp.val;
        result.dd += upper.dd * tmp.val
                     + upper.d * (function(parameters, upper).d + tmp.d);
        if (result.idx == passive_idx) {
            result.idx = forward_active_idx;
        }
    }
    return result;
}

auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const AdVar& lower,
               const AdVar& upper,
               const double rel_error,
               const double abs_error) -> AdVar
{
    if (lower.idx != passive_idx && upper.idx == passive_idx) {
        return integrate(
          function, parameters, lower, upper.val, rel_error, abs_error);
    } else if (lower.idx == passive_idx && upper.idx != passive_idx) {
        return integrate(
          function, parameters, lower.val, upper, rel_error, abs_error);
    }
    AdVar result { integrate(
      function, parameters, lower.val, upper.val, rel_error, abs_error) };
    // At this point either both or neither bound is active. Thus,
    // only need to test one of them.
    if (lower.idx > passive_idx) {
        if (result.idx == passive_idx) {
            result.idx = ++reverse::last_index;
            reverse::forwards[reverse::last_index] = result.val;
        }
        setIntegrandInactiveParameters(parameters);
        reverse::constants[++reverse::const_count] =
          function(workspaces.front().integrand_inactive_parameters,
                   AdVar { lower.val, passive_idx })
            .val;
        reverse::constants[++reverse::const_count] =
          function(workspaces.front().integrand_inactive_parameters,
                   AdVar { upper.val, passive_idx })
            .val;
        reverse::trace[++reverse::trace_count] = lower.idx;
        reverse::trace[++reverse::trace_count] = upper.idx;
        reverse::trace[++reverse::trace_count] = result.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::int_both_bounds);
    } else if (lower.idx == forward_active_idx) {
        const AdVar tmp_upper { function(parameters,
                                         AdVar { upper.val, passive_idx }) };
        const AdVar tmp_lower { function(parameters,
                                         AdVar { lower.val, passive_idx }) };
        result.d += upper.d * tmp_upper.val - lower.d * tmp_lower.val;
        result.dd += upper.dd * tmp_upper.val - lower.dd * tmp_lower.val
                     + upper.d * (function(parameters, upper).d + tmp_upper.d)
                     - lower.d * (function(parameters, lower).d + tmp_lower.d);
        if (result.idx == passive_idx) {
            result.idx = forward_active_idx;
        }
    }
    return result;
}

auto resetIntegrandParameters() -> void
{
    for (auto& work : workspaces) {
        work.inactive_parameters.clear();
    }
}

auto freeIntegration() -> void
{
    gk::roots = std::vector<double> {};
    gk::weights_gauss = std::vector<double> {};
    gk::weights_kronrod = std::vector<double> {};
    gk::weights_kronrod_2D = std::vector<double> {};
    workspaces = std::vector<Workspace> {};
}

} // namespace gadfit

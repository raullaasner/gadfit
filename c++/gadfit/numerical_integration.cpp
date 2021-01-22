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
std::vector<double> weights_gauss_2D {};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<double> weights_kronrod {};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<double> weights_kronrod_2D {};

} // gk namespace

struct Workspace
{
    struct IntegrationBounds
    {
        double y1; // Lower bound of the outer integral
        double y2; // Upper bound of the outer integral
        double x1; // Lower bound of the inner integral
        double x2; // Upper bound of the inner integral
    };
    std::vector<double> sums {};
    std::vector<double> abs_errors {};
    std::vector<std::array<double, 2>> bounds {};
    std::vector<IntegrationBounds> bounds_2D {};
    std::vector<AdVar> inactive_parameters {};
    std::vector<AdVar> integrand_inactive_parameters {};
    auto resize(const int size, const int size_2D = -1) -> void
    {
        sums.resize(size);
        abs_errors.resize(size);
        // In the main adaptive integration loop below, the work
        // arrays for the bounds need to be slightly larger than for
        // the sums or errors. This is because they always record
        // information for the current and the next iteration.
        bounds.resize(size + 1);
        bounds_2D.resize(size_2D + 1);
    }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<Workspace> workspaces {};

auto initIntegration(const int workspace_size,
                     const int n_workspaces,
                     const int sweep_size) -> void
{
    // AD work arrays need to be resized to accommodate the larger
    // cost of numerical integration.
    initializeADReverse(sweep_size);
    gk::roots = std::vector<double> { roots_15p.cbegin(), roots_15p.cend() };
    gk::weights_gauss = std::vector<double> { weights_gauss_7p.cbegin(),
                                              weights_gauss_7p.cend() };
    gk::weights_kronrod = std::vector<double> { weights_kronrod_15p.cbegin(),
                                                weights_kronrod_15p.cend() };
    workspaces.resize(n_workspaces);
    for (auto& work : workspaces) {
        work.resize(workspace_size);
    }
}

auto initIntegration2D() -> void
{
    const int gauss_size { static_cast<int>(gk::weights_gauss.size()) };
    gk::weights_gauss_2D.resize(gauss_size * gauss_size);
    for (int i {}; i < gauss_size; ++i) {
        for (int j {}; j < gauss_size; ++j) {
            gk::weights_gauss_2D[i * gauss_size + j] =
              gk::weights_gauss[i] * gk::weights_gauss[j];
        }
    }
    const int kronrod_size { static_cast<int>(gk::weights_kronrod.size()) };
    gk::weights_kronrod_2D.resize(kronrod_size * kronrod_size);
    for (int i {}; i < kronrod_size; ++i) {
        for (int j {}; j < kronrod_size; ++j) {
            gk::weights_kronrod_2D[i * kronrod_size + j] =
              gk::weights_kronrod[i] * gk::weights_kronrod[j];
        }
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
        sum += gk::weights_kronrod[i] * function(parameters, arg);
    }
    return scale * sum;
}

auto gaussKronrod(const integrandSignature2D& function,
                  const std::vector<AdVar>& parameters,
                  const double outer_lower,
                  const double outer_upper,
                  const double inner_lower,
                  const double inner_upper,
                  double& abs_error) -> double
{
    const double outer_scale { (outer_upper - outer_lower) / 2 };
    const double outer_shift { (outer_lower + outer_upper) / 2 };
    const double inner_scale { (inner_upper - inner_lower) / 2 };
    const double inner_shift { (inner_lower + inner_upper) / 2 };
    double sum_kronrod {};
    double sum_gauss {};
    AdVar arg_x {};
    AdVar arg_y {};
    arg_x.idx = passive_idx;
    arg_y.idx = passive_idx;
    int i_idx_g {}; // Row index for the Gauss weights
    for (int i {}; i < static_cast<int>(gk::roots.size()); ++i) {
        const int i_idx_k { i * static_cast<int>(gk::weights_kronrod.size()) };
        if (i % 2 == 1) {
            i_idx_g = (i / 2) * static_cast<int>(gk::weights_gauss.size());
        }
        arg_x.val = inner_scale * gk::roots[i] + inner_shift;
        for (int j {}; j < static_cast<int>(gk::roots.size()); ++j) {
            arg_y.val = outer_scale * gk::roots[j] + outer_shift;
            const double val { function(parameters, arg_x, arg_y).val };
            sum_kronrod += gk::weights_kronrod_2D[i_idx_k + j] * val;
            if (i % 2 == 1 && j % 2 == 1) {
                sum_gauss += gk::weights_gauss_2D[i_idx_g + j / 2] * val;
            }
        }
    }
    sum_kronrod *= outer_scale * inner_scale;
    abs_error = std::abs(sum_kronrod - outer_scale * inner_scale * sum_gauss);
    return sum_kronrod;
}

auto kronrod(const integrandSignature2D& function,
             const std::vector<AdVar>& parameters,
             const double outer_lower,
             const double outer_upper,
             const double inner_lower,
             const double inner_upper) -> AdVar
{
    const double outer_scale { (outer_upper - outer_lower) / 2 };
    const double outer_shift { (outer_lower + outer_upper) / 2 };
    const double inner_scale { (inner_upper - inner_lower) / 2 };
    const double inner_shift { (inner_lower + inner_upper) / 2 };
    AdVar sum { 0.0, passive_idx };
    AdVar arg_x {};
    AdVar arg_y {};
    arg_x.idx = passive_idx;
    arg_y.idx = passive_idx;
    for (int i {}; i < static_cast<int>(gk::roots.size()); ++i) {
        const int i_idx { i * static_cast<int>(gk::weights_kronrod.size()) };
        arg_x.val = inner_scale * gk::roots[i] + inner_shift;
        for (int j {}; j < static_cast<int>(gk::roots.size()); ++j) {
            arg_y.val = outer_scale * gk::roots[j] + outer_shift;
            sum += gk::weights_kronrod_2D[i_idx + j]
                   * function(parameters, arg_x, arg_y);
        }
    }
    return outer_scale * inner_scale * sum;
}

// inactive_parameters is a work array with the same values as
// 'parameters' but all the indices are inactive. It is used for
// function evaluation when AD is not required. i_work specifies which
// workspace is used and is relevant only for nested integrals.
static auto setInactiveParameters(const std::vector<AdVar>& parameters,
                                  const int i_work = 0) -> void
{
    if (workspaces[i_work].inactive_parameters.size() < parameters.size()) {
        workspaces[i_work].inactive_parameters.resize(parameters.size());
        for (auto& par : workspaces[i_work].inactive_parameters) {
            par.idx = passive_idx;
        }
    }
    for (int i {}; i < static_cast<int>(parameters.size()); ++i) {
        // In order to save time we only update the values if
        // different from the previous call. However, it never hurts
        // to overwrite them. That's why the floating point comparison
        // is safe here.
        if (workspaces[i_work].inactive_parameters[i].val
            != parameters[i].val) {
            workspaces[i_work].inactive_parameters[i].val = parameters[i].val;
        }
    }
}

auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const double lower,
               const double upper,
               const double rel_error,
               const double abs_error) -> AdVar
{
    // Integration order. Usually zero but in the case of double
    // integrals, equals 0 for the outer and 1 for the inner integral.
    static int int_order { -1 };
    ++int_order;
    if (workspaces.empty()) {
        initIntegration();
    }
    auto& work { workspaces[int_order] };
    // Make a copy of the all parameters and deactivate them. The main
    // loop should be performed without AD being active. AD is only
    // turned on for the final evaluation.
    setInactiveParameters(parameters, int_order);
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
        // Let's say the subinterval with the highest error has the
        // bounds a..b. It is replaced by a..middle and a new
        // subinterval middle..b is saved as the last item in the
        // work array.
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

// This function is called in conjunction with the main integration
// function. If the lower integration bound is an active AD variable
// in the reverse mode, it records the relevant information to the AD
// execution trace. Alternatively, if active in forward mode, compute
// the first and second derivative of result accordingly.
static auto traceRecordLower(const integrandSignature& function,
                             const std::vector<AdVar>& parameters,
                             const AdVar& lower,
                             AdVar& result) -> void
{
    if (lower.idx > passive_idx) {
        setInactiveParameters(parameters);
        if (result.idx == passive_idx) {
            result.idx = ++reverse::last_index;
            reverse::forwards[reverse::last_index] = result.val;
        }
        reverse::constants[++reverse::const_count] =
          -function(workspaces.front().inactive_parameters,
                    AdVar { lower.val, passive_idx })
             .val;
        reverse::trace[++reverse::trace_count] = lower.idx;
        reverse::trace[++reverse::trace_count] = result.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::integration_bound);
    } else if (lower.idx == forward_active_idx) {
        const AdVar tmp { function(parameters,
                                   AdVar { lower.val, passive_idx }) };
        result.d -= lower.d * tmp.val;
        result.dd -= lower.dd * tmp.val
                     + lower.d * (function(parameters, lower).d + tmp.d);
        if (result.idx == passive_idx) {
            result.idx = forward_active_idx;
        }
    }
}

// Same as traceRecordLower but in regards to the upper bound.
static auto traceRecordUpper(const integrandSignature& function,
                             const std::vector<AdVar>& parameters,
                             const AdVar& upper,
                             AdVar& result) -> void
{
    if (upper.idx > passive_idx) {
        setInactiveParameters(parameters);
        if (result.idx == passive_idx) {
            result.idx = ++reverse::last_index;
            reverse::forwards[reverse::last_index] = result.val;
        }
        reverse::constants[++reverse::const_count] =
          function(workspaces.front().inactive_parameters,
                   AdVar { upper.val, passive_idx })
            .val;
        reverse::trace[++reverse::trace_count] = upper.idx;
        reverse::trace[++reverse::trace_count] = result.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::integration_bound);
    } else if (upper.idx == forward_active_idx) {
        const AdVar tmp { function(parameters,
                                   AdVar { upper.val, passive_idx }) };
        result.d += upper.d * tmp.val;
        result.dd += upper.dd * tmp.val
                     + upper.d * (function(parameters, upper).d + tmp.d);
        if (result.idx == passive_idx) {
            result.idx = forward_active_idx;
        }
    }
}

// If the lower bound is an AD variable, call the main integration and
// then deal with the bound. These operations are not coupled in any
// way.
auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const AdVar& lower,
               const double upper,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(
      function, parameters, lower.val, upper, rel_error, abs_error) };
    traceRecordLower(function, parameters, lower, result);
    return result;
}

// Same but for the upper bound
auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const double lower,
               const AdVar& upper,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(
      function, parameters, lower, upper.val, rel_error, abs_error) };
    traceRecordUpper(function, parameters, upper, result);
    return result;
}

// Same but for the case of both bounds being AD variables
auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const AdVar& lower,
               const AdVar& upper,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(
      function, parameters, lower.val, upper.val, rel_error, abs_error) };
    traceRecordLower(function, parameters, lower, result);
    traceRecordUpper(function, parameters, upper, result);
    return result;
}

// Here we deal with the added complexity of two sets of integration
// bounds (inner and double integral) but don't have to deal with the
// integration order. That is, nested integrals are not supported.
auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1,
               const double y2,
               const double x1,
               const double x2,
               const double rel_error,
               const double abs_error) -> AdVar
{
    if (workspaces.empty()) {
        initIntegration();
    }
    auto& work { workspaces.front() };
    if (work.bounds_2D.empty()) {
        initIntegration2D();
        work.bounds_2D.resize(work.bounds.size());
    }
    setInactiveParameters(parameters);
    const double y_scale_inv { 1.0 / (y2 - y1) };
    const double x_scale_inv { 1.0 / (x2 - x1) };
    work.bounds_2D.front() = { y1, y2, x1, x2 };
    work.sums.front() = { gaussKronrod(function,
                                       work.inactive_parameters,
                                       y1,
                                       y2,
                                       x1,
                                       x2,
                                       work.abs_errors.front()) };
    // The main loop is similar to the 1D case, except that when
    // creating a subinterval, we have to decide whether to split
    // along the x-axis or the y-axix.
    for (int iter {}; iter < static_cast<int>(work.sums.size()); ++iter) {
        const int next_iter { iter + 1 };
        const int max_error_idx { static_cast<int>(std::distance(
          work.abs_errors.cbegin(),
          std::max_element(work.abs_errors.cbegin(),
                           work.abs_errors.cbegin() + next_iter))) };
        const Workspace::IntegrationBounds bounds_cur {
            work.bounds_2D[max_error_idx]
        };
        // We create a subinterval by splitting the y1...y2 bounds if
        // the scaled length y2-y1 is greated than the scaled length
        // of x2-x1. Otherwise, split along the x1...x2 bounds.
        if (y_scale_inv * (bounds_cur.y2 - bounds_cur.y1)
            > x_scale_inv * (bounds_cur.x2 - bounds_cur.x1)) {
            const double middle { (bounds_cur.y1 + bounds_cur.y2) / 2 };
            work.sums[max_error_idx] =
              gaussKronrod(function,
                           work.inactive_parameters,
                           bounds_cur.y1,
                           middle,
                           bounds_cur.x1,
                           bounds_cur.x2,
                           work.abs_errors[max_error_idx]);
            work.sums[next_iter] = gaussKronrod(function,
                                                work.inactive_parameters,
                                                middle,
                                                bounds_cur.y2,
                                                bounds_cur.x1,
                                                bounds_cur.x2,
                                                work.abs_errors[next_iter]);
            work.bounds_2D[max_error_idx].y2 = middle;
            work.bounds_2D[next_iter].y1 = middle;
            work.bounds_2D[next_iter].y2 = bounds_cur.y2;
            work.bounds_2D[next_iter].x1 = bounds_cur.x1;
            work.bounds_2D[next_iter].x2 = bounds_cur.x2;
        } else {
            const double middle { (bounds_cur.x1 + bounds_cur.x2) / 2 };
            work.sums[max_error_idx] =
              gaussKronrod(function,
                           work.inactive_parameters,
                           bounds_cur.y1,
                           bounds_cur.y2,
                           bounds_cur.x1,
                           middle,
                           work.abs_errors[max_error_idx]);
            work.sums[next_iter] = gaussKronrod(function,
                                                work.inactive_parameters,
                                                bounds_cur.y1,
                                                bounds_cur.y2,
                                                middle,
                                                bounds_cur.x2,
                                                work.abs_errors[next_iter]);
            work.bounds_2D[max_error_idx].x2 = middle;
            work.bounds_2D[next_iter].y1 = bounds_cur.y1;
            work.bounds_2D[next_iter].y2 = bounds_cur.y2;
            work.bounds_2D[next_iter].x1 = middle;
            work.bounds_2D[next_iter].x2 = bounds_cur.x2;
        }
        const double errors_sum { std::accumulate(
          work.abs_errors.cbegin(), work.abs_errors.cbegin() + iter + 2, 0.0) };
        const double sums_sum { std::accumulate(
          work.sums.cbegin(), work.sums.cbegin() + iter + 2, 0.0) };
        if (errors_sum < abs_error || errors_sum / sums_sum < rel_error) {
            AdVar result { 0.0, passive_idx };
            for (int i {}; i < iter + 2; ++i) {
                result += kronrod(function,
                                  parameters,
                                  work.bounds_2D[i].y1,
                                  work.bounds_2D[i].y2,
                                  work.bounds_2D[i].x1,
                                  work.bounds_2D[i].x2);
            }
            return result;
        }
    }
    throw InsufficientIntegrationWorkspace {};
}

// This function is called in conjunction with the main integration
// function. If the lower integration bound of the outer integral is
// an active AD variable in the reverse mode, record the relevant
// information to the AD execution trace. Alternatively, if active in
// forward mode, compute the first and second derivative of result.
static auto traceRecordY1(const integrandSignature2D& function,
                          const std::vector<AdVar>& parameters,
                          const AdVar& y1,
                          const AdVar& x1,
                          const AdVar& x2,
                          const double rel_error,
                          const double abs_error,
                          AdVar& result)
{
    const auto f_fixed_y { [&](const std::vector<AdVar>& pars, const AdVar& x) {
        return function(pars, x, AdVar { y1.val, passive_idx });
    } };
    if (y1.idx > passive_idx) {
        if (result.idx == passive_idx) {
            result.idx = ++reverse::last_index;
            reverse::forwards[reverse::last_index] = result.val;
        }
        reverse::constants[++reverse::const_count] =
          -integrate(f_fixed_y,
                     workspaces.front().inactive_parameters,
                     x1.val,
                     x2.val,
                     rel_error,
                     abs_error)
             .val;
        reverse::trace[++reverse::trace_count] = y1.idx;
        reverse::trace[++reverse::trace_count] = result.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::integration_bound);
    } else if (y1.idx == forward_active_idx) {
        const AdVar f_x { integrate(f_fixed_y,
                                    workspaces.front().inactive_parameters,
                                    x1.val,
                                    x2.val,
                                    rel_error,
                                    abs_error) };
        result.d -= y1.d * f_x.val;
        const AdVar f_x2_y1 { function(workspaces.front().inactive_parameters,
                                       AdVar { x2.val, passive_idx },
                                       AdVar { y1.val, passive_idx }) };
        const AdVar f_x1_y1 { function(workspaces.front().inactive_parameters,
                                       AdVar { x1.val, passive_idx },
                                       AdVar { y1.val, passive_idx }) };
        const AdVar f_chi { integrate(
          f_fixed_y, parameters, x1.val, x2.val, rel_error, abs_error) };
        const auto f_fixed_y_active { [&](const std::vector<AdVar>& pars,
                                          const AdVar& x) {
            return function(pars, x, y1);
        } };
        const AdVar f_full { integrate(
          f_fixed_y_active, parameters, x1.val, x2.val, rel_error, abs_error) };
        // clang-format off
        result.dd -= y1.dd * f_x.val + y1.d
                     * (x2.d * f_x2_y1.val - x1.d * f_x1_y1.val + f_chi.d
                        + f_full.d);
        // clang-format on
        if (result.idx == passive_idx) {
            result.idx = forward_active_idx;
        }
    }
}

// Same but for the upper integration bound of the outer integral
static auto traceRecordY2(const integrandSignature2D& function,
                          const std::vector<AdVar>& parameters,
                          const AdVar& y2,
                          const AdVar& x1,
                          const AdVar& x2,
                          const double rel_error,
                          const double abs_error,
                          AdVar& result)
{
    const auto f_fixed_y { [&](const std::vector<AdVar>& pars, const AdVar& x) {
        return function(pars, x, AdVar { y2.val, passive_idx });
    } };
    if (y2.idx > passive_idx) {
        if (result.idx == passive_idx) {
            result.idx = ++reverse::last_index;
            reverse::forwards[reverse::last_index] = result.val;
        }
        reverse::constants[++reverse::const_count] =
          integrate(f_fixed_y,
                    workspaces.front().inactive_parameters,
                    x1.val,
                    x2.val,
                    rel_error,
                    abs_error)
            .val;
        reverse::trace[++reverse::trace_count] = y2.idx;
        reverse::trace[++reverse::trace_count] = result.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::integration_bound);
    } else if (y2.idx == forward_active_idx) {
        const AdVar f_x { integrate(f_fixed_y,
                                    workspaces.front().inactive_parameters,
                                    x1.val,
                                    x2.val,
                                    rel_error,
                                    abs_error) };
        result.d += y2.d * f_x.val;
        const AdVar f_x2_y2 { function(workspaces.front().inactive_parameters,
                                       AdVar { x2.val, passive_idx },
                                       AdVar { y2.val, passive_idx }) };
        const AdVar f_x1_y2 { function(workspaces.front().inactive_parameters,
                                       AdVar { x1.val, passive_idx },
                                       AdVar { y2.val, passive_idx }) };
        const AdVar f_chi { integrate(
          f_fixed_y, parameters, x1.val, x2.val, rel_error, abs_error) };
        const auto f_fixed_y_active { [&](const std::vector<AdVar>& pars,
                                          const AdVar& x) {
            return function(pars, x, y2);
        } };
        const AdVar f_full { integrate(
          f_fixed_y_active, parameters, x1.val, x2.val, rel_error, abs_error) };
        // clang-format off
        result.dd += y2.dd * f_x.val + y2.d
                     * (x2.d * f_x2_y2.val - x1.d * f_x1_y2.val + f_chi.d
                        + f_full.d);
        // clang-format on
        if (result.idx == passive_idx) {
            result.idx = forward_active_idx;
        }
    }
}

// Same but for the lower integration bound of the inner integral
static auto traceRecordX1(const integrandSignature2D& function,
                          const std::vector<AdVar>& parameters,
                          const AdVar& x1,
                          const AdVar& y1,
                          const AdVar& y2,
                          const double rel_error,
                          const double abs_error,
                          AdVar& result)
{
    const auto f_fixed_x { [&](const std::vector<AdVar>& pars, const AdVar& y) {
        return function(pars, AdVar { x1.val, passive_idx }, y);
    } };
    if (x1.idx > passive_idx) {
        if (result.idx == passive_idx) {
            result.idx = ++reverse::last_index;
            reverse::forwards[reverse::last_index] = result.val;
        }
        reverse::constants[++reverse::const_count] =
          -integrate(f_fixed_x,
                     workspaces.front().inactive_parameters,
                     y1.val,
                     y2.val,
                     rel_error,
                     abs_error)
             .val;
        reverse::trace[++reverse::trace_count] = x1.idx;
        reverse::trace[++reverse::trace_count] = result.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::integration_bound);
    } else if (x1.idx == forward_active_idx) {
        const AdVar f_y { integrate(f_fixed_x,
                                    workspaces.front().inactive_parameters,
                                    y1.val,
                                    y2.val,
                                    rel_error,
                                    abs_error) };
        result.d -= x1.d * f_y.val;
        const AdVar f_x1_y2 { function(workspaces.front().inactive_parameters,
                                       AdVar { x1.val, passive_idx },
                                       AdVar { y2.val, passive_idx }) };
        const AdVar f_x1_y1 { function(workspaces.front().inactive_parameters,
                                       AdVar { x1.val, passive_idx },
                                       AdVar { y1.val, passive_idx }) };
        const AdVar f_chi { integrate(
          f_fixed_x, parameters, y1.val, y2.val, rel_error, abs_error) };
        const auto f_fixed_x_active { [&](const std::vector<AdVar>& pars,
                                          const AdVar& y) {
            return function(pars, x1, y);
        } };
        const AdVar f_full { integrate(
          f_fixed_x_active, parameters, y1.val, y2.val, rel_error, abs_error) };
        // clang-format off
        result.dd -= x1.dd * f_y.val + x1.d
                     * (y2.d * f_x1_y2.val - y1.d * f_x1_y1.val + f_chi.d
                        + f_full.d);
        // clang-format on
        if (result.idx == passive_idx) {
            result.idx = forward_active_idx;
        }
    }
}

// Same but for the upper integration bound of the inner integral
static auto traceRecordX2(const integrandSignature2D& function,
                          const std::vector<AdVar>& parameters,
                          const AdVar& x2,
                          const AdVar& y1,
                          const AdVar& y2,
                          const double rel_error,
                          const double abs_error,
                          AdVar& result)
{
    const auto f_fixed_x { [&](const std::vector<AdVar>& pars, const AdVar& y) {
        return function(pars, AdVar { x2.val, passive_idx }, y);
    } };
    if (x2.idx > passive_idx) {
        if (result.idx == passive_idx) {
            result.idx = ++reverse::last_index;
            reverse::forwards[reverse::last_index] = result.val;
        }
        reverse::constants[++reverse::const_count] =
          integrate(f_fixed_x,
                    workspaces.front().inactive_parameters,
                    y1.val,
                    y2.val,
                    rel_error,
                    abs_error)
            .val;
        reverse::trace[++reverse::trace_count] = x2.idx;
        reverse::trace[++reverse::trace_count] = result.idx;
        reverse::trace[++reverse::trace_count] =
          static_cast<int>(Op::integration_bound);
    } else if (x2.idx == forward_active_idx) {
        const AdVar f_y { integrate(f_fixed_x,
                                    workspaces.front().inactive_parameters,
                                    y1.val,
                                    y2.val,
                                    rel_error,
                                    abs_error) };
        result.d += x2.d * f_y.val;
        const AdVar f_x2_y2 { function(workspaces.front().inactive_parameters,
                                       AdVar { x2.val, passive_idx },
                                       AdVar { y2.val, passive_idx }) };
        const AdVar f_x2_y1 { function(workspaces.front().inactive_parameters,
                                       AdVar { x2.val, passive_idx },
                                       AdVar { y1.val, passive_idx }) };
        const AdVar f_chi { integrate(
          f_fixed_x, parameters, y1.val, y2.val, rel_error, abs_error) };
        const auto f_fixed_x_active { [&](const std::vector<AdVar>& pars,
                                          const AdVar& y) {
            return function(pars, x2, y);
        } };
        const AdVar f_full { integrate(
          f_fixed_x_active, parameters, y1.val, y2.val, rel_error, abs_error) };
        // clang-format off
        result.dd += x2.dd * f_y.val + x2.d
                     * (y2.d * f_x2_y2.val - y1.d * f_x2_y1.val + f_chi.d
                        + f_full.d);
        // clang-format on
        if (result.idx == passive_idx) {
            result.idx = forward_active_idx;
        }
    }
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const AdVar& y2,
               const AdVar& x1,
               const AdVar& x2,
               const double rel_error,
               const double abs_error) -> AdVar

{
    AdVar result { integrate(function,
                             parameters,
                             y1.val,
                             y2.val,
                             x1.val,
                             x2.val,
                             rel_error,
                             abs_error) };
    traceRecordY1(
      function, parameters, y1, x1, x2, rel_error, abs_error, result);
    traceRecordY2(
      function, parameters, y2, x1, x2, rel_error, abs_error, result);
    traceRecordX1(
      function, parameters, x1, y1, y2, rel_error, abs_error, result);
    traceRecordX2(
      function, parameters, x2, y1, y2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const AdVar& y2,
               const AdVar& x1,
               const double x2_val,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1.val,
                             y2.val,
                             x1.val,
                             x2_val,
                             rel_error,
                             abs_error) };
    static AdVar x2 { 0.0, 0.0, 0.0, passive_idx };
    x2.val = x2_val;
    traceRecordY1(
      function, parameters, y1, x1, x2, rel_error, abs_error, result);
    traceRecordY2(
      function, parameters, y2, x1, x2, rel_error, abs_error, result);
    traceRecordX1(
      function, parameters, x1, y1, y2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const AdVar& y2,
               const double x1_val,
               const AdVar& x2,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1.val,
                             y2.val,
                             x1_val,
                             x2.val,
                             rel_error,
                             abs_error) };
    static AdVar x1 { 0.0, 0.0, 0.0, passive_idx };
    x1.val = x1_val;
    traceRecordY1(
      function, parameters, y1, x1, x2, rel_error, abs_error, result);
    traceRecordY2(
      function, parameters, y2, x1, x2, rel_error, abs_error, result);
    traceRecordX2(
      function, parameters, x2, y1, y2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const double y2_val,
               const AdVar& x1,
               const AdVar& x2,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1.val,
                             y2_val,
                             x1.val,
                             x2.val,
                             rel_error,
                             abs_error) };
    static AdVar y2 { 0.0, 0.0, 0.0, passive_idx };
    y2.val = y2_val;
    traceRecordY1(
      function, parameters, y1, x1, x2, rel_error, abs_error, result);
    traceRecordX1(
      function, parameters, x1, y1, y2, rel_error, abs_error, result);
    traceRecordX2(
      function, parameters, x2, y1, y2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const AdVar& y2,
               const AdVar& x1,
               const AdVar& x2,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1_val,
                             y2.val,
                             x1.val,
                             x2.val,
                             rel_error,
                             abs_error) };
    static AdVar y1 { 0.0, 0.0, 0.0, passive_idx };
    y1.val = y1_val;
    traceRecordY2(
      function, parameters, y2, x1, x2, rel_error, abs_error, result);
    traceRecordX1(
      function, parameters, x1, y1, y2, rel_error, abs_error, result);
    traceRecordX2(
      function, parameters, x2, y1, y2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const AdVar& y2,
               const double x1_val,
               const double x2_val,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1.val,
                             y2.val,
                             x1_val,
                             x2_val,
                             rel_error,
                             abs_error) };
    static AdVar x1 { 0.0, 0.0, 0.0, passive_idx };
    x1.val = x1_val;
    static AdVar x2 { 0.0, 0.0, 0.0, passive_idx };
    x2.val = x2_val;
    traceRecordY1(
      function, parameters, y1, x1, x2, rel_error, abs_error, result);
    traceRecordY2(
      function, parameters, y2, x1, x2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const double y2_val,
               const AdVar& x1,
               const AdVar& x2,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1_val,
                             y2_val,
                             x1.val,
                             x2.val,
                             rel_error,
                             abs_error) };
    static AdVar y1 { 0.0, 0.0, 0.0, passive_idx };
    y1.val = y1_val;
    static AdVar y2 { 0.0, 0.0, 0.0, passive_idx };
    y2.val = y2_val;
    traceRecordX1(
      function, parameters, x1, y1, y2, rel_error, abs_error, result);
    traceRecordX2(
      function, parameters, x2, y1, y2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const double y2_val,
               const double x1_val,
               const AdVar& x2,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1.val,
                             y2_val,
                             x1_val,
                             x2.val,
                             rel_error,
                             abs_error) };
    static AdVar y2 { 0.0, 0.0, 0.0, passive_idx };
    y2.val = y2_val;
    static AdVar x1 { 0.0, 0.0, 0.0, passive_idx };
    x1.val = x1_val;
    traceRecordY1(
      function, parameters, y1, x1, x2, rel_error, abs_error, result);
    traceRecordX2(
      function, parameters, x2, y1, y2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const AdVar& y2,
               const AdVar& x1,
               const double x2_val,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1_val,
                             y2.val,
                             x1.val,
                             x2_val,
                             rel_error,
                             abs_error) };
    static AdVar y1 { 0.0, 0.0, 0.0, passive_idx };
    y1.val = y1_val;
    static AdVar x2 { 0.0, 0.0, 0.0, passive_idx };
    x2.val = x2_val;
    traceRecordY2(
      function, parameters, y2, x1, x2, rel_error, abs_error, result);
    traceRecordX1(
      function, parameters, x1, y1, y2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const double y2_val,
               const AdVar& x1,
               const double x2_val,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1.val,
                             y2_val,
                             x1.val,
                             x2_val,
                             rel_error,
                             abs_error) };
    static AdVar y2 { 0.0, 0.0, 0.0, passive_idx };
    y2.val = y2_val;
    static AdVar x2 { 0.0, 0.0, 0.0, passive_idx };
    x2.val = x2_val;
    traceRecordY1(
      function, parameters, y1, x1, x2, rel_error, abs_error, result);
    traceRecordX1(
      function, parameters, x1, y1, y2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const AdVar& y2,
               const double x1_val,
               const AdVar& x2,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1_val,
                             y2.val,
                             x1_val,
                             x2.val,
                             rel_error,
                             abs_error) };
    static AdVar y1 { 0.0, 0.0, 0.0, passive_idx };
    y1.val = y1_val;
    static AdVar x1 { 0.0, 0.0, 0.0, passive_idx };
    x1.val = x1_val;
    traceRecordY2(
      function, parameters, y2, x1, x2, rel_error, abs_error, result);
    traceRecordX2(
      function, parameters, x2, y1, y2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const double y2_val,
               const double x1_val,
               const double x2_val,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1.val,
                             y2_val,
                             x1_val,
                             x2_val,
                             rel_error,
                             abs_error) };
    static AdVar x1 { 0.0, 0.0, 0.0, passive_idx };
    x1.val = x1_val;
    static AdVar x2 { 0.0, 0.0, 0.0, passive_idx };
    x2.val = x2_val;
    traceRecordY1(
      function, parameters, y1, x1, x2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const AdVar& y2,
               const double x1_val,
               const double x2_val,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1_val,
                             y2.val,
                             x1_val,
                             x2_val,
                             rel_error,
                             abs_error) };
    static AdVar x1 { 0.0, 0.0, 0.0, passive_idx };
    x1.val = x1_val;
    static AdVar x2 { 0.0, 0.0, 0.0, passive_idx };
    x2.val = x2_val;
    traceRecordY2(
      function, parameters, y2, x1, x2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const double y2_val,
               const AdVar& x1,
               const double x2_val,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1_val,
                             y2_val,
                             x1.val,
                             x2_val,
                             rel_error,
                             abs_error) };
    static AdVar y1 { 0.0, 0.0, 0.0, passive_idx };
    y1.val = y1_val;
    static AdVar y2 { 0.0, 0.0, 0.0, passive_idx };
    y2.val = y2_val;
    traceRecordX1(
      function, parameters, x1, y1, y2, rel_error, abs_error, result);
    return result;
}

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const double y2_val,
               const double x1_val,
               const AdVar& x2,
               const double rel_error,
               const double abs_error) -> AdVar
{
    AdVar result { integrate(function,
                             parameters,
                             y1_val,
                             y2_val,
                             x1_val,
                             x2.val,
                             rel_error,
                             abs_error) };
    static AdVar y1 { 0.0, 0.0, 0.0, passive_idx };
    y1.val = y1_val;
    static AdVar y2 { 0.0, 0.0, 0.0, passive_idx };
    y2.val = y2_val;
    traceRecordX2(
      function, parameters, x2, y1, y2, rel_error, abs_error, result);
    return result;
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

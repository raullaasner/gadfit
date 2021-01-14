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

// Provides functions for adaptive numerical integration. Supports 1D
// and 2D integrals. The latter can be computed either directly or as
// nested 1D integrals. With direct 2D integrals the bounds cannot
// depends on the integration variables (but dependence on any fitting
// parameters is fine). In the nested case there are no restrictions
// on the integration bounds.

#pragma once

#include "fit_function.h"

#include <functional>
#include <limits>

namespace gadfit {

using integrandSignature =
  std::function<AdVar(const std::vector<AdVar>&, const AdVar&)>;

using integrandSignature2D =
  std::function<AdVar(const std::vector<AdVar>&, const AdVar&, const AdVar&)>;

class AdVar;

// Default relative threshold for the Gauss–Kronrod quadrature error
// estimate
constexpr double default_rel_error { 1e2
                                     * std::numeric_limits<double>::epsilon() };
constexpr int default_workspace_size { 1000 };
constexpr int default_n_workspaces { 1 };
// Characteristic size of AD work arrays when numerical integration is used
constexpr int default_sweep_size_int { 1'000'000 };

// Initialize the Gauss-Kronrod rules and the work arrays for adaptive
// integration. The work arrays store things like errors for each
// subinterval. This is automatically called in 'integrate'.
auto initIntegration(const int workspace_size = default_workspace_size,
                     const int n_workspaces = default_n_workspaces,
                     const int sweep_size = default_sweep_size_int) -> void;

// Initialize additional work arrays for 2D integration. This needs to
// be called in addition to initIntegration.
auto initIntegration2D() -> void;

// One-dimensional Gauss-Kronrod integration
auto gaussKronrod(const integrandSignature& function,
                  const std::vector<AdVar>& parameters,
                  const double lower,
                  const double upper,
                  double& abs_error) -> double;

// One-dimensional integration using only the Kronrod roots and
// weights. Here we don't require the error estimate.
auto kronrod(const integrandSignature& function,
             const std::vector<AdVar>& parameters,
             const double lower,
             const double upper) -> AdVar;

// Two-dimensional Gauss-Kronrod integration
auto gaussKronrod(const integrandSignature2D& function,
                  const std::vector<AdVar>& parameters,
                  const double outer_lower,
                  const double outer_upper,
                  const double inner_lower,
                  const double inner_upper,
                  double& abs_error) -> double;

// Two-dimensional integration using only the Kronrod roots and
// weights. Here we don't require the error estimate.
auto kronrod(const integrandSignature2D& function,
             const std::vector<AdVar>& parameters,
             const double outer_lower,
             const double outer_upper,
             const double inner_lower,
             const double inner_upper) -> AdVar;

// Both bounds are ordinary numbers. This function contains the main
// integration algorithm for 1D integration.
auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const double lower,
               const double upper,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

// Here the lower bound is an AD number. It calls the main integration
// function and records additional information to the AD execution
// trace which is related to the derivative with respect to the lower
// bound.
auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const AdVar& lower,
               const double upper,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

// Same as above but the upper bound is an AD number.
auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const double lower,
               const AdVar& upper,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

// Both bounds are AD numbers.
auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const AdVar& lower,
               const AdVar& upper,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

// Double integral where all bounds are ordinary floating-point
// numbers. y1 and y2 are the integration bounds of the outer integral
// and x1 and x2 those of the inner integral. Otherwise, same comments
// as for the 1D case.
auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1,
               const double y2,
               const double x1,
               const double x2,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

// All bounds are AD numbers
auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const AdVar& y2,
               const AdVar& x1,
               const AdVar& x2,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

// The following are all possible permutations of y1 y2 x1 x2 being
// either AD variables or double precision numbers.
auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const AdVar& y2,
               const AdVar& x1,
               const double x2_val,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const AdVar& y2,
               const double x1_val,
               const AdVar& x2,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const double y2_val,
               const AdVar& x1,
               const AdVar& x2,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const AdVar& y2,
               const AdVar& x1,
               const AdVar& x2,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const AdVar& y2,
               const double x1_val,
               const double x2_val,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const double y2_val,
               const AdVar& x1,
               const AdVar& x2,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const double y2_val,
               const double x1_val,
               const AdVar& x2,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const AdVar& y2,
               const AdVar& x1,
               const double x2_val,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const double y2_val,
               const AdVar& x1,
               const double x2_val,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const AdVar& y2,
               const double x1_val,
               const AdVar& x2,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const AdVar& y1,
               const double y2_val,
               const double x1_val,
               const double x2_val,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const AdVar& y2,
               const double x1_val,
               const double x2_val,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const double y2_val,
               const AdVar& x1,
               const double x2_val,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto integrate(const integrandSignature2D& function,
               const std::vector<AdVar>& parameters,
               const double y1_val,
               const double y2_val,
               const double x1_val,
               const AdVar& x2,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

auto freeIntegration() -> void;

} // namespace gadfit

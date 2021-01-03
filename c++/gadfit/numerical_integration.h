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

// Provides functions for adaptive numerical integration.

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

constexpr double default_rel_error { 1e2
                                     * std::numeric_limits<double>::epsilon() };
constexpr int default_workspace_size { 1000 };
constexpr int default_n_workspaces { 1 };
// Characteristic size of AD work arrays when numerical integration is used
constexpr int default_sweep_size_int { 1'000'000 };

// Initialize the Gauss-Kronrod rules and the work arrays for adaptive
// integration. The work arrays store things like errors for each
// subinterval. This is called automatically in 'integrate'.
auto initIntegration(const int workspace_size = default_workspace_size,
                     const int n_workspaces = default_n_workspaces,
                     const int sweep_size = default_sweep_size_int) -> void;

// One-dimensional Gauss-Kronrod integration
auto gaussKronrod(const integrandSignature& function,
                  const std::vector<AdVar>& parameters,
                  const double lower,
                  const double upper,
                  double& abs_error) -> double;

// One-dimensional integration using only Kronrod roots and
// weights. Here we don't require the error estimate.
auto kronrod(const integrandSignature& function,
             const std::vector<AdVar>& parameters,
             const double lower,
             const double upper) -> AdVar;

// Both bounds are ordinary numbers. The values of 'parameters' are
// copied to an internal work array on the first call. Subsequent
// calls assume that 'parameters' does not change. Otherwise,
// resetParameters must be called before the next call to 'integrate'.
auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const double lower,
               const double upper,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

// Lower bound is an AD number
auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const AdVar& lower,
               const double upper,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

// Upper bound is an AD number
auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const double lower,
               const AdVar& upper,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

// Both bounds are AD numbers
auto integrate(const integrandSignature& function,
               const std::vector<AdVar>& parameters,
               const AdVar& lower,
               const AdVar& upper,
               const double rel_error = default_rel_error,
               const double abs_error = 0.0) -> AdVar;

// Call this before 'integrate' if the integrand 'parameters' have
// changed since the last call to 'integrate'.
auto resetIntegrandParameters() -> void;

auto freeIntegration() -> void;

} // namespace gadfit

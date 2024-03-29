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

#include "exceptions.h"

namespace gadfit {

UnknownOperation::UnknownOperation(const int op_code)
{
    message = "Unknown operation code during the return sweep: "
              + std::to_string(op_code);
}

// LCOV_EXCL_START
[[nodiscard]] auto UnknownOperation::what() const noexcept -> const char*
{
    return message.c_str();
}

[[nodiscard]] auto LateAddDatasetCall::what() const noexcept -> const char*
{
    return "All calls to addDataset must precede any call to setPar.";
}
// LCOV_EXCL_STOP

SetParInvalidIndex::SetParInvalidIndex(const int index)
{
    message = "Invalid value of i_dataset: " + std::to_string(index)
              + ". Use a lower value or add more data sets.";
}

// LCOV_EXCL_START
[[nodiscard]] auto SetParInvalidIndex::what() const noexcept -> const char*
{
    return message.c_str();
}

[[nodiscard]] auto UninitializedParameter::what() const noexcept -> const char*
{
    return "Not all fitting parameters have been initialized.";
}

[[nodiscard]] auto NegativeDegreesOfFreedom::what() const noexcept -> const
  char*
{
    return "More independent fitting parameters than data points";
}

[[nodiscard]] auto NoFittingParameters::what() const noexcept -> const char*
{
    return "No active fitting parameters";
}

[[nodiscard]] auto NoGlobalParameters::what() const noexcept -> const char*
{
    return "With multiple data sets must have at least one global fitting "
           "parameter. Current algorithm is not optimized for a block-diagonal "
           "JTJ matrix.";
}

[[nodiscard]] auto InsufficientIntegrationWorkspace::what() const noexcept
  -> const char*
{
    return "Number of iterations was insufficient. "
           "Increase either workspace size or the error bound(s)";
}
// LCOV_EXCL_STOP

} // namespace gadfit

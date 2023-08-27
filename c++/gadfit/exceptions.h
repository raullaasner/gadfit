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

#pragma once

#include <exception>
#include <string>

namespace gadfit {

// LCOV_EXCL_START
class GADfitException : public std::exception
{};
// LCOV_EXCL_STOP

class UnknownOperation : public GADfitException
{
private:
    std::string message;

public:
    UnknownOperation(const int op_code);
    [[nodiscard]] auto what() const noexcept -> const char* override;
};

class LateAddDatasetCall : public GADfitException
{
public:
    [[nodiscard]] auto what() const noexcept -> const char* override;
};

class SetParInvalidIndex : public GADfitException
{
private:
    std::string message;

public:
    SetParInvalidIndex(const int index);
    [[nodiscard]] auto what() const noexcept -> const char* override;
};

class UninitializedParameter : public GADfitException
{
public:
    [[nodiscard]] auto what() const noexcept -> const char* override;
};

class NegativeDegreesOfFreedom : public GADfitException
{
public:
    [[nodiscard]] auto what() const noexcept -> const char* override;
};

class NoFittingParameters : public GADfitException
{
public:
    [[nodiscard]] auto what() const noexcept -> const char* override;
};

class NoGlobalParameters : public GADfitException
{
public:
    [[nodiscard]] auto what() const noexcept -> const char* override;
};

// LCOV_EXCL_START
class InsufficientIntegrationWorkspace : public GADfitException
{
public:
    [[nodiscard]] auto what() const noexcept -> const char* override;
};
// LCOV_EXCL_STOP

} // namespace gadfit

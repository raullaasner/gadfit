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

class GADFitException : public std::exception
{};

class UnknownOperationException : public GADFitException
{
private:
    std::string message;

public:
    UnknownOperationException(const int op_code);
    [[nodiscard]] auto what() const noexcept -> const char* override;
};

class LateAddDatasetCall : public GADFitException
{
public:
    [[nodiscard]] auto what() const noexcept -> const char* override;
};

class SetParInvalidIndex : public GADFitException
{
private:
    std::string message;

public:
    SetParInvalidIndex(const int index);
    [[nodiscard]] auto what() const noexcept -> const char* override;
};

class UninitializedParameter : public GADFitException
{
public:
    [[nodiscard]] auto what() const noexcept -> const char* override;
};

class NegativeDegreesOfFreedom : public GADFitException
{
public:
    [[nodiscard]] auto what() const noexcept -> const char* override;
};

} // namespace gadfit

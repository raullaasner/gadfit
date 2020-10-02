// This Source Code Form is subject to the terms of the GNU General
// Public License, v. 3.0. If a copy of the GPL was not distributed
// with this file, You can obtain one at
// http://gnu.org/copyleft/gpl.txt.

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

class UninitializedParameter : public GADFitException
{
public:
    [[nodiscard]] auto what() const noexcept -> const char* override;
};

} // namespace gadfit

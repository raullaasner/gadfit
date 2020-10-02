// This Source Code Form is subject to the terms of the GNU General
// Public License, v. 3.0. If a copy of the GPL was not distributed
// with this file, You can obtain one at
// http://gnu.org/copyleft/gpl.txt.

#include "exceptions.h"

namespace gadfit {

UnknownOperationException::UnknownOperationException(const int op_code)
{
    message = "Unknown operation code during the return sweep: "
              + std::to_string(op_code);
}

[[nodiscard]] auto UnknownOperationException::what() const noexcept -> const
  char*
{
    return message.c_str();
}

[[nodiscard]] auto LateAddDatasetCall::what() const noexcept -> const char*
{
    return "All calls to addDataset must precede any call to setPar.";
}

[[nodiscard]] auto UninitializedParameter::what() const noexcept -> const char*
{
    return "Not all fitting parameters have been initialized.";
}

} // namespace gadfit

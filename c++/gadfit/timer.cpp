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

#include "timer.h"

namespace gadfit {

auto Timer::reset() -> void
{
    n_calls = 0;
    total_wall_time = 0.0;
    total_cpu_time = 0.0;
}

auto Timer::start() -> void
{
    wall_timestamp = std::chrono::high_resolution_clock::now();
    cpu_timestamp = std::clock();
}

auto Timer::stop() -> void
{
    total_wall_time +=
      std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - wall_timestamp)
        .count();
    total_cpu_time +=
      static_cast<double>(std::clock() - cpu_timestamp) / CLOCKS_PER_SEC;
    ++n_calls;
}

[[nodiscard]] auto Timer::totalWallTime() const -> double
{
    return total_wall_time;
}

[[nodiscard]] auto Timer::totalCPUTime() const -> double
{
    return total_cpu_time;
}

[[nodiscard]] auto Timer::totalNumberOfCalls() const -> int
{
    return n_calls;
}

} // namespace gadfit

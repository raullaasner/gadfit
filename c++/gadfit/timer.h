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

// Provides a timer class which records both the wall and CPU time and
// the number of calls to it before the next reset.

#pragma once

#include <chrono>
#include <ctime>

namespace gadfit {

class Timer
{
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> wall_timestamp;
    std::clock_t cpu_timestamp {};
    double total_wall_time {};
    double total_cpu_time {};
    int n_calls {};

public:
    Timer() = default;
    auto reset() -> void;
    auto start() -> void;
    auto stop() -> void;
    [[nodiscard]] auto totalWallTime() const -> double;
    [[nodiscard]] auto totalCPUTime() const -> double;
    [[nodiscard]] auto totalNumberOfCalls() const -> int;
};

// A handy utility struct for collecting results from different processes
struct Times
{
    double wall;
    double cpu;
    double wall_ave;
    double cpu_ave;
};

} // namespace gadfit

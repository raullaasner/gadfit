#include <array>

#include "lm_solver_data.h"

#include "lm_solver.h"

// If you don't have spdlog installed, remove all references to spdlog
// from this file. It should still run.
#include <spdlog/spdlog.h>

// This is the fitting function. Its signature is a bit hard to
// remember but it's always the same so you can just copy/paste it.
static auto exponential(const std::vector<gadfit::AdVar>& parameters,
                        const double x) -> gadfit::AdVar
{
    using gadfit::AdVar;
    const AdVar& I0 { parameters[0] };
    const AdVar& tau { parameters[1] };
    const AdVar& bgr { parameters[2] };
    return I0 * exp(-x / tau) + bgr;
}

int main()
{
    // This is not required and can be commented out.
    spdlog::set_pattern("%v");
    // The Levenberg-Marquardt class is initialized using only the
    // fitting function.
    gadfit::LMsolver solver { exponential };
    // Next, add all data sets.
    solver.addDataset(x_data_1, y_data_1);
    solver.addDataset(x_data_2, y_data_2);
    // Initial values of I0 and bgr for curve 1
    solver.setPar(0, 1.0, true, 0);
    solver.setPar(2, 1.0, true, 0);
    // Initial values of I0 and bgr for curve 2
    solver.setPar(0, 1.0, true, 1);
    solver.setPar(2, 5.5, false, 1);
    // Initial value of tau
    solver.setPar(1, 1.0, true);
    // Perform the fitting procedure
    solver.fit();
}

#pragma once

#include <cstddef>

namespace mfsi_variational_poisson {

struct SolveStats {
    double action = 0.0;
    double objective = 0.0;
    double weak_relative_residual = 0.0;
    double scaled_weak_relative_residual = 0.0;
    double gauge_residual = 0.0;
    double compatibility_residual = 0.0;
    double compatibility_relative_residual = 0.0;
    double energy_load_identity_relative_error = 0.0;
    double condition_proxy = 0.0;
    int retained_rank = 0;
    int basis_size = 0;
    int eigensolver_sweeps = 0;
    int quadrature_underflow_count = 0;
    bool converged = false;
};

void solve_batch(
    const double* log_q_mass,
    const double* forcing,
    double* potential,
    SolveStats* stats,
    int batch,
    int height,
    int width,
    double dx,
    int maximum_mode,
    double rank_relative_tolerance,
    double weak_relative_tolerance,
    double eigensolver_tolerance,
    int maximum_eigensolver_sweeps);

}  // namespace mfsi_variational_poisson

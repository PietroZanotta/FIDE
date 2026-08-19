#pragma once

#include <cstddef>

namespace mfsi_active_poisson3d {

struct SolveStats {
    int iterations = 0;
    double relative_residual = 0.0;
    bool converged = false;
};

void weighted_laplacian(
    const double* psi,
    const double* q,
    double* out,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric);

void weighted_laplacian_diag(
    const double* q,
    const double* gauge,
    double* diagonal,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric,
    double gauge_strength);

void solve_batch(
    const double* q_operator,
    const double* rhs,
    const double* gauge,
    const double* initial_guess,
    double* potential,
    SolveStats* stats,
    int batch,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric,
    double gauge_strength,
    double tolerance,
    int maximum_iterations);

void weighted_operator_vjp_batch(
    const double* potential,
    const double* adjoint,
    double* q_bar,
    int batch,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric);

void gauge_vjp_batch(
    const double* potential,
    const double* adjoint,
    const double* gauge,
    double* gauge_bar,
    int batch,
    int nx,
    int ny,
    int ntheta,
    double gauge_strength);

void linearized_rhs_batch(
    const double* potential,
    const double* q_dot,
    const double* rhs_dot,
    const double* gauge,
    const double* gauge_dot,
    double* effective_rhs,
    int batch,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric,
    double gauge_strength);

}  // namespace mfsi_active_poisson3d

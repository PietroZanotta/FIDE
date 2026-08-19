#pragma once

#include <cstddef>

namespace mfsi_unbalanced_screened3d {

struct SolveStats {
    int iterations = 0;
    double relative_residual = 0.0;
    bool converged = false;
};

void weighted_laplacian(
    const double* potential,
    const double* q,
    double* out,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric);

void screened_operator(
    const double* potential,
    const double* q,
    double* out,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric,
    double inverse_kappa);

void solve_batch(
    const double* q_operator,
    const double* rhs,
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
    double inverse_kappa,
    double tolerance,
    int maximum_iterations);

void operator_q_vjp_batch(
    const double* potential,
    const double* adjoint,
    double* q_bar,
    int batch,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric,
    double inverse_kappa);

void linearized_rhs_batch(
    const double* potential,
    const double* q_dot,
    const double* rhs_dot,
    double* effective_rhs,
    int batch,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric,
    double inverse_kappa);

}  // namespace mfsi_unbalanced_screened3d


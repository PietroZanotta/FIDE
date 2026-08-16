#pragma once

#include <cstddef>

namespace mfsi_poisson {

struct SolveStats {
    int iterations = 0;
    double relative_residual = 0.0;
    bool converged = false;
};

void weighted_laplacian(
    const double* psi,
    const double* q,
    double* out,
    int height,
    int width,
    double dx);

void weighted_laplacian_diag(
    const double* q,
    const double* gauge,
    double* diag,
    int height,
    int width,
    double dx,
    double gauge_strength);

void solve_batch(
    const double* q_operator,
    const double* rhs,
    const double* gauge,
    double* psi,
    SolveStats* stats,
    int batch,
    int height,
    int width,
    double dx,
    double gauge_strength,
    double tol,
    int maxiter);

void weighted_operator_vjp_batch(
    const double* psi,
    const double* lambda,
    double* q_bar,
    int batch,
    int height,
    int width,
    double dx);

void gauge_vjp_batch(
    const double* psi,
    const double* lambda,
    const double* gauge,
    double* gauge_bar,
    int batch,
    int height,
    int width,
    double gauge_strength);

void linearized_rhs_batch(
    const double* psi,
    const double* q_dot,
    const double* rhs_dot,
    const double* gauge,
    const double* gauge_dot,
    double* effective_rhs,
    int batch,
    int height,
    int width,
    double dx,
    double gauge_strength);

}  // namespace mfsi_poisson


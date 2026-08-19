#include "periodic_poisson3d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mfsi_active_poisson3d {
namespace {

std::size_t index3(int ix, int iy, int itheta, int ny, int ntheta) {
    return (static_cast<std::size_t>(ix) * ny + iy) * ntheta + itheta;
}

double dot(const double* left, const double* right, std::size_t size) {
    double value = 0.0;
    for (std::size_t index = 0; index < size; ++index) {
        value += left[index] * right[index];
    }
    return value;
}

void add_axis_edges(
    const double* psi,
    const double* q,
    double* out,
    int nx,
    int ny,
    int ntheta,
    int axis,
    double inverse_spacing_squared) {
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int itheta = 0; itheta < ntheta; ++itheta) {
                int jx = ix;
                int jy = iy;
                int jtheta = itheta;
                if (axis == 0) {
                    jx = (ix + 1) % nx;
                } else if (axis == 1) {
                    jy = (iy + 1) % ny;
                } else {
                    jtheta = (itheta + 1) % ntheta;
                }
                const std::size_t a = index3(ix, iy, itheta, ny, ntheta);
                const std::size_t b = index3(jx, jy, jtheta, ny, ntheta);
                const double edge = 0.5 * (q[a] + q[b]) * inverse_spacing_squared;
                const double contribution = edge * (psi[a] - psi[b]);
                out[a] += contribution;
                out[b] -= contribution;
            }
        }
    }
}

void add_axis_diagonal(
    const double* q,
    double* diagonal,
    int nx,
    int ny,
    int ntheta,
    int axis,
    double inverse_spacing_squared) {
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int itheta = 0; itheta < ntheta; ++itheta) {
                int jx = ix;
                int jy = iy;
                int jtheta = itheta;
                if (axis == 0) {
                    jx = (ix + 1) % nx;
                } else if (axis == 1) {
                    jy = (iy + 1) % ny;
                } else {
                    jtheta = (itheta + 1) % ntheta;
                }
                const std::size_t a = index3(ix, iy, itheta, ny, ntheta);
                const std::size_t b = index3(jx, jy, jtheta, ny, ntheta);
                const double edge = 0.5 * (q[a] + q[b]) * inverse_spacing_squared;
                diagonal[a] += edge;
                diagonal[b] += edge;
            }
        }
    }
}

void matvec(
    const double* potential,
    const double* q,
    const double* gauge,
    double* out,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric,
    double gauge_strength) {
    weighted_laplacian(
        potential, q, out, nx, ny, ntheta, dx, dy, dtheta_metric);
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    const double projection = dot(gauge, potential, size);
    for (std::size_t index = 0; index < size; ++index) {
        out[index] += gauge_strength * gauge[index] * projection;
    }
}

void build_incomplete_factor(
    const double* q,
    const double* gauge,
    double* lower_x,
    double* lower_y,
    double* lower_theta,
    double* diagonal,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric,
    double gauge_strength) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    std::vector<double> operator_diagonal(size);
    weighted_laplacian_diag(
        q,
        gauge,
        operator_diagonal.data(),
        nx,
        ny,
        ntheta,
        dx,
        dy,
        dtheta_metric,
        gauge_strength);
    std::fill(lower_x, lower_x + size, 0.0);
    std::fill(lower_y, lower_y + size, 0.0);
    std::fill(lower_theta, lower_theta + size, 0.0);
    const double inverse_dx2 = 1.0 / (dx * dx);
    const double inverse_dy2 = 1.0 / (dy * dy);
    const double inverse_dtheta2 = 1.0 / (dtheta_metric * dtheta_metric);

    // The periodic wrap couplings and dense rank-one gauge are omitted from the
    // sparse factor, but remain in every operator matvec. This produces a robust
    // local IC(0)-style SPD preconditioner without changing the solved equation.
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int itheta = 0; itheta < ntheta; ++itheta) {
                const std::size_t current = index3(ix, iy, itheta, ny, ntheta);
                double pivot = operator_diagonal[current];
                if (itheta > 0) {
                    const std::size_t previous = current - 1;
                    const double edge = -0.5 * (q[current] + q[previous])
                        * inverse_dtheta2;
                    lower_theta[current] = edge / diagonal[previous];
                    pivot -= lower_theta[current] * lower_theta[current];
                }
                if (iy > 0) {
                    const std::size_t previous = current - ntheta;
                    const double edge = -0.5 * (q[current] + q[previous]) * inverse_dy2;
                    lower_y[current] = edge / diagonal[previous];
                    pivot -= lower_y[current] * lower_y[current];
                }
                if (ix > 0) {
                    const std::size_t previous = current
                        - static_cast<std::size_t>(ny) * ntheta;
                    const double edge = -0.5 * (q[current] + q[previous]) * inverse_dx2;
                    lower_x[current] = edge / diagonal[previous];
                    pivot -= lower_x[current] * lower_x[current];
                }
                const double pivot_floor = std::max(
                    1.0e-24,
                    1.0e-14 * std::max(operator_diagonal[current], 1.0));
                diagonal[current] = std::sqrt(std::max(pivot, pivot_floor));
            }
        }
    }
}

void apply_incomplete_factor(
    const double* residual,
    double* output,
    double* workspace,
    const double* lower_x,
    const double* lower_y,
    const double* lower_theta,
    const double* diagonal,
    int nx,
    int ny,
    int ntheta) {
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int itheta = 0; itheta < ntheta; ++itheta) {
                const std::size_t current = index3(ix, iy, itheta, ny, ntheta);
                double value = residual[current];
                if (itheta > 0) {
                    value -= lower_theta[current] * workspace[current - 1];
                }
                if (iy > 0) {
                    value -= lower_y[current] * workspace[current - ntheta];
                }
                if (ix > 0) {
                    value -= lower_x[current]
                        * workspace[current - static_cast<std::size_t>(ny) * ntheta];
                }
                workspace[current] = value / diagonal[current];
            }
        }
    }
    for (int ix = nx - 1; ix >= 0; --ix) {
        for (int iy = ny - 1; iy >= 0; --iy) {
            for (int itheta = ntheta - 1; itheta >= 0; --itheta) {
                const std::size_t current = index3(ix, iy, itheta, ny, ntheta);
                double value = workspace[current];
                if (itheta + 1 < ntheta) {
                    value -= lower_theta[current + 1] * output[current + 1];
                }
                if (iy + 1 < ny) {
                    value -= lower_y[current + ntheta] * output[current + ntheta];
                }
                if (ix + 1 < nx) {
                    value -= lower_x[current + static_cast<std::size_t>(ny) * ntheta]
                        * output[current + static_cast<std::size_t>(ny) * ntheta];
                }
                output[current] = value / diagonal[current];
            }
        }
    }
}

SolveStats solve_one(
    const double* q,
    const double* rhs,
    const double* gauge,
    const double* initial_guess,
    double* solution,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric,
    double gauge_strength,
    double tolerance,
    int maximum_iterations) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    std::vector<double> residual(size);
    std::vector<double> preconditioned(size);
    std::vector<double> direction(size);
    std::vector<double> operator_direction(size);
    std::vector<double> factor_workspace(size);
    std::vector<double> lower_x(size);
    std::vector<double> lower_y(size);
    std::vector<double> lower_theta(size);
    std::vector<double> factor_diagonal(size);

    if (initial_guess == nullptr) {
        std::fill(solution, solution + size, 0.0);
        std::copy(rhs, rhs + size, residual.begin());
    } else {
        std::copy(initial_guess, initial_guess + size, solution);
        matvec(
            solution,
            q,
            gauge,
            operator_direction.data(),
            nx,
            ny,
            ntheta,
            dx,
            dy,
            dtheta_metric,
            gauge_strength);
        for (std::size_t index = 0; index < size; ++index) {
            residual[index] = rhs[index] - operator_direction[index];
        }
    }

    const double rhs_norm = std::sqrt(dot(rhs, rhs, size));
    const double scale = std::max(rhs_norm, std::numeric_limits<double>::min());
    if (rhs_norm <= std::numeric_limits<double>::min()) {
        std::fill(solution, solution + size, 0.0);
        return {0, 0.0, true};
    }
    const double initial_relative_residual
        = std::sqrt(dot(residual.data(), residual.data(), size)) / scale;
    if (initial_relative_residual <= tolerance) {
        return {0, initial_relative_residual, true};
    }

    build_incomplete_factor(
        q,
        gauge,
        lower_x.data(),
        lower_y.data(),
        lower_theta.data(),
        factor_diagonal.data(),
        nx,
        ny,
        ntheta,
        dx,
        dy,
        dtheta_metric,
        gauge_strength);
    apply_incomplete_factor(
        residual.data(),
        preconditioned.data(),
        factor_workspace.data(),
        lower_x.data(),
        lower_y.data(),
        lower_theta.data(),
        factor_diagonal.data(),
        nx,
        ny,
        ntheta);
    direction = preconditioned;
    double residual_preconditioned = dot(
        residual.data(), preconditioned.data(), size);
    SolveStats result{0, initial_relative_residual, false};

    for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
        matvec(
            direction.data(),
            q,
            gauge,
            operator_direction.data(),
            nx,
            ny,
            ntheta,
            dx,
            dy,
            dtheta_metric,
            gauge_strength);
        const double denominator = dot(
            direction.data(), operator_direction.data(), size);
        if (!(denominator > 0.0) || !std::isfinite(denominator)
            || !std::isfinite(residual_preconditioned)) {
            break;
        }
        const double alpha = residual_preconditioned / denominator;
        for (std::size_t index = 0; index < size; ++index) {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * operator_direction[index];
        }
        result.iterations = iteration + 1;
        result.relative_residual = std::sqrt(
            dot(residual.data(), residual.data(), size)) / scale;
        const bool replace_residual = result.relative_residual <= tolerance;
        if (replace_residual) {
            // Recursive CG residuals can lose agreement with b-Ax on difficult
            // variable-coefficient systems. Recompute the true residual before
            // declaring convergence, then restart only if more work is required.
            matvec(
                solution,
                q,
                gauge,
                operator_direction.data(),
                nx,
                ny,
                ntheta,
                dx,
                dy,
                dtheta_metric,
                gauge_strength);
            for (std::size_t index = 0; index < size; ++index) {
                residual[index] = rhs[index] - operator_direction[index];
            }
            result.relative_residual = std::sqrt(
                dot(residual.data(), residual.data(), size)) / scale;
            if (result.relative_residual <= tolerance) {
                result.converged = true;
                break;
            }
            apply_incomplete_factor(
                residual.data(),
                preconditioned.data(),
                factor_workspace.data(),
                lower_x.data(),
                lower_y.data(),
                lower_theta.data(),
                factor_diagonal.data(),
                nx,
                ny,
                ntheta);
            residual_preconditioned = dot(
                residual.data(), preconditioned.data(), size);
            if (!(residual_preconditioned > 0.0)
                || !std::isfinite(residual_preconditioned)) {
                break;
            }
            direction = preconditioned;
            continue;
        }
        apply_incomplete_factor(
            residual.data(),
            preconditioned.data(),
            factor_workspace.data(),
            lower_x.data(),
            lower_y.data(),
            lower_theta.data(),
            factor_diagonal.data(),
            nx,
            ny,
            ntheta);
        const double next = dot(residual.data(), preconditioned.data(), size);
        if (!std::isfinite(next)) {
            break;
        }
        const double beta = next / residual_preconditioned;
        for (std::size_t index = 0; index < size; ++index) {
            direction[index] = preconditioned[index] + beta * direction[index];
        }
        residual_preconditioned = next;
    }

    // Recompute and report the actual operator residual.
    matvec(
        solution,
        q,
        gauge,
        operator_direction.data(),
        nx,
        ny,
        ntheta,
        dx,
        dy,
        dtheta_metric,
        gauge_strength);
    for (std::size_t index = 0; index < size; ++index) {
        operator_direction[index] -= rhs[index];
    }
    result.relative_residual = std::sqrt(
        dot(operator_direction.data(), operator_direction.data(), size)) / scale;
    result.converged = std::isfinite(result.relative_residual)
        && result.relative_residual
            <= std::max(tolerance * 1.1, tolerance + 1.0e-14);
    return result;
}

void add_axis_vjp(
    const double* potential,
    const double* adjoint,
    double* out,
    int nx,
    int ny,
    int ntheta,
    int axis,
    double factor) {
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int itheta = 0; itheta < ntheta; ++itheta) {
                int jx = ix;
                int jy = iy;
                int jtheta = itheta;
                if (axis == 0) {
                    jx = (ix + 1) % nx;
                } else if (axis == 1) {
                    jy = (iy + 1) % ny;
                } else {
                    jtheta = (itheta + 1) % ntheta;
                }
                const std::size_t a = index3(ix, iy, itheta, ny, ntheta);
                const std::size_t b = index3(jx, jy, jtheta, ny, ntheta);
                const double contribution = factor
                    * (adjoint[a] - adjoint[b])
                    * (potential[a] - potential[b]);
                out[a] += contribution;
                out[b] += contribution;
            }
        }
    }
}

}  // namespace

void weighted_laplacian(
    const double* psi,
    const double* q,
    double* out,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    std::fill(out, out + size, 0.0);
    add_axis_edges(psi, q, out, nx, ny, ntheta, 0, 1.0 / (dx * dx));
    add_axis_edges(psi, q, out, nx, ny, ntheta, 1, 1.0 / (dy * dy));
    add_axis_edges(
        psi, q, out, nx, ny, ntheta, 2, 1.0 / (dtheta_metric * dtheta_metric));
}

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
    double gauge_strength) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    for (std::size_t index = 0; index < size; ++index) {
        diagonal[index] = gauge_strength * gauge[index] * gauge[index];
    }
    add_axis_diagonal(q, diagonal, nx, ny, ntheta, 0, 1.0 / (dx * dx));
    add_axis_diagonal(q, diagonal, nx, ny, ntheta, 1, 1.0 / (dy * dy));
    add_axis_diagonal(
        q,
        diagonal,
        nx,
        ny,
        ntheta,
        2,
        1.0 / (dtheta_metric * dtheta_metric));
}

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
    int maximum_iterations) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    #pragma omp parallel for schedule(static)
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        const std::size_t offset = static_cast<std::size_t>(batch_index) * size;
        stats[batch_index] = solve_one(
            q_operator + offset,
            rhs + offset,
            gauge + offset,
            initial_guess == nullptr ? nullptr : initial_guess + offset,
            potential + offset,
            nx,
            ny,
            ntheta,
            dx,
            dy,
            dtheta_metric,
            gauge_strength,
            tolerance,
            maximum_iterations);
    }
}

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
    double dtheta_metric) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    #pragma omp parallel for schedule(static)
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        const std::size_t offset = static_cast<std::size_t>(batch_index) * size;
        double* out = q_bar + offset;
        std::fill(out, out + size, 0.0);
        add_axis_vjp(
            potential + offset,
            adjoint + offset,
            out,
            nx,
            ny,
            ntheta,
            0,
            -0.5 / (dx * dx));
        add_axis_vjp(
            potential + offset,
            adjoint + offset,
            out,
            nx,
            ny,
            ntheta,
            1,
            -0.5 / (dy * dy));
        add_axis_vjp(
            potential + offset,
            adjoint + offset,
            out,
            nx,
            ny,
            ntheta,
            2,
            -0.5 / (dtheta_metric * dtheta_metric));
    }
}

void gauge_vjp_batch(
    const double* potential,
    const double* adjoint,
    const double* gauge,
    double* gauge_bar,
    int batch,
    int nx,
    int ny,
    int ntheta,
    double gauge_strength) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    #pragma omp parallel for schedule(static)
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        const std::size_t offset = static_cast<std::size_t>(batch_index) * size;
        const double gauge_dot_potential = dot(gauge + offset, potential + offset, size);
        const double gauge_dot_adjoint = dot(gauge + offset, adjoint + offset, size);
        for (std::size_t index = 0; index < size; ++index) {
            gauge_bar[offset + index] = -gauge_strength * (
                adjoint[offset + index] * gauge_dot_potential
                + potential[offset + index] * gauge_dot_adjoint);
        }
    }
}

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
    double gauge_strength) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    #pragma omp parallel for schedule(static)
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        const std::size_t offset = static_cast<std::size_t>(batch_index) * size;
        weighted_laplacian(
            potential + offset,
            q_dot + offset,
            effective_rhs + offset,
            nx,
            ny,
            ntheta,
            dx,
            dy,
            dtheta_metric);
        const double gauge_dot_potential = dot(gauge + offset, potential + offset, size);
        const double gauge_tangent_dot_potential = dot(
            gauge_dot + offset, potential + offset, size);
        for (std::size_t index = 0; index < size; ++index) {
            const double operator_tangent = effective_rhs[offset + index]
                + gauge_strength * (
                    gauge_dot[offset + index] * gauge_dot_potential
                    + gauge[offset + index] * gauge_tangent_dot_potential);
            effective_rhs[offset + index] = rhs_dot[offset + index] - operator_tangent;
        }
    }
}

}  // namespace mfsi_active_poisson3d

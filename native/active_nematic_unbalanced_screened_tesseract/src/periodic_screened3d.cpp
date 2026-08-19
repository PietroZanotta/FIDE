#include "periodic_screened3d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mfsi_unbalanced_screened3d {
namespace {

inline std::size_t index3(int ix, int iy, int itheta, int ny, int ntheta) {
    return (static_cast<std::size_t>(ix) * static_cast<std::size_t>(ny)
            + static_cast<std::size_t>(iy))
        * static_cast<std::size_t>(ntheta) + static_cast<std::size_t>(itheta);
}

double dot(const double* left, const double* right, std::size_t size) {
    double value = 0.0;
    #pragma omp simd reduction(+:value)
    for (std::size_t i = 0; i < size; ++i) {
        value += left[i] * right[i];
    }
    return value;
}

void add_axis_edges(
    const double* potential,
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
                int jt = itheta;
                if (axis == 0) {
                    jx = (ix + 1) % nx;
                } else if (axis == 1) {
                    jy = (iy + 1) % ny;
                } else {
                    jt = (itheta + 1) % ntheta;
                }
                const std::size_t a = index3(ix, iy, itheta, ny, ntheta);
                const std::size_t b = index3(jx, jy, jt, ny, ntheta);
                const double edge = 0.5 * (q[a] + q[b]) * inverse_spacing_squared;
                const double flux = edge * (potential[a] - potential[b]);
                out[a] += flux;
                out[b] -= flux;
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
                int jt = itheta;
                if (axis == 0) {
                    jx = (ix + 1) % nx;
                } else if (axis == 1) {
                    jy = (iy + 1) % ny;
                } else {
                    jt = (itheta + 1) % ntheta;
                }
                const std::size_t a = index3(ix, iy, itheta, ny, ntheta);
                const std::size_t b = index3(jx, jy, jt, ny, ntheta);
                const double edge = 0.5 * (q[a] + q[b]) * inverse_spacing_squared;
                diagonal[a] += edge;
                diagonal[b] += edge;
            }
        }
    }
}

void build_incomplete_factor(
    const double* q,
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
    double inverse_kappa) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    std::vector<double> operator_diagonal(size);
    for (std::size_t i = 0; i < size; ++i) {
        operator_diagonal[i] = inverse_kappa * q[i];
    }
    add_axis_diagonal(q, operator_diagonal.data(), nx, ny, ntheta, 0, 1.0 / (dx * dx));
    add_axis_diagonal(q, operator_diagonal.data(), nx, ny, ntheta, 1, 1.0 / (dy * dy));
    add_axis_diagonal(
        q, operator_diagonal.data(), nx, ny, ntheta, 2,
        1.0 / (dtheta_metric * dtheta_metric));
    std::fill(lower_x, lower_x + size, 0.0);
    std::fill(lower_y, lower_y + size, 0.0);
    std::fill(lower_theta, lower_theta + size, 0.0);
    const double inverse_dx2 = 1.0 / (dx * dx);
    const double inverse_dy2 = 1.0 / (dy * dy);
    const double inverse_dt2 = 1.0 / (dtheta_metric * dtheta_metric);

    // IC(0) retains local negative-neighbour couplings. Periodic wrap edges are
    // intentionally omitted from the factor but remain in every exact matvec.
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int itheta = 0; itheta < ntheta; ++itheta) {
                const std::size_t current = index3(ix, iy, itheta, ny, ntheta);
                double pivot = operator_diagonal[current];
                if (itheta > 0) {
                    const std::size_t previous = current - 1;
                    const double edge = -0.5 * (q[current] + q[previous]) * inverse_dt2;
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
                const double floor = std::max(
                    1.0e-24,
                    1.0e-14 * std::max(operator_diagonal[current], 1.0));
                diagonal[current] = std::sqrt(std::max(pivot, floor));
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
                if (itheta > 0) value -= lower_theta[current] * workspace[current - 1];
                if (iy > 0) value -= lower_y[current] * workspace[current - ntheta];
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
                if (itheta + 1 < ntheta) value -= lower_theta[current + 1] * output[current + 1];
                if (iy + 1 < ny) value -= lower_y[current + ntheta] * output[current + ntheta];
                if (ix + 1 < nx) {
                    const std::size_t next = current + static_cast<std::size_t>(ny) * ntheta;
                    value -= lower_x[next] * output[next];
                }
                output[current] = value / diagonal[current];
            }
        }
    }
}

SolveStats solve_one(
    const double* q,
    const double* rhs,
    const double* initial_guess,
    double* solution,
    int nx,
    int ny,
    int ntheta,
    double dx,
    double dy,
    double dtheta_metric,
    double inverse_kappa,
    double tolerance,
    int maximum_iterations) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    std::vector<double> residual(size), preconditioned(size), direction(size);
    std::vector<double> operator_direction(size), factor_workspace(size);
    std::vector<double> lower_x(size), lower_y(size), lower_theta(size), factor_diagonal(size);

    if (initial_guess == nullptr) {
        std::fill(solution, solution + size, 0.0);
        std::copy(rhs, rhs + size, residual.begin());
    } else {
        std::copy(initial_guess, initial_guess + size, solution);
        screened_operator(
            solution, q, operator_direction.data(), nx, ny, ntheta,
            dx, dy, dtheta_metric, inverse_kappa);
        for (std::size_t i = 0; i < size; ++i) residual[i] = rhs[i] - operator_direction[i];
    }

    const double rhs_norm = std::sqrt(dot(rhs, rhs, size));
    const double scale = std::max(rhs_norm, std::numeric_limits<double>::min());
    if (rhs_norm <= std::numeric_limits<double>::min()) {
        std::fill(solution, solution + size, 0.0);
        return {0, 0.0, true};
    }
    double relative = std::sqrt(dot(residual.data(), residual.data(), size)) / scale;
    if (relative <= tolerance) return {0, relative, true};

    build_incomplete_factor(
        q, lower_x.data(), lower_y.data(), lower_theta.data(), factor_diagonal.data(),
        nx, ny, ntheta, dx, dy, dtheta_metric, inverse_kappa);
    apply_incomplete_factor(
        residual.data(), preconditioned.data(), factor_workspace.data(),
        lower_x.data(), lower_y.data(), lower_theta.data(), factor_diagonal.data(),
        nx, ny, ntheta);
    direction = preconditioned;
    double residual_preconditioned = dot(residual.data(), preconditioned.data(), size);
    SolveStats result{0, relative, false};

    for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
        screened_operator(
            direction.data(), q, operator_direction.data(), nx, ny, ntheta,
            dx, dy, dtheta_metric, inverse_kappa);
        const double denominator = dot(direction.data(), operator_direction.data(), size);
        if (!(denominator > 0.0) || !(residual_preconditioned > 0.0)
            || !std::isfinite(denominator) || !std::isfinite(residual_preconditioned)) break;
        const double alpha = residual_preconditioned / denominator;
        #pragma omp simd
        for (std::size_t i = 0; i < size; ++i) {
            solution[i] += alpha * direction[i];
            residual[i] -= alpha * operator_direction[i];
        }
        result.iterations = iteration + 1;
        relative = std::sqrt(dot(residual.data(), residual.data(), size)) / scale;

        // Confirm the true residual at convergence and periodically control
        // recursive-residual drift on strongly variable density fields.
        const bool refresh = relative <= tolerance || ((iteration + 1) % 50 == 0);
        if (refresh) {
            screened_operator(
                solution, q, operator_direction.data(), nx, ny, ntheta,
                dx, dy, dtheta_metric, inverse_kappa);
            #pragma omp simd
            for (std::size_t i = 0; i < size; ++i) residual[i] = rhs[i] - operator_direction[i];
            relative = std::sqrt(dot(residual.data(), residual.data(), size)) / scale;
            if (relative <= tolerance) {
                result.relative_residual = relative;
                result.converged = true;
                break;
            }
        }
        apply_incomplete_factor(
            residual.data(), preconditioned.data(), factor_workspace.data(),
            lower_x.data(), lower_y.data(), lower_theta.data(), factor_diagonal.data(),
            nx, ny, ntheta);
        const double next = dot(residual.data(), preconditioned.data(), size);
        if (!(next > 0.0) || !std::isfinite(next)) break;
        if (refresh) {
            direction = preconditioned;
        } else {
            const double beta = next / residual_preconditioned;
            #pragma omp simd
            for (std::size_t i = 0; i < size; ++i) {
                direction[i] = preconditioned[i] + beta * direction[i];
            }
        }
        residual_preconditioned = next;
        result.relative_residual = relative;
    }

    screened_operator(
        solution, q, operator_direction.data(), nx, ny, ntheta,
        dx, dy, dtheta_metric, inverse_kappa);
    #pragma omp simd
    for (std::size_t i = 0; i < size; ++i) operator_direction[i] -= rhs[i];
    result.relative_residual = std::sqrt(dot(operator_direction.data(), operator_direction.data(), size)) / scale;
    result.converged = std::isfinite(result.relative_residual)
        && result.relative_residual <= std::max(tolerance * 1.1, tolerance + 1.0e-14);
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
                int jt = itheta;
                if (axis == 0) jx = (ix + 1) % nx;
                else if (axis == 1) jy = (iy + 1) % ny;
                else jt = (itheta + 1) % ntheta;
                const std::size_t a = index3(ix, iy, itheta, ny, ntheta);
                const std::size_t b = index3(jx, jy, jt, ny, ntheta);
                const double value = factor * (adjoint[a] - adjoint[b])
                    * (potential[a] - potential[b]);
                out[a] += value;
                out[b] += value;
            }
        }
    }
}

}  // namespace

void weighted_laplacian(
    const double* potential, const double* q, double* out,
    int nx, int ny, int ntheta, double dx, double dy, double dtheta_metric) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    std::fill(out, out + size, 0.0);
    add_axis_edges(potential, q, out, nx, ny, ntheta, 0, 1.0 / (dx * dx));
    add_axis_edges(potential, q, out, nx, ny, ntheta, 1, 1.0 / (dy * dy));
    add_axis_edges(
        potential, q, out, nx, ny, ntheta, 2,
        1.0 / (dtheta_metric * dtheta_metric));
}

void screened_operator(
    const double* potential, const double* q, double* out,
    int nx, int ny, int ntheta, double dx, double dy, double dtheta_metric,
    double inverse_kappa) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    weighted_laplacian(potential, q, out, nx, ny, ntheta, dx, dy, dtheta_metric);
    #pragma omp simd
    for (std::size_t i = 0; i < size; ++i) out[i] += inverse_kappa * q[i] * potential[i];
}

void solve_batch(
    const double* q_operator, const double* rhs, const double* initial_guess,
    double* potential, SolveStats* stats, int batch, int nx, int ny, int ntheta,
    double dx, double dy, double dtheta_metric, double inverse_kappa,
    double tolerance, int maximum_iterations) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch; ++b) {
        const std::size_t offset = static_cast<std::size_t>(b) * size;
        stats[b] = solve_one(
            q_operator + offset, rhs + offset,
            initial_guess == nullptr ? nullptr : initial_guess + offset,
            potential + offset, nx, ny, ntheta, dx, dy, dtheta_metric,
            inverse_kappa, tolerance, maximum_iterations);
    }
}

void operator_q_vjp_batch(
    const double* potential, const double* adjoint, double* q_bar,
    int batch, int nx, int ny, int ntheta, double dx, double dy,
    double dtheta_metric, double inverse_kappa) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch; ++b) {
        const std::size_t offset = static_cast<std::size_t>(b) * size;
        double* out = q_bar + offset;
        for (std::size_t i = 0; i < size; ++i) {
            out[i] = -inverse_kappa * adjoint[offset + i] * potential[offset + i];
        }
        add_axis_vjp(potential + offset, adjoint + offset, out, nx, ny, ntheta, 0, -0.5 / (dx * dx));
        add_axis_vjp(potential + offset, adjoint + offset, out, nx, ny, ntheta, 1, -0.5 / (dy * dy));
        add_axis_vjp(
            potential + offset, adjoint + offset, out, nx, ny, ntheta, 2,
            -0.5 / (dtheta_metric * dtheta_metric));
    }
}

void linearized_rhs_batch(
    const double* potential, const double* q_dot, const double* rhs_dot,
    double* effective_rhs, int batch, int nx, int ny, int ntheta,
    double dx, double dy, double dtheta_metric, double inverse_kappa) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny * ntheta;
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch; ++b) {
        const std::size_t offset = static_cast<std::size_t>(b) * size;
        screened_operator(
            potential + offset, q_dot + offset, effective_rhs + offset,
            nx, ny, ntheta, dx, dy, dtheta_metric, inverse_kappa);
        #pragma omp simd
        for (std::size_t i = 0; i < size; ++i) {
            effective_rhs[offset + i] = rhs_dot[offset + i] - effective_rhs[offset + i];
        }
    }
}

}  // namespace mfsi_unbalanced_screened3d

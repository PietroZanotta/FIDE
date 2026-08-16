#include "poisson_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mfsi_poisson {
namespace {

double dot(const double* x, const double* y, std::size_t n) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result += x[i] * y[i];
    }
    return result;
}

void matvec(
    const double* psi,
    const double* q,
    const double* gauge,
    double* out,
    int height,
    int width,
    double dx,
    double gauge_strength) {
    weighted_laplacian(psi, q, out, height, width, dx);
    const std::size_t size = static_cast<std::size_t>(height) * width;
    const double projection = dot(gauge, psi, size);
    for (std::size_t i = 0; i < size; ++i) {
        out[i] += gauge_strength * gauge[i] * projection;
    }
}

void build_ic0(
    const double* q,
    const double* gauge,
    double* lower_left,
    double* lower_up,
    double* diagonal,
    int height,
    int width,
    double dx,
    double gauge_strength) {
    const std::size_t size = static_cast<std::size_t>(height) * width;
    std::vector<double> operator_diagonal(size);
    weighted_laplacian_diag(
        q,
        gauge,
        operator_diagonal.data(),
        height,
        width,
        dx,
        gauge_strength);
    std::fill(lower_left, lower_left + size, 0.0);
    std::fill(lower_up, lower_up + size, 0.0);
    const double inv_dx2 = 1.0 / (dx * dx);

    // Zero-fill incomplete Cholesky of the five-point diffusion stencil.  The
    // dense gauge rank-one term is represented on the diagonal; omitting its
    // off-diagonal fill keeps the factor sparse while preserving an SPD
    // preconditioner for the gauge-regularized M-matrix.
    for (int i = 0; i < height; ++i) {
        const std::size_t row = static_cast<std::size_t>(i) * width;
        for (int j = 0; j < width; ++j) {
            const std::size_t index = row + j;
            double pivot = operator_diagonal[index];
            if (j > 0) {
                const std::size_t left = index - 1;
                const double edge = -0.5 * (q[index] + q[left]) * inv_dx2;
                lower_left[index] = edge / diagonal[left];
                pivot -= lower_left[index] * lower_left[index];
            }
            if (i > 0) {
                const std::size_t up = index - width;
                const double edge = -0.5 * (q[index] + q[up]) * inv_dx2;
                lower_up[index] = edge / diagonal[up];
                pivot -= lower_up[index] * lower_up[index];
            }
            const double floor = std::max(
                1.0e-24,
                1.0e-14 * std::max(operator_diagonal[index], 1.0));
            diagonal[index] = std::sqrt(std::max(pivot, floor));
        }
    }
}

void apply_ic0(
    const double* residual,
    double* output,
    double* workspace,
    const double* lower_left,
    const double* lower_up,
    const double* diagonal,
    int height,
    int width) {
    // Forward substitution: L y = r.
    for (int i = 0; i < height; ++i) {
        const std::size_t row = static_cast<std::size_t>(i) * width;
        for (int j = 0; j < width; ++j) {
            const std::size_t index = row + j;
            double value = residual[index];
            if (j > 0) {
                value -= lower_left[index] * workspace[index - 1];
            }
            if (i > 0) {
                value -= lower_up[index] * workspace[index - width];
            }
            workspace[index] = value / diagonal[index];
        }
    }

    // Back substitution: L^T z = y.
    for (int i = height - 1; i >= 0; --i) {
        const std::size_t row = static_cast<std::size_t>(i) * width;
        for (int j = width - 1; j >= 0; --j) {
            const std::size_t index = row + j;
            double value = workspace[index];
            if (j + 1 < width) {
                value -= lower_left[index + 1] * output[index + 1];
            }
            if (i + 1 < height) {
                value -= lower_up[index + width] * output[index + width];
            }
            output[index] = value / diagonal[index];
        }
    }
}

SolveStats pcg_one_system(
    const double* q,
    const double* rhs,
    const double* gauge,
    double* x,
    int height,
    int width,
    double dx,
    double gauge_strength,
    double tol,
    int maxiter) {
    const std::size_t size = static_cast<std::size_t>(height) * width;
    std::vector<double> r(size);
    std::vector<double> z(size);
    std::vector<double> p(size);
    std::vector<double> ap(size);
    std::vector<double> ic_workspace(size);
    std::vector<double> ic_lower_left(size);
    std::vector<double> ic_lower_up(size);
    std::vector<double> ic_diagonal(size);

    std::fill(x, x + size, 0.0);
    std::copy(rhs, rhs + size, r.begin());
    build_ic0(
        q,
        gauge,
        ic_lower_left.data(),
        ic_lower_up.data(),
        ic_diagonal.data(),
        height,
        width,
        dx,
        gauge_strength);

    const double rhs_norm = std::sqrt(dot(rhs, rhs, size));
    const double scale = std::max(rhs_norm, std::numeric_limits<double>::min());
    if (rhs_norm <= std::numeric_limits<double>::min()) {
        return {0, 0.0, true};
    }

    apply_ic0(
        r.data(),
        z.data(),
        ic_workspace.data(),
        ic_lower_left.data(),
        ic_lower_up.data(),
        ic_diagonal.data(),
        height,
        width);
    p = z;
    double rz = dot(r.data(), z.data(), size);
    SolveStats result{0, 1.0, false};

    for (int iteration = 0; iteration < maxiter; ++iteration) {
        matvec(p.data(), q, gauge, ap.data(), height, width, dx, gauge_strength);
        const double denominator = dot(p.data(), ap.data(), size);
        if (!(denominator > 0.0) || !std::isfinite(denominator) || !std::isfinite(rz)) {
            break;
        }
        const double alpha = rz / denominator;
        for (std::size_t i = 0; i < size; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        result.iterations = iteration + 1;
        result.relative_residual = std::sqrt(dot(r.data(), r.data(), size)) / scale;
        if (result.relative_residual <= tol) {
            result.converged = true;
            break;
        }

        apply_ic0(
            r.data(),
            z.data(),
            ic_workspace.data(),
            ic_lower_left.data(),
            ic_lower_up.data(),
            ic_diagonal.data(),
            height,
            width);
        const double next_rz = dot(r.data(), z.data(), size);
        if (!std::isfinite(next_rz)) {
            break;
        }
        const double beta = next_rz / rz;
        for (std::size_t i = 0; i < size; ++i) {
            p[i] = z[i] + beta * p[i];
        }
        rz = next_rz;
    }

    // Report the true residual, not only the recursively updated CG residual.
    matvec(x, q, gauge, ap.data(), height, width, dx, gauge_strength);
    for (std::size_t i = 0; i < size; ++i) {
        ap[i] -= rhs[i];
    }
    result.relative_residual = std::sqrt(dot(ap.data(), ap.data(), size)) / scale;
    // Recursive and explicitly recomputed residuals differ by a few ulps after
    // many iterations.  Keep the requested stopping test in-loop and allow only
    // a small reporting factor for that roundoff discrepancy.
    result.converged = result.converged && std::isfinite(result.relative_residual)
        && result.relative_residual <= std::max(tol * 1.1, tol + 1.0e-14);
    return result;
}

}  // namespace

void weighted_laplacian(
    const double* psi,
    const double* q,
    double* out,
    int height,
    int width,
    double dx) {
    const std::size_t size = static_cast<std::size_t>(height) * width;
    std::fill(out, out + size, 0.0);
    const double inv_dx2 = 1.0 / (dx * dx);

    for (int i = 0; i < height; ++i) {
        const std::size_t row = static_cast<std::size_t>(i) * width;
        for (int j = 0; j + 1 < width; ++j) {
            const std::size_t a = row + j;
            const std::size_t b = a + 1;
            const double edge = 0.5 * (q[a] + q[b]) * inv_dx2;
            const double contribution = edge * (psi[a] - psi[b]);
            out[a] += contribution;
            out[b] -= contribution;
        }
    }
    for (int i = 0; i + 1 < height; ++i) {
        const std::size_t row = static_cast<std::size_t>(i) * width;
        for (int j = 0; j < width; ++j) {
            const std::size_t a = row + j;
            const std::size_t b = a + width;
            const double edge = 0.5 * (q[a] + q[b]) * inv_dx2;
            const double contribution = edge * (psi[a] - psi[b]);
            out[a] += contribution;
            out[b] -= contribution;
        }
    }
}

void weighted_laplacian_diag(
    const double* q,
    const double* gauge,
    double* diag,
    int height,
    int width,
    double dx,
    double gauge_strength) {
    const std::size_t size = static_cast<std::size_t>(height) * width;
    const double inv_dx2 = 1.0 / (dx * dx);
    for (std::size_t i = 0; i < size; ++i) {
        diag[i] = gauge_strength * gauge[i] * gauge[i];
    }
    for (int i = 0; i < height; ++i) {
        const std::size_t row = static_cast<std::size_t>(i) * width;
        for (int j = 0; j + 1 < width; ++j) {
            const std::size_t a = row + j;
            const std::size_t b = a + 1;
            const double edge = 0.5 * (q[a] + q[b]) * inv_dx2;
            diag[a] += edge;
            diag[b] += edge;
        }
    }
    for (int i = 0; i + 1 < height; ++i) {
        const std::size_t row = static_cast<std::size_t>(i) * width;
        for (int j = 0; j < width; ++j) {
            const std::size_t a = row + j;
            const std::size_t b = a + width;
            const double edge = 0.5 * (q[a] + q[b]) * inv_dx2;
            diag[a] += edge;
            diag[b] += edge;
        }
    }
}

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
    int maxiter) {
    const std::size_t size = static_cast<std::size_t>(height) * width;
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch; ++b) {
        const std::size_t offset = static_cast<std::size_t>(b) * size;
        stats[b] = pcg_one_system(
            q_operator + offset,
            rhs + offset,
            gauge + offset,
            psi + offset,
            height,
            width,
            dx,
            gauge_strength,
            tol,
            maxiter);
    }
}

void weighted_operator_vjp_batch(
    const double* psi,
    const double* lambda,
    double* q_bar,
    int batch,
    int height,
    int width,
    double dx) {
    const std::size_t size = static_cast<std::size_t>(height) * width;
    const double factor = -0.5 / (dx * dx);
    #pragma omp parallel for schedule(static)
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        const std::size_t offset = static_cast<std::size_t>(batch_index) * size;
        const double* x = psi + offset;
        const double* adjoint = lambda + offset;
        double* out = q_bar + offset;
        std::fill(out, out + size, 0.0);
        for (int i = 0; i < height; ++i) {
            const std::size_t row = static_cast<std::size_t>(i) * width;
            for (int j = 0; j + 1 < width; ++j) {
                const std::size_t a = row + j;
                const std::size_t b = a + 1;
                const double contribution = factor
                    * (adjoint[a] - adjoint[b]) * (x[a] - x[b]);
                out[a] += contribution;
                out[b] += contribution;
            }
        }
        for (int i = 0; i + 1 < height; ++i) {
            const std::size_t row = static_cast<std::size_t>(i) * width;
            for (int j = 0; j < width; ++j) {
                const std::size_t a = row + j;
                const std::size_t b = a + width;
                const double contribution = factor
                    * (adjoint[a] - adjoint[b]) * (x[a] - x[b]);
                out[a] += contribution;
                out[b] += contribution;
            }
        }
    }
}

void gauge_vjp_batch(
    const double* psi,
    const double* lambda,
    const double* gauge,
    double* gauge_bar,
    int batch,
    int height,
    int width,
    double gauge_strength) {
    const std::size_t size = static_cast<std::size_t>(height) * width;
    #pragma omp parallel for schedule(static)
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        const std::size_t offset = static_cast<std::size_t>(batch_index) * size;
        const double* x = psi + offset;
        const double* adjoint = lambda + offset;
        const double* v = gauge + offset;
        double* out = gauge_bar + offset;
        const double v_dot_x = dot(v, x, size);
        const double v_dot_adjoint = dot(v, adjoint, size);
        for (std::size_t i = 0; i < size; ++i) {
            out[i] = -gauge_strength
                * (adjoint[i] * v_dot_x + x[i] * v_dot_adjoint);
        }
    }
}

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
    double gauge_strength) {
    const std::size_t size = static_cast<std::size_t>(height) * width;
    #pragma omp parallel for schedule(static)
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        const std::size_t offset = static_cast<std::size_t>(batch_index) * size;
        const double* x = psi + offset;
        const double* dq = q_dot + offset;
        const double* drhs = rhs_dot + offset;
        const double* v = gauge + offset;
        const double* dv = gauge_dot + offset;
        double* out = effective_rhs + offset;
        weighted_laplacian(x, dq, out, height, width, dx);
        const double v_dot_x = dot(v, x, size);
        const double dv_dot_x = dot(dv, x, size);
        for (std::size_t i = 0; i < size; ++i) {
            const double d_k_x = out[i] + gauge_strength
                * (dv[i] * v_dot_x + v[i] * dv_dot_x);
            out[i] = drhs[i] - d_k_x;
        }
    }
}

}  // namespace mfsi_poisson

#include "variational_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfsi_variational_poisson {
namespace {

constexpr long double kPi = 3.141592653589793238462643383279502884L;

struct EigenResult {
    std::vector<double> values;
    std::vector<double> vectors;
    int sweeps = 0;
    bool converged = false;
};

double l2_norm(const std::vector<double>& values) {
    long double sum = 0.0L;
    for (const double value : values) {
        sum += static_cast<long double>(value) * value;
    }
    return std::sqrt(static_cast<double>(sum));
}

EigenResult jacobi_eigendecomposition(
    std::vector<double> matrix,
    int size,
    double tolerance,
    int maximum_sweeps) {
    EigenResult result;
    result.vectors.assign(static_cast<std::size_t>(size) * size, 0.0);
    for (int i = 0; i < size; ++i) {
        result.vectors[static_cast<std::size_t>(i) * size + i] = 1.0;
    }

    for (int sweep = 0; sweep < maximum_sweeps; ++sweep) {
        double maximum_off_diagonal = 0.0;
        double maximum_diagonal = 0.0;
        for (int p = 0; p < size; ++p) {
            maximum_diagonal = std::max(
                maximum_diagonal,
                std::abs(matrix[static_cast<std::size_t>(p) * size + p]));
            for (int q = p + 1; q < size; ++q) {
                maximum_off_diagonal = std::max(
                    maximum_off_diagonal,
                    std::abs(matrix[static_cast<std::size_t>(p) * size + q]));
            }
        }
        result.sweeps = sweep;
        if (maximum_off_diagonal <= tolerance * std::max(maximum_diagonal, 1.0)) {
            result.converged = true;
            break;
        }

        for (int p = 0; p < size; ++p) {
            for (int q = p + 1; q < size; ++q) {
                const std::size_t pp = static_cast<std::size_t>(p) * size + p;
                const std::size_t qq = static_cast<std::size_t>(q) * size + q;
                const std::size_t pq = static_cast<std::size_t>(p) * size + q;
                const double apq = matrix[pq];
                if (std::abs(apq) <= tolerance) {
                    continue;
                }
                const double app = matrix[pp];
                const double aqq = matrix[qq];
                const double tau = (aqq - app) / (2.0 * apq);
                const double tangent = std::copysign(
                    1.0 / (std::abs(tau) + std::hypot(1.0, tau)), tau);
                const double cosine = 1.0 / std::hypot(1.0, tangent);
                const double sine = tangent * cosine;

                for (int k = 0; k < size; ++k) {
                    if (k == p || k == q) {
                        continue;
                    }
                    const std::size_t kp = static_cast<std::size_t>(k) * size + p;
                    const std::size_t kq = static_cast<std::size_t>(k) * size + q;
                    const double akp = matrix[kp];
                    const double akq = matrix[kq];
                    matrix[kp] = cosine * akp - sine * akq;
                    matrix[static_cast<std::size_t>(p) * size + k] = matrix[kp];
                    matrix[kq] = sine * akp + cosine * akq;
                    matrix[static_cast<std::size_t>(q) * size + k] = matrix[kq];
                }
                matrix[pp] = cosine * cosine * app - 2.0 * sine * cosine * apq
                    + sine * sine * aqq;
                matrix[qq] = sine * sine * app + 2.0 * sine * cosine * apq
                    + cosine * cosine * aqq;
                matrix[pq] = 0.0;
                matrix[static_cast<std::size_t>(q) * size + p] = 0.0;

                for (int k = 0; k < size; ++k) {
                    const std::size_t kp = static_cast<std::size_t>(k) * size + p;
                    const std::size_t kq = static_cast<std::size_t>(k) * size + q;
                    const double vkp = result.vectors[kp];
                    const double vkq = result.vectors[kq];
                    result.vectors[kp] = cosine * vkp - sine * vkq;
                    result.vectors[kq] = sine * vkp + cosine * vkq;
                }
            }
        }
        result.sweeps = sweep + 1;
    }

    if (!result.converged) {
        double maximum_off_diagonal = 0.0;
        double maximum_diagonal = 0.0;
        for (int p = 0; p < size; ++p) {
            maximum_diagonal = std::max(
                maximum_diagonal,
                std::abs(matrix[static_cast<std::size_t>(p) * size + p]));
            for (int q = p + 1; q < size; ++q) {
                maximum_off_diagonal = std::max(
                    maximum_off_diagonal,
                    std::abs(matrix[static_cast<std::size_t>(p) * size + q]));
            }
        }
        result.converged = maximum_off_diagonal
            <= tolerance * std::max(maximum_diagonal, 1.0);
    }

    result.values.resize(size);
    for (int i = 0; i < size; ++i) {
        result.values[i] = matrix[static_cast<std::size_t>(i) * size + i];
    }
    return result;
}

struct TrialSpace {
    std::vector<int> kx;
    std::vector<int> ky;
    std::vector<long double> cos_x;
    std::vector<long double> sin_x;
    std::vector<long double> cos_y;
    std::vector<long double> sin_y;
    int mode_count = 0;
};

TrialSpace build_trial_space(int height, int width, double dx, int maximum_mode) {
    TrialSpace trial;
    for (int y_mode = 0; y_mode <= maximum_mode; ++y_mode) {
        for (int x_mode = 0; x_mode <= maximum_mode; ++x_mode) {
            if (x_mode != 0 || y_mode != 0) {
                trial.kx.push_back(x_mode);
                trial.ky.push_back(y_mode);
            }
        }
    }
    trial.mode_count = static_cast<int>(trial.kx.size());
    trial.cos_x.resize(static_cast<std::size_t>(maximum_mode + 1) * width);
    trial.sin_x.resize(static_cast<std::size_t>(maximum_mode + 1) * width);
    trial.cos_y.resize(static_cast<std::size_t>(maximum_mode + 1) * height);
    trial.sin_y.resize(static_cast<std::size_t>(maximum_mode + 1) * height);
    const long double length_x = static_cast<long double>(width) * dx;
    const long double length_y = static_cast<long double>(height) * dx;
    for (int mode = 0; mode <= maximum_mode; ++mode) {
        for (int j = 0; j < width; ++j) {
            const long double phase = mode * kPi * (j + 0.5L) * dx / length_x;
            trial.cos_x[static_cast<std::size_t>(mode) * width + j] = std::cos(phase);
            trial.sin_x[static_cast<std::size_t>(mode) * width + j] = std::sin(phase);
        }
        for (int i = 0; i < height; ++i) {
            const long double phase = mode * kPi * (i + 0.5L) * dx / length_y;
            trial.cos_y[static_cast<std::size_t>(mode) * height + i] = std::cos(phase);
            trial.sin_y[static_cast<std::size_t>(mode) * height + i] = std::sin(phase);
        }
    }
    return trial;
}

void evaluate_modes(
    const TrialSpace& trial,
    int i,
    int j,
    int height,
    int width,
    double dx,
    std::vector<long double>& value,
    std::vector<long double>& gradient_x,
    std::vector<long double>& gradient_y) {
    const long double length_x = static_cast<long double>(width) * dx;
    const long double length_y = static_cast<long double>(height) * dx;
    for (int mode = 0; mode < trial.mode_count; ++mode) {
        const int kx = trial.kx[mode];
        const int ky = trial.ky[mode];
        const long double cx = trial.cos_x[static_cast<std::size_t>(kx) * width + j];
        const long double sx = trial.sin_x[static_cast<std::size_t>(kx) * width + j];
        const long double cy = trial.cos_y[static_cast<std::size_t>(ky) * height + i];
        const long double sy = trial.sin_y[static_cast<std::size_t>(ky) * height + i];
        value[mode] = cx * cy;
        gradient_x[mode] = -(kx * kPi / length_x) * sx * cy;
        gradient_y[mode] = -(ky * kPi / length_y) * cx * sy;
    }
}

SolveStats solve_one(
    const double* log_q,
    const double* forcing,
    double* potential,
    int height,
    int width,
    double dx,
    int maximum_mode,
    double rank_relative_tolerance,
    double weak_relative_tolerance,
    double eigensolver_tolerance,
    int maximum_eigensolver_sweeps) {
    const std::size_t cell_count = static_cast<std::size_t>(height) * width;
    const TrialSpace trial = build_trial_space(height, width, dx, maximum_mode);
    const int mode_count = trial.mode_count;
    SolveStats stats;
    stats.basis_size = mode_count;

    const double maximum_log_q = *std::max_element(log_q, log_q + cell_count);
    std::vector<long double> unnormalized_weight(cell_count);
    long double normalization = 0.0L;
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        const long double weight = std::exp(
            static_cast<long double>(log_q[cell]) - maximum_log_q);
        unnormalized_weight[cell] = weight;
        normalization += weight;
        stats.quadrature_underflow_count += weight == 0.0L ? 1 : 0;
    }
    if (!(normalization > 0.0L) || !std::isfinite(normalization)) {
        std::fill(potential, potential + cell_count, 0.0);
        return stats;
    }

    std::vector<long double> mean(mode_count, 0.0L);
    std::vector<long double> load(mode_count, 0.0L);
    std::vector<long double> gram(static_cast<std::size_t>(mode_count) * mode_count, 0.0L);
    std::vector<long double> value(mode_count);
    std::vector<long double> gradient_x(mode_count);
    std::vector<long double> gradient_y(mode_count);
    long double forcing_mean = 0.0L;
    long double forcing_square_mean = 0.0L;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const std::size_t cell = static_cast<std::size_t>(i) * width + j;
            const long double weight = unnormalized_weight[cell] / normalization;
            evaluate_modes(
                trial, i, j, height, width, dx, value, gradient_x, gradient_y);
            const long double local_forcing = forcing[cell];
            forcing_mean += weight * local_forcing;
            forcing_square_mean += weight * local_forcing * local_forcing;
            for (int a = 0; a < mode_count; ++a) {
                mean[a] += weight * value[a];
                for (int b = 0; b <= a; ++b) {
                    gram[static_cast<std::size_t>(a) * mode_count + b] += weight * (
                        gradient_x[a] * gradient_x[b]
                        + gradient_y[a] * gradient_y[b]);
                }
            }
        }
    }
    for (int a = 0; a < mode_count; ++a) {
        for (int b = 0; b < a; ++b) {
            gram[static_cast<std::size_t>(b) * mode_count + a]
                = gram[static_cast<std::size_t>(a) * mode_count + b];
        }
    }

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const std::size_t cell = static_cast<std::size_t>(i) * width + j;
            const long double weight = unnormalized_weight[cell] / normalization;
            evaluate_modes(
                trial, i, j, height, width, dx, value, gradient_x, gradient_y);
            for (int a = 0; a < mode_count; ++a) {
                load[a] += weight * forcing[cell] * (value[a] - mean[a]);
            }
        }
    }

    stats.compatibility_residual = static_cast<double>(forcing_mean);
    stats.compatibility_relative_residual = std::abs(stats.compatibility_residual)
        / std::max(std::sqrt(static_cast<double>(forcing_square_mean)), 1.0e-300);

    std::vector<double> scale(mode_count, 0.0);
    std::vector<double> scaled_gram(static_cast<std::size_t>(mode_count) * mode_count, 0.0);
    std::vector<double> scaled_load(mode_count, 0.0);
    for (int a = 0; a < mode_count; ++a) {
        const long double diagonal = gram[static_cast<std::size_t>(a) * mode_count + a];
        if (diagonal > 0.0L) {
            scale[a] = 1.0 / std::sqrt(static_cast<double>(diagonal));
        }
    }
    for (int a = 0; a < mode_count; ++a) {
        scaled_load[a] = scale[a] * static_cast<double>(load[a]);
        for (int b = 0; b < mode_count; ++b) {
            scaled_gram[static_cast<std::size_t>(a) * mode_count + b]
                = scale[a] * static_cast<double>(
                    gram[static_cast<std::size_t>(a) * mode_count + b]) * scale[b];
        }
    }

    EigenResult eigen = jacobi_eigendecomposition(
        scaled_gram,
        mode_count,
        eigensolver_tolerance,
        maximum_eigensolver_sweeps);
    stats.eigensolver_sweeps = eigen.sweeps;
    const double maximum_eigenvalue = *std::max_element(
        eigen.values.begin(), eigen.values.end());
    const double rank_threshold = rank_relative_tolerance
        * std::max(maximum_eigenvalue, std::numeric_limits<double>::min());
    std::vector<double> scaled_coefficients(mode_count, 0.0);
    double minimum_retained_eigenvalue = std::numeric_limits<double>::infinity();
    for (int k = 0; k < mode_count; ++k) {
        if (eigen.values[k] <= rank_threshold) {
            continue;
        }
        ++stats.retained_rank;
        minimum_retained_eigenvalue = std::min(
            minimum_retained_eigenvalue, eigen.values[k]);
        long double projection = 0.0L;
        for (int a = 0; a < mode_count; ++a) {
            projection += eigen.vectors[static_cast<std::size_t>(a) * mode_count + k]
                * scaled_load[a];
        }
        const double amplitude = -static_cast<double>(projection) / eigen.values[k];
        for (int a = 0; a < mode_count; ++a) {
            scaled_coefficients[a] += amplitude
                * eigen.vectors[static_cast<std::size_t>(a) * mode_count + k];
        }
    }
    stats.condition_proxy = stats.retained_rank > 0
        ? maximum_eigenvalue / minimum_retained_eigenvalue
        : std::numeric_limits<double>::infinity();

    std::vector<double> coefficients(mode_count);
    for (int a = 0; a < mode_count; ++a) {
        coefficients[a] = scale[a] * scaled_coefficients[a];
    }

    std::vector<double> weak_residual(mode_count, 0.0);
    std::vector<double> scaled_weak_residual(mode_count, 0.0);
    std::vector<double> load_double(mode_count);
    for (int a = 0; a < mode_count; ++a) {
        load_double[a] = static_cast<double>(load[a]);
        long double residual = load[a];
        for (int b = 0; b < mode_count; ++b) {
            residual += gram[static_cast<std::size_t>(a) * mode_count + b]
                * coefficients[b];
        }
        weak_residual[a] = static_cast<double>(residual);
        scaled_weak_residual[a] = scale[a] * weak_residual[a];
    }
    stats.weak_relative_residual = l2_norm(weak_residual)
        / std::max(l2_norm(load_double), 1.0e-300);
    stats.scaled_weak_relative_residual = l2_norm(scaled_weak_residual)
        / std::max(l2_norm(scaled_load), 1.0e-300);

    long double action = 0.0L;
    long double load_value = 0.0L;
    for (int a = 0; a < mode_count; ++a) {
        load_value += coefficients[a] * load[a];
        for (int b = 0; b < mode_count; ++b) {
            action += coefficients[a]
                * gram[static_cast<std::size_t>(a) * mode_count + b]
                * coefficients[b];
        }
    }
    stats.action = static_cast<double>(action);
    stats.objective = static_cast<double>(0.5L * action + load_value);
    stats.energy_load_identity_relative_error = static_cast<double>(
        std::abs(action + load_value)
        / std::max({std::abs(action), std::abs(load_value), 1.0e-300L}));

    long double gauge = 0.0L;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const std::size_t cell = static_cast<std::size_t>(i) * width + j;
            evaluate_modes(
                trial, i, j, height, width, dx, value, gradient_x, gradient_y);
            long double local_potential = 0.0L;
            for (int a = 0; a < mode_count; ++a) {
                local_potential += coefficients[a] * (value[a] - mean[a]);
            }
            potential[cell] = static_cast<double>(local_potential);
            gauge += (unnormalized_weight[cell] / normalization) * local_potential;
        }
    }
    stats.gauge_residual = static_cast<double>(gauge);
    stats.converged = eigen.converged
        && stats.retained_rank > 0
        && std::isfinite(stats.action)
        && std::isfinite(stats.scaled_weak_relative_residual)
        && stats.scaled_weak_relative_residual <= weak_relative_tolerance;
    return stats;
}

}  // namespace

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
    int maximum_eigensolver_sweeps) {
    const std::size_t size = static_cast<std::size_t>(height) * width;
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch; ++b) {
        const std::size_t offset = static_cast<std::size_t>(b) * size;
        stats[b] = solve_one(
            log_q_mass + offset,
            forcing + offset,
            potential + offset,
            height,
            width,
            dx,
            maximum_mode,
            rank_relative_tolerance,
            weak_relative_tolerance,
            eigensolver_tolerance,
            maximum_eigensolver_sweeps);
    }
}

}  // namespace mfsi_variational_poisson

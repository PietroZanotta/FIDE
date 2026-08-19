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
    std::vector<long double> values;
    std::vector<long double> vectors;
    int sweeps = 0;
    bool converged = false;
};

long double l2_norm(const std::vector<long double>& values) {
    long double sum = 0.0L;
    for (const long double value : values) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

EigenResult symmetric_eigendecomposition(
    std::vector<long double> matrix,
    int size,
    double tolerance,
    int maximum_sweeps) {
    EigenResult result;
    result.values.resize(size);
    result.vectors.assign(static_cast<std::size_t>(size) * size, 0.0L);
    for (int i = 0; i < size; ++i) {
        result.vectors[static_cast<std::size_t>(i) * size + i] = 1.0L;
    }

    const long double requested = static_cast<long double>(tolerance);
    const long double relative_tolerance = std::max(
        requested,
        8.0L * std::numeric_limits<long double>::epsilon());
    for (int sweep = 0; sweep < maximum_sweeps; ++sweep) {
        long double maximum_diagonal = 0.0L;
        long double maximum_off_diagonal = 0.0L;
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
        if (maximum_off_diagonal <= relative_tolerance
                * std::max(maximum_diagonal, 1.0e-300L)) {
            result.sweeps = sweep;
            result.converged = true;
            break;
        }

        for (int p = 0; p < size - 1; ++p) {
            for (int q = p + 1; q < size; ++q) {
                const std::size_t pp = static_cast<std::size_t>(p) * size + p;
                const std::size_t qq = static_cast<std::size_t>(q) * size + q;
                const std::size_t pq = static_cast<std::size_t>(p) * size + q;
                const long double apq = matrix[pq];
                if (std::abs(apq) <= relative_tolerance
                        * std::sqrt(std::max(
                            std::abs(matrix[pp] * matrix[qq]), 1.0e-300L))) {
                    continue;
                }
                const long double tau = (matrix[qq] - matrix[pp]) / (2.0L * apq);
                const long double tangent = std::copysign(
                    1.0L, tau) / (std::abs(tau) + std::sqrt(1.0L + tau * tau));
                const long double cosine = 1.0L / std::sqrt(1.0L + tangent * tangent);
                const long double sine = tangent * cosine;
                const long double app = matrix[pp];
                const long double aqq = matrix[qq];
                matrix[pp] = cosine * cosine * app
                    - 2.0L * sine * cosine * apq + sine * sine * aqq;
                matrix[qq] = sine * sine * app
                    + 2.0L * sine * cosine * apq + cosine * cosine * aqq;
                matrix[pq] = 0.0L;
                matrix[static_cast<std::size_t>(q) * size + p] = 0.0L;
                for (int k = 0; k < size; ++k) {
                    if (k != p && k != q) {
                        const std::size_t kp = static_cast<std::size_t>(k) * size + p;
                        const std::size_t kq = static_cast<std::size_t>(k) * size + q;
                        const long double akp = matrix[kp];
                        const long double akq = matrix[kq];
                        matrix[kp] = cosine * akp - sine * akq;
                        matrix[static_cast<std::size_t>(p) * size + k] = matrix[kp];
                        matrix[kq] = sine * akp + cosine * akq;
                        matrix[static_cast<std::size_t>(q) * size + k] = matrix[kq];
                    }
                    const std::size_t vp = static_cast<std::size_t>(k) * size + p;
                    const std::size_t vq = static_cast<std::size_t>(k) * size + q;
                    const long double vkp = result.vectors[vp];
                    const long double vkq = result.vectors[vq];
                    result.vectors[vp] = cosine * vkp - sine * vkq;
                    result.vectors[vq] = sine * vkp + cosine * vkq;
                }
            }
        }
        result.sweeps = sweep + 1;
    }

    if (!result.converged) {
        return result;
    }
    std::vector<int> order(size);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int left, int right) {
        return matrix[static_cast<std::size_t>(left) * size + left]
            < matrix[static_cast<std::size_t>(right) * size + right];
    });
    std::vector<long double> sorted_vectors(result.vectors.size());
    for (int column = 0; column < size; ++column) {
        const int source = order[column];
        result.values[column]
            = matrix[static_cast<std::size_t>(source) * size + source];
        for (int row = 0; row < size; ++row) {
            sorted_vectors[static_cast<std::size_t>(row) * size + column]
                = result.vectors[static_cast<std::size_t>(row) * size + source];
        }
    }
    result.vectors = std::move(sorted_vectors);
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

    std::vector<long double> scale(mode_count, 0.0L);
    std::vector<long double> scaled_gram(
        static_cast<std::size_t>(mode_count) * mode_count, 0.0L);
    std::vector<long double> scaled_load(mode_count, 0.0L);
    for (int a = 0; a < mode_count; ++a) {
        const long double diagonal = gram[static_cast<std::size_t>(a) * mode_count + a];
        if (diagonal > 0.0L) {
            scale[a] = 1.0L / std::sqrt(diagonal);
        }
    }
    for (int a = 0; a < mode_count; ++a) {
        scaled_load[a] = scale[a] * load[a];
        for (int b = 0; b < mode_count; ++b) {
            scaled_gram[static_cast<std::size_t>(a) * mode_count + b]
                = scale[a] * gram[static_cast<std::size_t>(a) * mode_count + b]
                * scale[b];
        }
    }

    EigenResult eigen = symmetric_eigendecomposition(
        scaled_gram,
        mode_count,
        eigensolver_tolerance,
        maximum_eigensolver_sweeps);
    stats.eigensolver_sweeps = eigen.sweeps;
    const long double maximum_eigenvalue = *std::max_element(
        eigen.values.begin(), eigen.values.end());
    const long double rank_threshold = static_cast<long double>(rank_relative_tolerance)
        * std::max(maximum_eigenvalue, std::numeric_limits<long double>::min());
    std::vector<long double> scaled_coefficients(mode_count, 0.0L);
    std::vector<bool> retained_mode(mode_count, false);
    long double minimum_retained_eigenvalue
        = std::numeric_limits<long double>::infinity();
    long double spectral_action = 0.0L;
    long double retained_algebraic_residual_square = 0.0L;
    long double retained_algebraic_load_square = 0.0L;
    for (int k = 0; k < mode_count; ++k) {
        if (eigen.values[k] <= rank_threshold) {
            continue;
        }
        retained_mode[k] = true;
        ++stats.retained_rank;
        minimum_retained_eigenvalue = std::min(
            minimum_retained_eigenvalue, eigen.values[k]);
        long double projection = 0.0L;
        for (int a = 0; a < mode_count; ++a) {
            projection += eigen.vectors[static_cast<std::size_t>(a) * mode_count + k]
                * scaled_load[a];
        }
        const long double amplitude = -projection / eigen.values[k];
        const long double algebraic_residual
            = eigen.values[k] * amplitude + projection;
        retained_algebraic_residual_square
            += algebraic_residual * algebraic_residual;
        retained_algebraic_load_square += projection * projection;
        spectral_action += projection * projection / eigen.values[k];
        for (int a = 0; a < mode_count; ++a) {
            scaled_coefficients[a] += amplitude
                * eigen.vectors[static_cast<std::size_t>(a) * mode_count + k];
        }
    }
    stats.condition_proxy = stats.retained_rank > 0
        ? maximum_eigenvalue / minimum_retained_eigenvalue
        : std::numeric_limits<double>::infinity();

    std::vector<long double> coefficients(mode_count);
    for (int a = 0; a < mode_count; ++a) {
        coefficients[a] = scale[a] * scaled_coefficients[a];
    }

    std::vector<long double> weak_residual(mode_count, 0.0L);
    std::vector<long double> scaled_weak_residual(mode_count, 0.0L);
    for (int a = 0; a < mode_count; ++a) {
        long double residual = load[a];
        for (int b = 0; b < mode_count; ++b) {
            residual += gram[static_cast<std::size_t>(a) * mode_count + b]
                * coefficients[b];
        }
        weak_residual[a] = residual;
        scaled_weak_residual[a] = scale[a] * weak_residual[a];
    }
    stats.weak_relative_residual = l2_norm(weak_residual)
        / std::max(l2_norm(load), 1.0e-300L);
    stats.scaled_weak_relative_residual = l2_norm(scaled_weak_residual)
        / std::max(l2_norm(scaled_load), 1.0e-300L);

    long double discarded_load_square = 0.0L;
    for (int k = 0; k < mode_count; ++k) {
        long double load_projection = 0.0L;
        for (int a = 0; a < mode_count; ++a) {
            const long double component
                = eigen.vectors[static_cast<std::size_t>(a) * mode_count + k];
            load_projection += component * scaled_load[a];
        }
        if (!retained_mode[k]) {
            discarded_load_square += load_projection * load_projection;
        }
    }
    stats.retained_scaled_weak_relative_residual = static_cast<double>(
        std::sqrt(retained_algebraic_residual_square)
        / std::max(std::sqrt(retained_algebraic_load_square), 1.0e-300L));
    stats.discarded_scaled_load_relative_residual = static_cast<double>(
        std::sqrt(discarded_load_square)
        / std::max(l2_norm(scaled_load), 1.0e-300L));

    long double load_value = 0.0L;
    for (int a = 0; a < mode_count; ++a) {
        load_value += coefficients[a] * load[a];
    }
    stats.action = static_cast<double>(spectral_action);
    stats.objective = static_cast<double>(0.5L * spectral_action + load_value);
    stats.energy_load_identity_relative_error = static_cast<double>(
        std::abs(spectral_action + load_value)
        / std::max({std::abs(spectral_action), std::abs(load_value), 1.0e-300L}));

    long double gauge = 0.0L;
    long double potential_square_mean = 0.0L;
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
            potential_square_mean += (unnormalized_weight[cell] / normalization)
                * local_potential * local_potential;
        }
    }
    stats.gauge_residual = static_cast<double>(gauge);
    stats.gauge_relative_residual = static_cast<double>(
        std::abs(gauge)
        / std::max(std::sqrt(potential_square_mean), 1.0e-300L));
    stats.converged = eigen.converged
        && stats.retained_rank > 0
        && std::isfinite(stats.action)
        && std::isfinite(stats.retained_scaled_weak_relative_residual)
        && stats.retained_scaled_weak_relative_residual <= weak_relative_tolerance;
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

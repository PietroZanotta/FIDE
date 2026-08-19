#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

struct Shape {
    int batch;
    int times;
    int particles;
    int moments;
};

struct Config {
    int max_steps;
    double residual_tol;
    double newton_ridge;
    double step_cap;
    double lambda_clip;
    int line_search_steps;
    double implicit_ridge;
};

Shape validate_inputs(
    const py::array& phi, const py::array& log_base, const py::array& targets) {
    for (const auto* item : {&phi, &log_base, &targets}) {
        if (!item->dtype().is(py::dtype::of<double>())) {
            throw py::type_error("I-projection arrays must have dtype float64");
        }
        if (!(item->flags() & py::array::c_style)) {
            throw py::value_error("I-projection arrays must be C-contiguous");
        }
    }
    if (phi.ndim() != 3) {
        throw py::value_error("phi must have shape [T,N,M]");
    }
    if (log_base.ndim() != 2) {
        throw py::value_error("log_base_weights must have shape [T,N]");
    }
    if (targets.ndim() != 3) {
        throw py::value_error("targets must have shape [B,T,M]");
    }
    const Shape shape{
        static_cast<int>(targets.shape(0)),
        static_cast<int>(phi.shape(0)),
        static_cast<int>(phi.shape(1)),
        static_cast<int>(phi.shape(2)),
    };
    if (shape.batch < 1 || shape.times < 1 || shape.particles < 1 || shape.moments < 1) {
        throw py::value_error("all I-projection dimensions must be positive");
    }
    if (log_base.shape(0) != shape.times || log_base.shape(1) != shape.particles
        || targets.shape(1) != shape.times || targets.shape(2) != shape.moments) {
        throw py::value_error("inconsistent phi/log_base_weights/targets shapes");
    }
    return shape;
}

Config validate_config(
    int max_steps, double residual_tol, double newton_ridge, double step_cap,
    double lambda_clip, int line_search_steps, double implicit_ridge) {
    if (max_steps < 1 || line_search_steps < 1) {
        throw py::value_error("max_steps and line_search_steps must be positive");
    }
    if (!(residual_tol > 0.0) || newton_ridge < 0.0 || !(step_cap > 0.0)
        || !(lambda_clip > 0.0) || implicit_ridge < 0.0) {
        throw py::value_error("invalid I-projection numerical configuration");
    }
    return {max_steps, residual_tol, newton_ridge, step_cap, lambda_clip,
            line_search_steps, implicit_ridge};
}

bool solve_dense(std::vector<double> a, std::vector<double> b, int m, std::vector<double>& x) {
    for (int k = 0; k < m; ++k) {
        int pivot = k;
        double best = std::abs(a[static_cast<std::size_t>(k) * m + k]);
        for (int i = k + 1; i < m; ++i) {
            const double candidate = std::abs(a[static_cast<std::size_t>(i) * m + k]);
            if (candidate > best) {
                best = candidate;
                pivot = i;
            }
        }
        if (!(best > 1.0e-18) || !std::isfinite(best)) {
            return false;
        }
        if (pivot != k) {
            for (int j = k; j < m; ++j) {
                std::swap(a[static_cast<std::size_t>(k) * m + j],
                          a[static_cast<std::size_t>(pivot) * m + j]);
            }
            std::swap(b[k], b[pivot]);
        }
        const double diagonal = a[static_cast<std::size_t>(k) * m + k];
        for (int i = k + 1; i < m; ++i) {
            const double factor = a[static_cast<std::size_t>(i) * m + k] / diagonal;
            a[static_cast<std::size_t>(i) * m + k] = 0.0;
            for (int j = k + 1; j < m; ++j) {
                a[static_cast<std::size_t>(i) * m + j]
                    -= factor * a[static_cast<std::size_t>(k) * m + j];
            }
            b[i] -= factor * b[k];
        }
    }
    x.assign(m, 0.0);
    for (int i = m - 1; i >= 0; --i) {
        double value = b[i];
        for (int j = i + 1; j < m; ++j) {
            value -= a[static_cast<std::size_t>(i) * m + j] * x[j];
        }
        x[i] = value / a[static_cast<std::size_t>(i) * m + i];
    }
    return true;
}

double state(
    const double* phi, const double* log_base, const double* target,
    const std::vector<double>& lambda, int n, int m, std::vector<double>& weights,
    std::vector<double>& mean, std::vector<double>& covariance,
    std::vector<double>& residual) {
    weights.resize(n);
    mean.assign(m, 0.0);
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        double logit = log_base[i];
        for (int j = 0; j < m; ++j) {
            logit += phi[static_cast<std::size_t>(i) * m + j] * lambda[j];
        }
        weights[i] = logit;
        max_logit = std::max(max_logit, logit);
    }
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        weights[i] = std::exp(weights[i] - max_logit);
        total += weights[i];
    }
    for (int i = 0; i < n; ++i) {
        weights[i] /= total;
        for (int j = 0; j < m; ++j) {
            mean[j] += weights[i] * phi[static_cast<std::size_t>(i) * m + j];
        }
    }
    covariance.assign(static_cast<std::size_t>(m) * m, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            const double dj = phi[static_cast<std::size_t>(i) * m + j] - mean[j];
            for (int k = 0; k < m; ++k) {
                const double dk = phi[static_cast<std::size_t>(i) * m + k] - mean[k];
                covariance[static_cast<std::size_t>(j) * m + k] += weights[i] * dj * dk;
            }
        }
    }
    residual.resize(m);
    double norm2 = 0.0;
    for (int j = 0; j < m; ++j) {
        residual[j] = mean[j] - target[j];
        norm2 += residual[j] * residual[j];
    }
    return std::sqrt(norm2);
}

double dual_value(
    const double* phi, const double* log_base, const double* target,
    const std::vector<double>& lambda, int n, int m) {
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        double value = log_base[i];
        for (int j = 0; j < m; ++j) {
            value += phi[static_cast<std::size_t>(i) * m + j] * lambda[j];
        }
        max_logit = std::max(max_logit, value);
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double value = log_base[i];
        for (int j = 0; j < m; ++j) {
            value += phi[static_cast<std::size_t>(i) * m + j] * lambda[j];
        }
        sum += std::exp(value - max_logit);
    }
    double result = max_logit + std::log(sum);
    for (int j = 0; j < m; ++j) {
        result -= lambda[j] * target[j];
    }
    return result;
}

void solve_one(
    const double* phi, const double* log_base, const double* target, int n, int m,
    const Config& cfg, std::vector<double>& lambda, int& iterations,
    double& residual_norm) {
    std::vector<double> weights, mean, covariance, residual, delta;
    residual_norm = state(phi, log_base, target, lambda, n, m, weights, mean, covariance, residual);
    iterations = 0;
    while (iterations < cfg.max_steps && residual_norm > cfg.residual_tol) {
        std::vector<double> hessian = covariance;
        for (int j = 0; j < m; ++j) {
            hessian[static_cast<std::size_t>(j) * m + j] += cfg.newton_ridge;
        }
        if (!solve_dense(hessian, residual, m, delta)) {
            break;
        }
        double norm2 = 0.0;
        for (double value : delta) {
            norm2 += value * value;
        }
        const double norm = std::sqrt(norm2);
        const double cap_scale = std::min(1.0, cfg.step_cap / std::max(norm, 1.0e-30));
        for (double& value : delta) {
            value *= cap_scale;
        }

        const double current_dual = dual_value(phi, log_base, target, lambda, n, m);
        std::vector<double> best_lambda(m);
        double best_dual = std::numeric_limits<double>::infinity();
        double scale = 1.0;
        for (int line = 0; line < cfg.line_search_steps; ++line) {
            std::vector<double> candidate(m);
            for (int j = 0; j < m; ++j) {
                candidate[j] = std::clamp(
                    lambda[j] - scale * delta[j], -cfg.lambda_clip, cfg.lambda_clip);
            }
            const double value = dual_value(phi, log_base, target, candidate, n, m);
            if (value < best_dual) {
                best_dual = value;
                best_lambda = candidate;
            }
            // A Newton step for this convex dual is almost always accepted at
            // unit scale. Stop after the first monotone candidate instead of
            // paying for every configured candidate as the generic JAX graph
            // must. The converged root and implicit derivative are unchanged.
            if (value <= current_dual + 1.0e-14) {
                best_lambda = std::move(candidate);
                break;
            }
            scale *= 0.5;
        }
        lambda = std::move(best_lambda);
        residual_norm = state(
            phi, log_base, target, lambda, n, m, weights, mean, covariance, residual);
        ++iterations;
    }
}

double soft_state(
    const double* phi, const double* log_base, const double* target,
    const double* penalty, const std::vector<double>& lambda, int n, int m,
    std::vector<double>& weights, std::vector<double>& mean,
    std::vector<double>& covariance, std::vector<double>& residual,
    double& hard_residual_norm) {
    hard_residual_norm = state(
        phi, log_base, target, lambda, n, m,
        weights, mean, covariance, residual);
    double norm2 = 0.0;
    for (int j = 0; j < m; ++j) {
        for (int k = 0; k < m; ++k) {
            residual[j] += penalty[static_cast<std::size_t>(j) * m + k] * lambda[k];
            covariance[static_cast<std::size_t>(j) * m + k]
                += penalty[static_cast<std::size_t>(j) * m + k];
        }
        norm2 += residual[j] * residual[j];
    }
    return std::sqrt(norm2);
}

double soft_dual_value(
    const double* phi, const double* log_base, const double* target,
    const double* penalty, const std::vector<double>& lambda, int n, int m) {
    double result = dual_value(phi, log_base, target, lambda, n, m);
    double quadratic = 0.0;
    for (int j = 0; j < m; ++j) {
        for (int k = 0; k < m; ++k) {
            quadratic += lambda[j]
                * penalty[static_cast<std::size_t>(j) * m + k] * lambda[k];
        }
    }
    return result + 0.5 * quadratic;
}

void solve_soft_one(
    const double* phi, const double* log_base, const double* target,
    const double* penalty, int n, int m, const Config& cfg,
    std::vector<double>& lambda, int& iterations, double& residual_norm,
    double& hard_residual_norm) {
    std::vector<double> weights, mean, hessian, residual, delta;
    residual_norm = soft_state(
        phi, log_base, target, penalty, lambda, n, m,
        weights, mean, hessian, residual, hard_residual_norm);
    iterations = 0;
    while (iterations < cfg.max_steps && residual_norm > cfg.residual_tol) {
        for (int j = 0; j < m; ++j) {
            hessian[static_cast<std::size_t>(j) * m + j] += cfg.newton_ridge;
        }
        if (!solve_dense(hessian, residual, m, delta)) {
            break;
        }
        double norm2 = 0.0;
        for (const double value : delta) {
            norm2 += value * value;
        }
        const double norm = std::sqrt(norm2);
        const double cap_scale = std::min(
            1.0, cfg.step_cap / std::max(norm, 1.0e-30));
        for (double& value : delta) {
            value *= cap_scale;
        }

        const double current_dual = soft_dual_value(
            phi, log_base, target, penalty, lambda, n, m);
        std::vector<double> best_lambda(m);
        double best_dual = std::numeric_limits<double>::infinity();
        double scale = 1.0;
        for (int line = 0; line < cfg.line_search_steps; ++line) {
            std::vector<double> candidate(m);
            for (int j = 0; j < m; ++j) {
                candidate[j] = std::clamp(
                    lambda[j] - scale * delta[j],
                    -cfg.lambda_clip, cfg.lambda_clip);
            }
            const double value = soft_dual_value(
                phi, log_base, target, penalty, candidate, n, m);
            if (value < best_dual) {
                best_dual = value;
                best_lambda = candidate;
            }
            if (value <= current_dual + 1.0e-14) {
                best_lambda = std::move(candidate);
                break;
            }
            scale *= 0.5;
        }
        lambda = std::move(best_lambda);
        residual_norm = soft_state(
            phi, log_base, target, penalty, lambda, n, m,
            weights, mean, hessian, residual, hard_residual_norm);
        ++iterations;
    }
}

py::dict solve_batch(
    const py::array& phi_array, const py::array& log_base_array,
    const py::array& targets_array, int max_steps, double residual_tol,
    double newton_ridge, double step_cap, double lambda_clip,
    int line_search_steps, double implicit_ridge) {
    const Shape s = validate_inputs(phi_array, log_base_array, targets_array);
    const Config cfg = validate_config(max_steps, residual_tol, newton_ridge, step_cap,
                                       lambda_clip, line_search_steps, implicit_ridge);
    const auto* phi = static_cast<const double*>(phi_array.data());
    const auto* log_base = static_cast<const double*>(log_base_array.data());
    const auto* targets = static_cast<const double*>(targets_array.data());
    py::array_t<double> lambda_out({s.batch, s.times, s.moments});
    py::array_t<std::int32_t> iteration_out({s.batch, s.times});
    py::array_t<double> residual_out({s.batch, s.times});
    py::array_t<bool> converged_out({s.batch, s.times});
    auto* lambdas = lambda_out.mutable_data();
    auto* iterations = iteration_out.mutable_data();
    auto* residuals = residual_out.mutable_data();
    auto* converged = converged_out.mutable_data();
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < s.batch; ++b) {
            std::vector<double> lambda(s.moments, 0.0);
            for (int t = 0; t < s.times; ++t) {
                const std::size_t pt = static_cast<std::size_t>(t) * s.particles * s.moments;
                const std::size_t wt = static_cast<std::size_t>(t) * s.particles;
                const std::size_t bt = (static_cast<std::size_t>(b) * s.times + t) * s.moments;
                int count = 0;
                double norm = 0.0;
                solve_one(phi + pt, log_base + wt, targets + bt, s.particles, s.moments,
                          cfg, lambda, count, norm);
                for (int j = 0; j < s.moments; ++j) {
                    lambdas[bt + j] = lambda[j];
                }
                const std::size_t row = static_cast<std::size_t>(b) * s.times + t;
                iterations[row] = count;
                residuals[row] = norm;
                converged[row] = norm <= cfg.residual_tol;
            }
        }
    }
    return py::dict(
        "lambda_values"_a = std::move(lambda_out),
        "iterations"_a = std::move(iteration_out),
        "residual_norm"_a = std::move(residual_out),
        "converged"_a = std::move(converged_out));
}

py::dict solve_soft_batch(
    const py::array& phi_array, const py::array& log_base_array,
    const py::array& targets_array, const py::array& penalties_array,
    int max_steps, double residual_tol, double newton_ridge,
    double step_cap, double lambda_clip, int line_search_steps) {
    const Shape s = validate_inputs(phi_array, log_base_array, targets_array);
    if (penalties_array.ndim() != 4
        || !penalties_array.dtype().is(py::dtype::of<double>())
        || !(penalties_array.flags() & py::array::c_style)
        || penalties_array.shape(0) != s.batch
        || penalties_array.shape(1) != s.times
        || penalties_array.shape(2) != s.moments
        || penalties_array.shape(3) != s.moments) {
        throw py::value_error(
            "penalties must be contiguous float64 [B,T,M,M]");
    }
    const Config cfg = validate_config(
        max_steps, residual_tol, newton_ridge, step_cap,
        lambda_clip, line_search_steps, 0.0);
    const auto* penalties = static_cast<const double*>(penalties_array.data());
    for (ssize_t index = 0; index < penalties_array.size(); ++index) {
        if (!std::isfinite(penalties[index])) {
            throw py::value_error("penalties must be finite");
        }
    }
    const auto* phi = static_cast<const double*>(phi_array.data());
    const auto* log_base = static_cast<const double*>(log_base_array.data());
    const auto* targets = static_cast<const double*>(targets_array.data());
    py::array_t<double> lambda_out({s.batch, s.times, s.moments});
    py::array_t<std::int32_t> iteration_out({s.batch, s.times});
    py::array_t<double> residual_out({s.batch, s.times});
    py::array_t<double> hard_residual_out({s.batch, s.times});
    py::array_t<bool> converged_out({s.batch, s.times});
    auto* lambdas = lambda_out.mutable_data();
    auto* iterations = iteration_out.mutable_data();
    auto* residuals = residual_out.mutable_data();
    auto* hard_residuals = hard_residual_out.mutable_data();
    auto* converged = converged_out.mutable_data();
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < s.batch; ++b) {
            std::vector<double> lambda(s.moments, 0.0);
            for (int t = 0; t < s.times; ++t) {
                const std::size_t pt
                    = static_cast<std::size_t>(t) * s.particles * s.moments;
                const std::size_t wt
                    = static_cast<std::size_t>(t) * s.particles;
                const std::size_t bt
                    = (static_cast<std::size_t>(b) * s.times + t) * s.moments;
                const std::size_t penalty_offset = bt * s.moments;
                int count = 0;
                double norm = 0.0;
                double hard_norm = 0.0;
                solve_soft_one(
                    phi + pt, log_base + wt, targets + bt,
                    penalties + penalty_offset, s.particles, s.moments,
                    cfg, lambda, count, norm, hard_norm);
                for (int j = 0; j < s.moments; ++j) {
                    lambdas[bt + j] = lambda[j];
                }
                const std::size_t row
                    = static_cast<std::size_t>(b) * s.times + t;
                iterations[row] = count;
                residuals[row] = norm;
                hard_residuals[row] = hard_norm;
                converged[row] = norm <= cfg.residual_tol;
            }
        }
    }
    return py::dict(
        "lambda_values"_a = std::move(lambda_out),
        "iterations"_a = std::move(iteration_out),
        "residual_norm"_a = std::move(residual_out),
        "hard_moment_residual_norm"_a = std::move(hard_residual_out),
        "converged"_a = std::move(converged_out));
}

py::dict vjp_batch(
    const py::array& phi_array, const py::array& log_base_array,
    const py::array& targets_array, const py::array& lambda_values_array,
    const py::array& lambda_bar_array,
    int max_steps, double residual_tol, double newton_ridge, double step_cap,
    double lambda_clip, int line_search_steps, double implicit_ridge) {
    const Shape s = validate_inputs(phi_array, log_base_array, targets_array);
    if (lambda_values_array.ndim() != 3 || !lambda_values_array.dtype().is(py::dtype::of<double>())
        || !(lambda_values_array.flags() & py::array::c_style)
        || lambda_values_array.shape(0) != s.batch || lambda_values_array.shape(1) != s.times
        || lambda_values_array.shape(2) != s.moments) {
        throw py::value_error("lambda_values must be contiguous float64 [B,T,M]");
    }
    if (lambda_bar_array.ndim() != 3 || !lambda_bar_array.dtype().is(py::dtype::of<double>())
        || !(lambda_bar_array.flags() & py::array::c_style)
        || lambda_bar_array.shape(0) != s.batch || lambda_bar_array.shape(1) != s.times
        || lambda_bar_array.shape(2) != s.moments) {
        throw py::value_error("lambda_bar must be contiguous float64 [B,T,M]");
    }
    const Config cfg = validate_config(max_steps, residual_tol, newton_ridge, step_cap,
                                       lambda_clip, line_search_steps, implicit_ridge);
    const auto* phi = static_cast<const double*>(phi_array.data());
    const auto* log_base = static_cast<const double*>(log_base_array.data());
    const auto* targets = static_cast<const double*>(targets_array.data());
    const auto* lambda_values = static_cast<const double*>(lambda_values_array.data());
    const auto* lambda_bar = static_cast<const double*>(lambda_bar_array.data());
    py::array_t<double> phi_bar_out({s.times, s.particles, s.moments});
    py::array_t<double> log_bar_out({s.times, s.particles});
    py::array_t<double> target_bar_out({s.batch, s.times, s.moments});
    std::fill(phi_bar_out.mutable_data(), phi_bar_out.mutable_data() + phi_bar_out.size(), 0.0);
    std::fill(log_bar_out.mutable_data(), log_bar_out.mutable_data() + log_bar_out.size(), 0.0);
    auto* phi_bar = phi_bar_out.mutable_data();
    auto* log_bar = log_bar_out.mutable_data();
    auto* target_bar = target_bar_out.mutable_data();
    // Each thread accumulates shared phi/log-weight cotangents privately.
    const int thread_count = std::min(s.batch, omp_get_max_threads());
    std::vector<std::vector<double>> private_phi(
        thread_count, std::vector<double>(static_cast<std::size_t>(s.times) * s.particles * s.moments, 0.0));
    std::vector<std::vector<double>> private_log(
        thread_count, std::vector<double>(static_cast<std::size_t>(s.times) * s.particles, 0.0));
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int b = 0; b < s.batch; ++b) {
            const int thread = omp_get_thread_num();
            auto& local_phi = private_phi[thread];
            auto& local_log = private_log[thread];
            std::vector<double> lambda(s.moments), weights, mean, covariance, residual, adjoint;
            for (int t = 0; t < s.times; ++t) {
                const std::size_t pt = static_cast<std::size_t>(t) * s.particles * s.moments;
                const std::size_t wt = static_cast<std::size_t>(t) * s.particles;
                const std::size_t bt = (static_cast<std::size_t>(b) * s.times + t) * s.moments;
                for (int j = 0; j < s.moments; ++j) lambda[j] = lambda_values[bt + j];
                state(phi + pt, log_base + wt, targets + bt, lambda, s.particles, s.moments,
                      weights, mean, covariance, residual);
                for (int j = 0; j < s.moments; ++j) {
                    covariance[static_cast<std::size_t>(j) * s.moments + j] += cfg.implicit_ridge;
                }
                std::vector<double> rhs(s.moments);
                for (int j = 0; j < s.moments; ++j) rhs[j] = lambda_bar[bt + j];
                if (!solve_dense(covariance, rhs, s.moments, adjoint)) {
                    throw std::runtime_error("singular covariance in I-projection VJP");
                }
                for (int j = 0; j < s.moments; ++j) target_bar[bt + j] = adjoint[j];
                for (int i = 0; i < s.particles; ++i) {
                    double centered_dot = 0.0;
                    for (int j = 0; j < s.moments; ++j) {
                        centered_dot += (phi[pt + static_cast<std::size_t>(i) * s.moments + j] - mean[j]) * adjoint[j];
                    }
                    local_log[wt + i] -= weights[i] * centered_dot;
                    for (int j = 0; j < s.moments; ++j) {
                        local_phi[pt + static_cast<std::size_t>(i) * s.moments + j]
                            -= weights[i] * (adjoint[j] + lambda[j] * centered_dot);
                    }
                }
            }
        }
    }
    for (int thread = 0; thread < thread_count; ++thread) {
        for (std::size_t i = 0; i < private_phi[thread].size(); ++i) phi_bar[i] += private_phi[thread][i];
        for (std::size_t i = 0; i < private_log[thread].size(); ++i) log_bar[i] += private_log[thread][i];
    }
    return py::dict("phi"_a = std::move(phi_bar_out),
                    "log_base_weights"_a = std::move(log_bar_out),
                    "targets"_a = std::move(target_bar_out));
}

py::array_t<double> jvp_batch(
    const py::array& phi_array, const py::array& log_base_array,
    const py::array& targets_array, const py::array& lambda_values_array,
    const py::array& phi_dot_array,
    const py::array& log_dot_array, const py::array& target_dot_array,
    int max_steps, double residual_tol, double newton_ridge, double step_cap,
    double lambda_clip, int line_search_steps, double implicit_ridge) {
    const Shape s = validate_inputs(phi_array, log_base_array, targets_array);
    validate_inputs(phi_dot_array, log_dot_array, target_dot_array);
    if (lambda_values_array.ndim() != 3 || !lambda_values_array.dtype().is(py::dtype::of<double>())
        || !(lambda_values_array.flags() & py::array::c_style)
        || lambda_values_array.shape(0) != s.batch || lambda_values_array.shape(1) != s.times
        || lambda_values_array.shape(2) != s.moments) {
        throw py::value_error("lambda_values must be contiguous float64 [B,T,M]");
    }
    const Config cfg = validate_config(max_steps, residual_tol, newton_ridge, step_cap,
                                       lambda_clip, line_search_steps, implicit_ridge);
    const auto* phi = static_cast<const double*>(phi_array.data());
    const auto* log_base = static_cast<const double*>(log_base_array.data());
    const auto* targets = static_cast<const double*>(targets_array.data());
    const auto* lambda_values = static_cast<const double*>(lambda_values_array.data());
    const auto* phi_dot = static_cast<const double*>(phi_dot_array.data());
    const auto* log_dot = static_cast<const double*>(log_dot_array.data());
    const auto* target_dot = static_cast<const double*>(target_dot_array.data());
    py::array_t<double> output({s.batch, s.times, s.moments});
    auto* lambda_dot = output.mutable_data();
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < s.batch; ++b) {
            std::vector<double> lambda(s.moments), weights, mean, covariance, residual, solution;
            for (int t = 0; t < s.times; ++t) {
                const std::size_t pt = static_cast<std::size_t>(t) * s.particles * s.moments;
                const std::size_t wt = static_cast<std::size_t>(t) * s.particles;
                const std::size_t bt = (static_cast<std::size_t>(b) * s.times + t) * s.moments;
                for (int j = 0; j < s.moments; ++j) lambda[j] = lambda_values[bt + j];
                state(phi + pt, log_base + wt, targets + bt, lambda, s.particles, s.moments,
                      weights, mean, covariance, residual);
                for (int j = 0; j < s.moments; ++j) {
                    covariance[static_cast<std::size_t>(j) * s.moments + j] += cfg.implicit_ridge;
                }
                std::vector<double> dF(s.moments, 0.0);
                for (int j = 0; j < s.moments; ++j) dF[j] = -target_dot[bt + j];
                for (int i = 0; i < s.particles; ++i) {
                    double logit_dot = log_dot[wt + i];
                    for (int j = 0; j < s.moments; ++j) {
                        logit_dot += phi_dot[pt + static_cast<std::size_t>(i) * s.moments + j] * lambda[j];
                    }
                    for (int j = 0; j < s.moments; ++j) {
                        dF[j] += weights[i] * (
                            phi_dot[pt + static_cast<std::size_t>(i) * s.moments + j]
                            + (phi[pt + static_cast<std::size_t>(i) * s.moments + j] - mean[j]) * logit_dot);
                    }
                }
                for (double& value : dF) value = -value;
                if (!solve_dense(covariance, dF, s.moments, solution)) {
                    throw std::runtime_error("singular covariance in I-projection JVP");
                }
                for (int j = 0; j < s.moments; ++j) lambda_dot[bt + j] = solution[j];
            }
        }
    }
    return output;
}

}  // namespace

PYBIND11_MODULE(_iprojection_native, module) {
    module.doc() = "C++17/OpenMP batched empirical I-projection trajectory solver";
    module.attr("__version__") = "1";
    module.def("solve_batch", &solve_batch,
               py::arg("phi").noconvert(), py::arg("log_base_weights").noconvert(),
               py::arg("targets").noconvert(), py::arg("max_steps"),
               py::arg("residual_tol"), py::arg("newton_ridge"), py::arg("step_cap"),
               py::arg("lambda_clip"), py::arg("line_search_steps"), py::arg("implicit_ridge"));
    module.def("solve_soft_batch", &solve_soft_batch,
               py::arg("phi").noconvert(), py::arg("log_base_weights").noconvert(),
               py::arg("targets").noconvert(), py::arg("penalties").noconvert(),
               py::arg("max_steps"), py::arg("residual_tol"),
               py::arg("newton_ridge"), py::arg("step_cap"),
               py::arg("lambda_clip"), py::arg("line_search_steps"));
    module.def("vjp_batch", &vjp_batch,
               py::arg("phi").noconvert(), py::arg("log_base_weights").noconvert(),
               py::arg("targets").noconvert(), py::arg("lambda_values").noconvert(),
               py::arg("lambda_bar").noconvert(),
               py::arg("max_steps"), py::arg("residual_tol"), py::arg("newton_ridge"),
               py::arg("step_cap"), py::arg("lambda_clip"), py::arg("line_search_steps"),
               py::arg("implicit_ridge"));
    module.def("jvp_batch", &jvp_batch,
               py::arg("phi").noconvert(), py::arg("log_base_weights").noconvert(),
               py::arg("targets").noconvert(), py::arg("lambda_values").noconvert(),
               py::arg("phi_dot").noconvert(),
               py::arg("log_base_weights_dot").noconvert(), py::arg("targets_dot").noconvert(),
               py::arg("max_steps"), py::arg("residual_tol"), py::arg("newton_ridge"),
               py::arg("step_cap"), py::arg("lambda_clip"), py::arg("line_search_steps"),
               py::arg("implicit_ridge"));
}

#include "variational_solver.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;
using mfsi_variational_poisson::SolveStats;

namespace {

struct Shape3 {
    int batch;
    int height;
    int width;
};

Shape3 validate_array(const py::array& array, const char* name) {
    if (!array.dtype().is(py::dtype::of<double>())) {
        throw py::type_error(std::string(name) + " must have dtype float64");
    }
    if (array.ndim() != 3) {
        throw py::value_error(std::string(name) + " must have rank 3 [B,H,W]");
    }
    if (!(array.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    if (array.shape(0) < 1 || array.shape(1) < 3 || array.shape(2) < 3) {
        throw py::value_error(std::string(name) + " must have shape [B,H>=3,W>=3]");
    }
    return {
        static_cast<int>(array.shape(0)),
        static_cast<int>(array.shape(1)),
        static_cast<int>(array.shape(2)),
    };
}

void validate_values(const py::array& array, const char* name) {
    const auto size = static_cast<std::size_t>(array.size());
    const auto* values = static_cast<const double*>(array.data());
    for (std::size_t i = 0; i < size; ++i) {
        if (!std::isfinite(values[i])) {
            throw py::value_error(std::string(name) + " must contain only finite values");
        }
    }
}

py::dict solve_batch_binding(
    const py::array& log_q_mass,
    const py::array& forcing,
    double dx,
    int maximum_mode,
    double rank_relative_tolerance,
    double weak_relative_tolerance,
    double eigensolver_tolerance,
    int maximum_eigensolver_sweeps) {
    const Shape3 shape = validate_array(log_q_mass, "log_q_mass");
    const Shape3 forcing_shape = validate_array(forcing, "forcing");
    if (shape.batch != forcing_shape.batch || shape.height != forcing_shape.height
        || shape.width != forcing_shape.width) {
        throw py::value_error("forcing must have the same [B,H,W] shape as log_q_mass");
    }
    validate_values(log_q_mass, "log_q_mass");
    validate_values(forcing, "forcing");
    if (!(dx > 0.0) || !std::isfinite(dx)) {
        throw py::value_error("dx must be finite and positive");
    }
    if (maximum_mode < 1 || maximum_mode >= std::min(shape.height, shape.width)) {
        throw py::value_error("maximum_mode must lie in [1,min(H,W)-1]");
    }
    for (const auto value : {
             rank_relative_tolerance, weak_relative_tolerance, eigensolver_tolerance}) {
        if (!(value > 0.0) || !std::isfinite(value)) {
            throw py::value_error("all tolerances must be finite and positive");
        }
    }
    if (maximum_eigensolver_sweeps < 1) {
        throw py::value_error("maximum_eigensolver_sweeps must be positive");
    }

    py::array_t<double> potential({shape.batch, shape.height, shape.width});
    py::array_t<double> action(shape.batch);
    py::array_t<double> objective(shape.batch);
    py::array_t<double> weak_relative_residual(shape.batch);
    py::array_t<double> scaled_weak_relative_residual(shape.batch);
    py::array_t<double> gauge_residual(shape.batch);
    py::array_t<double> compatibility_residual(shape.batch);
    py::array_t<double> compatibility_relative_residual(shape.batch);
    py::array_t<double> energy_load_identity_relative_error(shape.batch);
    py::array_t<double> condition_proxy(shape.batch);
    py::array_t<double> retained_rank(shape.batch);
    py::array_t<double> basis_size(shape.batch);
    py::array_t<double> eigensolver_sweeps(shape.batch);
    py::array_t<double> quadrature_underflow_count(shape.batch);
    py::array_t<double> converged(shape.batch);
    std::vector<SolveStats> stats(static_cast<std::size_t>(shape.batch));

    {
        py::gil_scoped_release release;
        mfsi_variational_poisson::solve_batch(
            static_cast<const double*>(log_q_mass.data()),
            static_cast<const double*>(forcing.data()),
            potential.mutable_data(),
            stats.data(),
            shape.batch,
            shape.height,
            shape.width,
            dx,
            maximum_mode,
            rank_relative_tolerance,
            weak_relative_tolerance,
            eigensolver_tolerance,
            maximum_eigensolver_sweeps);
    }
    for (int b = 0; b < shape.batch; ++b) {
        action.mutable_at(b) = stats[b].action;
        objective.mutable_at(b) = stats[b].objective;
        weak_relative_residual.mutable_at(b) = stats[b].weak_relative_residual;
        scaled_weak_relative_residual.mutable_at(b) = stats[b].scaled_weak_relative_residual;
        gauge_residual.mutable_at(b) = stats[b].gauge_residual;
        compatibility_residual.mutable_at(b) = stats[b].compatibility_residual;
        compatibility_relative_residual.mutable_at(b)
            = stats[b].compatibility_relative_residual;
        energy_load_identity_relative_error.mutable_at(b)
            = stats[b].energy_load_identity_relative_error;
        condition_proxy.mutable_at(b) = stats[b].condition_proxy;
        retained_rank.mutable_at(b) = stats[b].retained_rank;
        basis_size.mutable_at(b) = stats[b].basis_size;
        eigensolver_sweeps.mutable_at(b) = stats[b].eigensolver_sweeps;
        quadrature_underflow_count.mutable_at(b) = stats[b].quadrature_underflow_count;
        converged.mutable_at(b) = stats[b].converged ? 1.0 : 0.0;
    }
    return py::dict(
        "potential"_a = std::move(potential),
        "action"_a = std::move(action),
        "objective"_a = std::move(objective),
        "weak_relative_residual"_a = std::move(weak_relative_residual),
        "scaled_weak_relative_residual"_a = std::move(scaled_weak_relative_residual),
        "gauge_residual"_a = std::move(gauge_residual),
        "compatibility_residual"_a = std::move(compatibility_residual),
        "compatibility_relative_residual"_a = std::move(compatibility_relative_residual),
        "energy_load_identity_relative_error"_a
            = std::move(energy_load_identity_relative_error),
        "condition_proxy"_a = std::move(condition_proxy),
        "retained_rank"_a = std::move(retained_rank),
        "basis_size"_a = std::move(basis_size),
        "eigensolver_sweeps"_a = std::move(eigensolver_sweeps),
        "quadrature_underflow_count"_a = std::move(quadrature_underflow_count),
        "converged"_a = std::move(converged));
}

}  // namespace

PYBIND11_MODULE(_variational_poisson_native, module) {
    module.doc() = "Weak cosine-Galerkin weighted-Poisson solver";
    module.attr("__version__") = "1";
    module.def(
        "solve_batch",
        &solve_batch_binding,
        py::arg("log_q_mass").noconvert(),
        py::arg("forcing").noconvert(),
        py::arg("dx"),
        py::arg("maximum_mode"),
        py::arg("rank_relative_tolerance"),
        py::arg("weak_relative_tolerance"),
        py::arg("eigensolver_tolerance"),
        py::arg("maximum_eigensolver_sweeps"));
}

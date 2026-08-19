#include "periodic_poisson3d.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;
using mfsi_active_poisson3d::SolveStats;

namespace {

struct Shape4 {
    int batch;
    int nx;
    int ny;
    int ntheta;
};

Shape4 validate_array(const py::array& array, const char* name) {
    if (!array.dtype().is(py::dtype::of<double>())) {
        throw py::type_error(std::string(name) + " must have dtype float64");
    }
    if (array.ndim() != 4) {
        throw py::value_error(
            std::string(name) + " must have rank 4 [B,Nx,Ny,Ntheta]");
    }
    if (!(array.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    if (array.shape(0) < 1 || array.shape(1) < 3 || array.shape(2) < 3
        || array.shape(3) < 3) {
        throw py::value_error(
            std::string(name) + " requires B >= 1 and every periodic axis >= 3");
    }
    return {
        static_cast<int>(array.shape(0)),
        static_cast<int>(array.shape(1)),
        static_cast<int>(array.shape(2)),
        static_cast<int>(array.shape(3)),
    };
}

void require_same_shape(Shape4 expected, Shape4 actual, const char* name) {
    if (expected.batch != actual.batch || expected.nx != actual.nx
        || expected.ny != actual.ny || expected.ntheta != actual.ntheta) {
        throw py::value_error(
            std::string(name) + " must have the same [B,Nx,Ny,Ntheta] shape");
    }
}

void validate_spacings(double dx, double dy, double dtheta_metric) {
    if (!(dx > 0.0) || !std::isfinite(dx)) {
        throw py::value_error("dx must be finite and positive");
    }
    if (!(dy > 0.0) || !std::isfinite(dy)) {
        throw py::value_error("dy must be finite and positive");
    }
    if (!(dtheta_metric > 0.0) || !std::isfinite(dtheta_metric)) {
        throw py::value_error("dtheta_metric must be finite and positive");
    }
}

void validate_numerics(
    double dx,
    double dy,
    double dtheta_metric,
    double gauge_strength,
    double tolerance,
    int maximum_iterations) {
    validate_spacings(dx, dy, dtheta_metric);
    if (!(gauge_strength > 0.0) || !std::isfinite(gauge_strength)) {
        throw py::value_error("gauge_strength must be finite and positive");
    }
    if (!(tolerance > 0.0) || !std::isfinite(tolerance)) {
        throw py::value_error("tolerance must be finite and positive");
    }
    if (maximum_iterations < 1) {
        throw py::value_error("maximum_iterations must be positive");
    }
}

py::array_t<double> empty_like(Shape4 shape) {
    return py::array_t<double>({shape.batch, shape.nx, shape.ny, shape.ntheta});
}

std::size_t grid_size(Shape4 shape) {
    return static_cast<std::size_t>(shape.nx) * shape.ny * shape.ntheta;
}

py::dict solve_batch_binding(
    const py::array& q_operator,
    const py::array& rhs,
    const py::array& gauge,
    double dx,
    double dy,
    double dtheta_metric,
    double gauge_strength,
    double tolerance,
    int maximum_iterations,
    const py::object& initial_guess_object) {
    const Shape4 shape = validate_array(q_operator, "q_operator");
    require_same_shape(shape, validate_array(rhs, "rhs"), "rhs");
    require_same_shape(shape, validate_array(gauge, "gauge"), "gauge");
    validate_numerics(
        dx, dy, dtheta_metric, gauge_strength, tolerance, maximum_iterations);

    const double* initial_guess = nullptr;
    py::array initial_guess_array;
    if (!initial_guess_object.is_none()) {
        initial_guess_array = py::cast<py::array>(initial_guess_object);
        require_same_shape(
            shape,
            validate_array(initial_guess_array, "initial_guess"),
            "initial_guess");
        initial_guess = static_cast<const double*>(initial_guess_array.data());
    }

    auto potential = empty_like(shape);
    py::array_t<std::int32_t> iterations(shape.batch);
    py::array_t<double> relative_residual(shape.batch);
    py::array_t<bool> converged(shape.batch);
    std::vector<SolveStats> stats(static_cast<std::size_t>(shape.batch));
    {
        py::gil_scoped_release release;
        mfsi_active_poisson3d::solve_batch(
            static_cast<const double*>(q_operator.data()),
            static_cast<const double*>(rhs.data()),
            static_cast<const double*>(gauge.data()),
            initial_guess,
            potential.mutable_data(),
            stats.data(),
            shape.batch,
            shape.nx,
            shape.ny,
            shape.ntheta,
            dx,
            dy,
            dtheta_metric,
            gauge_strength,
            tolerance,
            maximum_iterations);
    }
    for (int batch_index = 0; batch_index < shape.batch; ++batch_index) {
        iterations.mutable_at(batch_index) = stats[batch_index].iterations;
        relative_residual.mutable_at(batch_index) = stats[batch_index].relative_residual;
        converged.mutable_at(batch_index) = stats[batch_index].converged;
    }
    return py::dict(
        "potential"_a = std::move(potential),
        "iterations"_a = std::move(iterations),
        "relative_residual"_a = std::move(relative_residual),
        "converged"_a = std::move(converged));
}

}  // namespace

PYBIND11_MODULE(_active_nematic_poisson3d_native, module) {
    module.doc() =
        "C++17/OpenMP batched anisotropic 3D periodic weighted-Poisson solver";
    module.attr("__version__") = "1";

    module.def(
        "solve_batch",
        &solve_batch_binding,
        py::arg("q_operator").noconvert(),
        py::arg("rhs").noconvert(),
        py::arg("gauge").noconvert(),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dtheta_metric"),
        py::arg("gauge_strength"),
        py::arg("tolerance"),
        py::arg("maximum_iterations"),
        py::arg("initial_guess") = py::none());

    module.def(
        "weighted_laplacian_batch",
        [](const py::array& potential, const py::array& q, double dx, double dy,
           double dtheta_metric) {
            const Shape4 shape = validate_array(potential, "potential");
            require_same_shape(shape, validate_array(q, "q"), "q");
            validate_spacings(dx, dy, dtheta_metric);
            auto out = empty_like(shape);
            const std::size_t size = grid_size(shape);
            {
                py::gil_scoped_release release;
                #pragma omp parallel for schedule(static)
                for (int batch_index = 0; batch_index < shape.batch; ++batch_index) {
                    const std::size_t offset
                        = static_cast<std::size_t>(batch_index) * size;
                    mfsi_active_poisson3d::weighted_laplacian(
                        static_cast<const double*>(potential.data()) + offset,
                        static_cast<const double*>(q.data()) + offset,
                        out.mutable_data() + offset,
                        shape.nx,
                        shape.ny,
                        shape.ntheta,
                        dx,
                        dy,
                        dtheta_metric);
                }
            }
            return out;
        },
        py::arg("potential").noconvert(),
        py::arg("q").noconvert(),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dtheta_metric"));

    module.def(
        "diagonal_batch",
        [](const py::array& q, const py::array& gauge, double dx, double dy,
           double dtheta_metric, double gauge_strength) {
            const Shape4 shape = validate_array(q, "q");
            require_same_shape(shape, validate_array(gauge, "gauge"), "gauge");
            validate_numerics(dx, dy, dtheta_metric, gauge_strength, 1.0, 1);
            auto out = empty_like(shape);
            const std::size_t size = grid_size(shape);
            {
                py::gil_scoped_release release;
                #pragma omp parallel for schedule(static)
                for (int batch_index = 0; batch_index < shape.batch; ++batch_index) {
                    const std::size_t offset
                        = static_cast<std::size_t>(batch_index) * size;
                    mfsi_active_poisson3d::weighted_laplacian_diag(
                        static_cast<const double*>(q.data()) + offset,
                        static_cast<const double*>(gauge.data()) + offset,
                        out.mutable_data() + offset,
                        shape.nx,
                        shape.ny,
                        shape.ntheta,
                        dx,
                        dy,
                        dtheta_metric,
                        gauge_strength);
                }
            }
            return out;
        },
        py::arg("q").noconvert(),
        py::arg("gauge").noconvert(),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dtheta_metric"),
        py::arg("gauge_strength"));

    module.def(
        "weighted_operator_vjp",
        [](const py::array& potential, const py::array& adjoint, double dx,
           double dy, double dtheta_metric) {
            const Shape4 shape = validate_array(potential, "potential");
            require_same_shape(shape, validate_array(adjoint, "adjoint"), "adjoint");
            validate_spacings(dx, dy, dtheta_metric);
            auto out = empty_like(shape);
            {
                py::gil_scoped_release release;
                mfsi_active_poisson3d::weighted_operator_vjp_batch(
                    static_cast<const double*>(potential.data()),
                    static_cast<const double*>(adjoint.data()),
                    out.mutable_data(),
                    shape.batch,
                    shape.nx,
                    shape.ny,
                    shape.ntheta,
                    dx,
                    dy,
                    dtheta_metric);
            }
            return out;
        },
        py::arg("potential").noconvert(),
        py::arg("adjoint").noconvert(),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dtheta_metric"));

    module.def(
        "gauge_vjp",
        [](const py::array& potential, const py::array& adjoint,
           const py::array& gauge, double gauge_strength) {
            const Shape4 shape = validate_array(potential, "potential");
            require_same_shape(shape, validate_array(adjoint, "adjoint"), "adjoint");
            require_same_shape(shape, validate_array(gauge, "gauge"), "gauge");
            if (!(gauge_strength > 0.0) || !std::isfinite(gauge_strength)) {
                throw py::value_error("gauge_strength must be finite and positive");
            }
            auto out = empty_like(shape);
            {
                py::gil_scoped_release release;
                mfsi_active_poisson3d::gauge_vjp_batch(
                    static_cast<const double*>(potential.data()),
                    static_cast<const double*>(adjoint.data()),
                    static_cast<const double*>(gauge.data()),
                    out.mutable_data(),
                    shape.batch,
                    shape.nx,
                    shape.ny,
                    shape.ntheta,
                    gauge_strength);
            }
            return out;
        },
        py::arg("potential").noconvert(),
        py::arg("adjoint").noconvert(),
        py::arg("gauge").noconvert(),
        py::arg("gauge_strength"));

    module.def(
        "linearized_rhs",
        [](const py::array& potential, const py::array& q_dot,
           const py::array& rhs_dot, const py::array& gauge,
           const py::array& gauge_dot, double dx, double dy,
           double dtheta_metric, double gauge_strength) {
            const Shape4 shape = validate_array(potential, "potential");
            require_same_shape(shape, validate_array(q_dot, "q_dot"), "q_dot");
            require_same_shape(shape, validate_array(rhs_dot, "rhs_dot"), "rhs_dot");
            require_same_shape(shape, validate_array(gauge, "gauge"), "gauge");
            require_same_shape(
                shape, validate_array(gauge_dot, "gauge_dot"), "gauge_dot");
            validate_numerics(dx, dy, dtheta_metric, gauge_strength, 1.0, 1);
            auto out = empty_like(shape);
            {
                py::gil_scoped_release release;
                mfsi_active_poisson3d::linearized_rhs_batch(
                    static_cast<const double*>(potential.data()),
                    static_cast<const double*>(q_dot.data()),
                    static_cast<const double*>(rhs_dot.data()),
                    static_cast<const double*>(gauge.data()),
                    static_cast<const double*>(gauge_dot.data()),
                    out.mutable_data(),
                    shape.batch,
                    shape.nx,
                    shape.ny,
                    shape.ntheta,
                    dx,
                    dy,
                    dtheta_metric,
                    gauge_strength);
            }
            return out;
        },
        py::arg("potential").noconvert(),
        py::arg("q_dot").noconvert(),
        py::arg("rhs_dot").noconvert(),
        py::arg("gauge").noconvert(),
        py::arg("gauge_dot").noconvert(),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dtheta_metric"),
        py::arg("gauge_strength"));
}

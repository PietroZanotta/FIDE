#include "poisson_solver.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;
using mfsi_poisson::SolveStats;

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
    if (array.shape(0) < 1 || array.shape(1) < 1 || array.shape(2) < 1) {
        throw py::value_error(std::string(name) + " dimensions must be positive");
    }
    return {
        static_cast<int>(array.shape(0)),
        static_cast<int>(array.shape(1)),
        static_cast<int>(array.shape(2)),
    };
}

void require_same_shape(Shape3 expected, Shape3 actual, const char* name) {
    if (expected.batch != actual.batch || expected.height != actual.height
        || expected.width != actual.width) {
        throw py::value_error(std::string(name) + " must have the same [B,H,W] shape");
    }
}

void validate_numerics(double dx, double gauge_strength, double tol, int maxiter) {
    if (!(dx > 0.0) || !std::isfinite(dx)) {
        throw py::value_error("dx must be finite and positive");
    }
    if (!(gauge_strength > 0.0) || !std::isfinite(gauge_strength)) {
        throw py::value_error("gauge_strength must be finite and positive");
    }
    if (!(tol > 0.0) || !std::isfinite(tol)) {
        throw py::value_error("tol must be finite and positive");
    }
    if (maxiter < 1) {
        throw py::value_error("maxiter must be positive");
    }
}

py::array_t<double> empty_like(Shape3 shape) {
    return py::array_t<double>({shape.batch, shape.height, shape.width});
}

py::dict solve_batch_binding(
    const py::array& q_operator,
    const py::array& rhs,
    const py::array& gauge,
    double dx,
    double gauge_strength,
    double tol,
    int maxiter) {
    const Shape3 shape = validate_array(q_operator, "q_operator");
    require_same_shape(shape, validate_array(rhs, "rhs"), "rhs");
    require_same_shape(shape, validate_array(gauge, "gauge"), "gauge");
    validate_numerics(dx, gauge_strength, tol, maxiter);

    auto psi = empty_like(shape);
    py::array_t<std::int32_t> iterations(shape.batch);
    py::array_t<double> relative_residual(shape.batch);
    py::array_t<bool> converged(shape.batch);
    std::vector<SolveStats> stats(static_cast<std::size_t>(shape.batch));

    {
        py::gil_scoped_release release;
        mfsi_poisson::solve_batch(
            static_cast<const double*>(q_operator.data()),
            static_cast<const double*>(rhs.data()),
            static_cast<const double*>(gauge.data()),
            psi.mutable_data(),
            stats.data(),
            shape.batch,
            shape.height,
            shape.width,
            dx,
            gauge_strength,
            tol,
            maxiter);
    }
    for (int b = 0; b < shape.batch; ++b) {
        iterations.mutable_at(b) = stats[b].iterations;
        relative_residual.mutable_at(b) = stats[b].relative_residual;
        converged.mutable_at(b) = stats[b].converged;
    }
    return py::dict(
        "psi"_a = std::move(psi),
        "iterations"_a = std::move(iterations),
        "relative_residual"_a = std::move(relative_residual),
        "converged"_a = std::move(converged));
}

}  // namespace

PYBIND11_MODULE(_poisson_native, module) {
    using namespace pybind11::literals;
    module.doc() = "C++17/OpenMP batched weighted-Poisson solver for the MFSI stage-4 proxy";
    module.attr("__version__") = "1";

    module.def(
        "solve_batch",
        &solve_batch_binding,
        py::arg("q_operator").noconvert(),
        py::arg("rhs").noconvert(),
        py::arg("gauge").noconvert(),
        py::arg("dx"),
        py::arg("gauge_strength"),
        py::arg("tol"),
        py::arg("maxiter"));

    module.def(
        "weighted_laplacian_batch",
        [](const py::array& psi, const py::array& q, double dx) {
            const Shape3 shape = validate_array(psi, "psi");
            require_same_shape(shape, validate_array(q, "q"), "q");
            if (!(dx > 0.0) || !std::isfinite(dx)) {
                throw py::value_error("dx must be finite and positive");
            }
            auto out = empty_like(shape);
            const std::size_t size = static_cast<std::size_t>(shape.height) * shape.width;
            {
                py::gil_scoped_release release;
                #pragma omp parallel for schedule(static)
                for (int b = 0; b < shape.batch; ++b) {
                    const std::size_t offset = static_cast<std::size_t>(b) * size;
                    mfsi_poisson::weighted_laplacian(
                        static_cast<const double*>(psi.data()) + offset,
                        static_cast<const double*>(q.data()) + offset,
                        out.mutable_data() + offset,
                        shape.height,
                        shape.width,
                        dx);
                }
            }
            return out;
        },
        py::arg("psi").noconvert(),
        py::arg("q").noconvert(),
        py::arg("dx"));

    module.def(
        "diagonal_batch",
        [](const py::array& q, const py::array& gauge, double dx, double gauge_strength) {
            const Shape3 shape = validate_array(q, "q");
            require_same_shape(shape, validate_array(gauge, "gauge"), "gauge");
            validate_numerics(dx, gauge_strength, 1.0, 1);
            auto out = empty_like(shape);
            const std::size_t size = static_cast<std::size_t>(shape.height) * shape.width;
            {
                py::gil_scoped_release release;
                #pragma omp parallel for schedule(static)
                for (int b = 0; b < shape.batch; ++b) {
                    const std::size_t offset = static_cast<std::size_t>(b) * size;
                    mfsi_poisson::weighted_laplacian_diag(
                        static_cast<const double*>(q.data()) + offset,
                        static_cast<const double*>(gauge.data()) + offset,
                        out.mutable_data() + offset,
                        shape.height,
                        shape.width,
                        dx,
                        gauge_strength);
                }
            }
            return out;
        },
        py::arg("q").noconvert(),
        py::arg("gauge").noconvert(),
        py::arg("dx"),
        py::arg("gauge_strength"));

    module.def(
        "weighted_operator_vjp",
        [](const py::array& psi, const py::array& lambda, double dx) {
            const Shape3 shape = validate_array(psi, "psi");
            require_same_shape(shape, validate_array(lambda, "lambda"), "lambda");
            if (!(dx > 0.0) || !std::isfinite(dx)) {
                throw py::value_error("dx must be finite and positive");
            }
            auto out = empty_like(shape);
            {
                py::gil_scoped_release release;
                mfsi_poisson::weighted_operator_vjp_batch(
                    static_cast<const double*>(psi.data()),
                    static_cast<const double*>(lambda.data()),
                    out.mutable_data(),
                    shape.batch,
                    shape.height,
                    shape.width,
                    dx);
            }
            return out;
        },
        py::arg("psi").noconvert(),
        py::arg("lambda").noconvert(),
        py::arg("dx"));

    module.def(
        "gauge_vjp",
        [](const py::array& psi, const py::array& lambda, const py::array& gauge,
           double gauge_strength) {
            const Shape3 shape = validate_array(psi, "psi");
            require_same_shape(shape, validate_array(lambda, "lambda"), "lambda");
            require_same_shape(shape, validate_array(gauge, "gauge"), "gauge");
            if (!(gauge_strength > 0.0) || !std::isfinite(gauge_strength)) {
                throw py::value_error("gauge_strength must be finite and positive");
            }
            auto out = empty_like(shape);
            {
                py::gil_scoped_release release;
                mfsi_poisson::gauge_vjp_batch(
                    static_cast<const double*>(psi.data()),
                    static_cast<const double*>(lambda.data()),
                    static_cast<const double*>(gauge.data()),
                    out.mutable_data(),
                    shape.batch,
                    shape.height,
                    shape.width,
                    gauge_strength);
            }
            return out;
        },
        py::arg("psi").noconvert(),
        py::arg("lambda").noconvert(),
        py::arg("gauge").noconvert(),
        py::arg("gauge_strength"));

    module.def(
        "linearized_rhs",
        [](const py::array& psi, const py::array& q_dot, const py::array& rhs_dot,
           const py::array& gauge, const py::array& gauge_dot, double dx,
           double gauge_strength) {
            const Shape3 shape = validate_array(psi, "psi");
            require_same_shape(shape, validate_array(q_dot, "q_dot"), "q_dot");
            require_same_shape(shape, validate_array(rhs_dot, "rhs_dot"), "rhs_dot");
            require_same_shape(shape, validate_array(gauge, "gauge"), "gauge");
            require_same_shape(shape, validate_array(gauge_dot, "gauge_dot"), "gauge_dot");
            validate_numerics(dx, gauge_strength, 1.0, 1);
            auto out = empty_like(shape);
            {
                py::gil_scoped_release release;
                mfsi_poisson::linearized_rhs_batch(
                    static_cast<const double*>(psi.data()),
                    static_cast<const double*>(q_dot.data()),
                    static_cast<const double*>(rhs_dot.data()),
                    static_cast<const double*>(gauge.data()),
                    static_cast<const double*>(gauge_dot.data()),
                    out.mutable_data(),
                    shape.batch,
                    shape.height,
                    shape.width,
                    dx,
                    gauge_strength);
            }
            return out;
        },
        py::arg("psi").noconvert(),
        py::arg("q_dot").noconvert(),
        py::arg("rhs_dot").noconvert(),
        py::arg("gauge").noconvert(),
        py::arg("gauge_dot").noconvert(),
        py::arg("dx"),
        py::arg("gauge_strength"));
}

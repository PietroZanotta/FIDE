#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cblas.h>
#include <stdexcept>
#include <vector>

#include <omp.h>

#ifdef MFSI_HAVE_SCIPY_OPENBLAS
extern "C" void scipy_cblas_dgemm(
    const CBLAS_LAYOUT, const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
    const int, const int, const int, const double, const double*, const int,
    const double*, const int, const double, double*, const int);
#endif

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

struct Shape { int samples; int basis; int particles; int dimensions; };

Shape validate(const py::array& values, const py::array& gradients,
               const py::array& weights, const py::array& forcing) {
    for (const auto* item : {&values, &gradients, &weights, &forcing}) {
        if (!item->dtype().is(py::dtype::of<double>()))
            throw py::type_error("Galerkin arrays must have dtype float64");
        if (!(item->flags() & py::array::c_style))
            throw py::value_error("Galerkin arrays must be C-contiguous");
    }
    if (values.ndim() != 2 || gradients.ndim() != 4
        || weights.ndim() != 1 || forcing.ndim() != 1)
        throw py::value_error("expected [N,K], [N,K,P,D], [N], [N]");
    Shape s{static_cast<int>(values.shape(0)), static_cast<int>(values.shape(1)),
            static_cast<int>(gradients.shape(2)), static_cast<int>(gradients.shape(3))};
    if (s.samples < 1 || s.basis < 1 || s.particles < 1 || s.dimensions < 1
        || gradients.shape(0) != s.samples || gradients.shape(1) != s.basis
        || weights.shape(0) != s.samples || forcing.shape(0) != s.samples)
        throw py::value_error("inconsistent or empty Galerkin shapes");
    return s;
}

py::dict assemble(const py::array& values_array, const py::array& gradients_array,
                  const py::array& weights_array, const py::array& forcing_array) {
    const Shape s = validate(values_array, gradients_array, weights_array, forcing_array);
    const auto* values = static_cast<const double*>(values_array.data());
    const auto* gradients = static_cast<const double*>(gradients_array.data());
    const auto* weights = static_cast<const double*>(weights_array.data());
    const auto* forcing = static_cast<const double*>(forcing_array.data());
    for (int n = 0; n < s.samples; ++n)
        if (!std::isfinite(weights[n]) || weights[n] < 0.0 || !std::isfinite(forcing[n]))
            throw py::value_error("weights must be finite/nonnegative and forcing finite");

    py::array_t<double> gram_out({s.basis, s.basis});
    py::array_t<double> raw_load_out({s.basis});
    py::array_t<double> mean_out({s.basis});
    py::array_t<double> forcing_sum_out({1});
    auto* gram = gram_out.mutable_data(); auto* raw_load = raw_load_out.mutable_data();
    auto* mean = mean_out.mutable_data(); auto* forcing_sum = forcing_sum_out.mutable_data();
    std::fill(gram, gram + gram_out.size(), 0.0);
    std::fill(raw_load, raw_load + raw_load_out.size(), 0.0);
    std::fill(mean, mean + mean_out.size(), 0.0);

    const std::size_t q_count = static_cast<std::size_t>(s.particles) * s.dimensions;
    const std::size_t rows = static_cast<std::size_t>(s.samples) * q_count;
    std::vector<double> weighted_gradients(rows * s.basis);
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < s.samples; ++n) {
            const double scale = std::sqrt(weights[n]);
            for (std::size_t q = 0; q < q_count; ++q) {
                const std::size_t destination =
                    (static_cast<std::size_t>(n) * q_count + q) * s.basis;
                for (int k = 0; k < s.basis; ++k)
                    weighted_gradients[destination + k] = scale * gradients[
                        (static_cast<std::size_t>(n) * s.basis + k) * q_count + q];
            }
        }
        #ifdef MFSI_HAVE_SCIPY_OPENBLAS
        scipy_cblas_dgemm(
        #else
        cblas_dgemm(
        #endif
            CblasRowMajor, CblasTrans, CblasNoTrans, s.basis, s.basis,
            static_cast<int>(rows), 1.0, weighted_gradients.data(), s.basis,
            weighted_gradients.data(), s.basis, 0.0, gram, s.basis);
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < s.basis; ++k) {
            double m = 0.0, f = 0.0;
            for (int n = 0; n < s.samples; ++n) {
                const double v = values[static_cast<std::size_t>(n) * s.basis + k];
                m += weights[n] * v; f += weights[n] * forcing[n] * v;
            }
            mean[k] = m; raw_load[k] = f;
        }
        double f = 0.0;
        for (int n = 0; n < s.samples; ++n) f += weights[n] * forcing[n];
        forcing_sum[0] = f;
    }
    return py::dict("gram"_a=std::move(gram_out), "raw_load"_a=std::move(raw_load_out),
                    "basis_mean"_a=std::move(mean_out),
                    "forcing_sum"_a=std::move(forcing_sum_out));
}
}  // namespace

PYBIND11_MODULE(_galerkin_native, module) {
    module.doc() = "C++17/OpenMP/BLAS fixed-feature Galerkin chunk assembler";
    module.attr("__version__") = "1";
    module.def("assemble_chunk", &assemble, py::arg("values").noconvert(),
               py::arg("gradients").noconvert(), py::arg("weights").noconvert(),
               py::arg("forcing").noconvert());
}


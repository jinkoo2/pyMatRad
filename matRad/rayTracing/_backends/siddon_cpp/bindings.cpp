/**
 * bindings.cpp — pybind11 bindings for the C++ Siddon backend.
 *
 * Compatible with pybind11 v2.x and v3.x.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "siddon.hpp"

namespace py = pybind11;

static py::tuple siddon_ray_tracer_py(
    py::array_t<double> isocenter_cube,
    py::dict            resolution,
    py::array_t<double> source_point,
    py::array_t<double> target_point,
    py::list            cubes
) {
    // Access input arrays
    auto iso = isocenter_cube.unchecked<1>();
    auto sp  = source_point.unchecked<1>();
    auto tp  = target_point.unchecked<1>();

    double rx = resolution["x"].cast<double>();
    double ry = resolution["y"].cast<double>();
    double rz = resolution["z"].cast<double>();

    // Get first cube as Fortran-order array
    auto np = py::module_::import("numpy");
    py::array cube0_fort = np.attr("asfortranarray")(cubes[0]).attr("astype")(py::str("float64"));
    py::buffer_info buf  = cube0_fort.request();

    int Ny = (int)buf.shape[0];
    int Nx = (int)buf.shape[1];
    int Nz = (int)buf.shape[2];
    const double *cube_data = static_cast<const double *>(buf.ptr);

    siddon::SiddonResult result = siddon::trace(
        iso(0), iso(1), iso(2),
        rx, ry, rz,
        sp(0), sp(1), sp(2),
        tp(0), tp(1), tp(2),
        cube_data, Ny, Nx, Nz
    );

    if (result.alphas.empty()) {
        py::list empty_rho;
        py::ssize_t ncubes = static_cast<py::ssize_t>(cubes.size());
        for (py::ssize_t i = 0; i < ncubes; i++)
            empty_rho.append(py::array_t<double>(0));
        return py::make_tuple(
            py::array_t<double>(0),
            py::array_t<double>(0),
            empty_rho,
            result.d12,
            py::array_t<long long>(0)
        );
    }

    int n_seg = (int)result.l.size();

    // Build output numpy arrays
    py::array_t<double>    alphas_out(n_seg + 1);
    py::array_t<double>    l_out(n_seg);
    py::array_t<long long> ix_out(n_seg);

    std::copy(result.alphas.begin(), result.alphas.end(), alphas_out.mutable_data());
    std::copy(result.l.begin(),      result.l.end(),      l_out.mutable_data());
    std::copy(result.ix.begin(),     result.ix.end(),     ix_out.mutable_data());

    // Build rho_list
    py::list rho_list;

    // First cube rho (already computed)
    {
        py::array_t<double> rho0(n_seg);
        std::copy(result.rho.begin(), result.rho.end(), rho0.mutable_data());
        rho_list.append(rho0);
    }

    // Remaining cubes
    py::ssize_t ncubes = static_cast<py::ssize_t>(cubes.size());
    for (py::ssize_t ci = 1; ci < ncubes; ci++) {
        py::array ci_fort = np.attr("asfortranarray")(cubes[ci]).attr("astype")(py::str("float64"));
        py::buffer_info ci_buf = ci_fort.request();
        const double *cd = static_cast<const double *>(ci_buf.ptr);

        py::array_t<double> rho_i(n_seg);
        for (int s = 0; s < n_seg; s++)
            rho_i.mutable_at(s) = cd[result.ix[s] - 1];
        rho_list.append(rho_i);
    }

    return py::make_tuple(alphas_out, l_out, rho_list, result.d12, ix_out);
}

PYBIND11_MODULE(siddon_cpp, m) {
    m.doc() = "pybind11 C++ Siddon ray-tracer backend";
    m.def("siddon_ray_tracer", &siddon_ray_tracer_py,
          py::arg("isocenter_cube"),
          py::arg("resolution"),
          py::arg("source_point"),
          py::arg("target_point"),
          py::arg("cubes"),
          "Siddon ray tracer — drop-in for matRad.rayTracing.siddon.siddon_ray_tracer");
}

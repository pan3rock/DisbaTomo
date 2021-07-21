#include "gradient.hpp"

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace Eigen;

PYBIND11_MODULE(dpcxx, m) {
  pybind11::class_<GradientEval>(m, "GradientEval")
      .def(pybind11::init<const Ref<const ArrayXXd> &, const std::string &>())
      .def("compute", &GradientEval::compute);
}
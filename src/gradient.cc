#include "gradient.hpp"
#include "gradient_psv.hpp"
#include "gradient_sh.hpp"

#include <Eigen/Dense>
#include <exception>
#include <fmt/format.h>
#include <memory>

using namespace Eigen;
using grad_psv::GradientPSV;
using grad_sh::GradientSH;

Gradient::Gradient(const Ref<const ArrayXXd> &model)
    : model_(model), nl_(model.rows()) {}

Gradient::~Gradient() = default;

GradientEval::GradientEval(const Ref<const ArrayXXd> &model,
                           const std::string &type) {
  if (type == "rayleigh") {
    grad_ = std::make_unique<GradientPSV>(model);
  } else if (type == "love") {
    grad_ = std::make_unique<GradientSH>(model);
  } else {
    std::string msg = fmt::format("type {:s} is invalid.", type);
    throw std::runtime_error(msg);
  }
}

ArrayXd GradientEval::compute(const double freq, const double c) {
  ArrayXd kvs = grad_->compute(freq, c);
  return kvs;
}

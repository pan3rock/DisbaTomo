#ifndef GRADIENT_H_
#define GRADIENT_H_

#include <Eigen/Dense>
#include <memory>
#include <string>

class Gradient {
public:
  Gradient(const Eigen::Ref<const Eigen::ArrayXXd> &model);
  virtual ~Gradient();
  virtual Eigen::ArrayXd compute(const double freq, const double c) const = 0;

protected:
  const Eigen::ArrayXXd model_;
  const int nl_;
};

class GradientEval {
public:
  GradientEval(const Eigen::Ref<const Eigen::ArrayXXd> &model,
               const std::string &wave_type);
  Eigen::ArrayXd compute(const double freq, const double c);

private:
  std::unique_ptr<Gradient> grad_;
};
#endif
#ifndef GRADIENT_SH_H_
#define GRADIENT_SH_H_

#include "gradient.hpp"
#include <Eigen/Dense>
#include <complex>
#include <memory>
#include <vector>

using VecAry2cd =
    std::vector<Eigen::Array22cd, Eigen::aligned_allocator<Eigen::Array22cd>>;

namespace grad_sh {

class GRTCoeff {
  friend class IntegralLayer;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GRTCoeff(const Eigen::Ref<const Eigen::ArrayXXd> model, const double freq,
           const double c);

private:
  void initialize_nv();
  void initialize_E();
  std::complex<double> get_Ad(const double z, const int ind_layer) const;
  std::complex<double> get_Au(const double z, const int ind_layer) const;
  std::complex<double> get_Ad_der(const double z, const int ind_layer) const;
  std::complex<double> get_Au_der(const double z, const int ind_layer) const;

  void compute_rtc();
  void compute_grtc();
  void compute_CdCu();

  const Eigen::ArrayXd z_, rho_, beta_, alpha_, mu_;
  const int nl_;
  const std::complex<double> angfreq_;
  const double c_;

  Eigen::ArrayXcd nv_;
  std::vector<Eigen::Matrix2cd, Eigen::aligned_allocator<Eigen::Matrix2cd>> e_;
  Eigen::VectorXcd t_d_, r_ud_, r_du_, t_u_;
  Eigen::VectorXcd gt_d_, gr_ud_, gr_du_, gt_u_;
  Eigen::VectorXcd cd_, cu_;
};

class IntegralLayer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IntegralLayer(const Eigen::Ref<const Eigen::ArrayXXd> model,
                const double freq, const double c);

  double compute_I1();
  double compute_I2();
  double compute_I3();
  Eigen::ArrayXd compute_kvs();

private:
  void initialize_P();
  void initialize_sigma();

  double intker_ut2_top(int id_layer);
  double intker_dut2_top(int id_layer);

  double intker_ut2_bottom(int id_layer);
  double intker_dut2_bottom(int id_layer);

  void integrate_ut2();
  void integrate_dut2();

  const std::unique_ptr<GRTCoeff> grtc_;
  const int nl_;
  const double k_;
  const double pvel_;
  Eigen::ArrayXd z_, beta_, rho_, mu_;
  Eigen::ArrayXcd nv_;
  Eigen::ArrayXd thickness_;

  Eigen::VectorXcd cd_, cu_;
  VecAry2cd matP_u_u_;
  VecAry2cd matP_uc_u_;
  VecAry2cd matP_uc_uc_;
  VecAry2cd matP_du_du_;
  VecAry2cd matP_duc_du_;
  VecAry2cd matP_duc_duc_;
  VecAry2cd sigma_x_sigma_top_;
  VecAry2cd sigmac_x_sigma_top_;
  VecAry2cd sigma_x_sigmac_top_;
  VecAry2cd sigmac_x_sigmac_top_;
  VecAry2cd sigma_x_sigma_bottom_;
  VecAry2cd sigmac_x_sigma_bottom_;
  VecAry2cd sigma_x_sigmac_bottom_;
  VecAry2cd sigmac_x_sigmac_bottom_;

  Eigen::ArrayXd int_ut2_;
  Eigen::ArrayXd int_dut2_;
};

class GradientSH : public Gradient {
public:
  using Gradient::Gradient;
  ~GradientSH();
  Eigen::ArrayXd compute(const double freq, const double c) const override;
};

} // namespace grad_sh
#endif
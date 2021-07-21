#ifndef GRADIENT_PSV_H_
#define GRADIENT_PSV_H_

#include "gradient.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>

using VecMat4cd =
    std::vector<Eigen::Matrix4cd, Eigen::aligned_allocator<Eigen::Matrix4cd>>;
using VecMat24cd = std::vector<
    Eigen::Matrix<std::complex<double>, 2, 4>,
    Eigen::aligned_allocator<Eigen::Matrix<std::complex<double>, 2, 4>>>;

namespace grad_psv {
class GRTCoeff {
  friend class IntegralLayer;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GRTCoeff(const Eigen::Ref<const Eigen::ArrayXXd> model, const double freq,
           const double c);

private:
  void initialize_gamma();
  void initialize_nv();
  void initialize_E();
  Eigen::Matrix2cd get_Ad(const double z, const int ind_layer) const;
  Eigen::Matrix2cd get_Au(const double z, const int ind_layer) const;
  Eigen::Matrix2cd get_Ad_der(const double z, const int ind_layer) const;
  Eigen::Matrix2cd get_Au_der(const double z, const int ind_layer) const;

  void compute_rtc();
  void compute_grtc();
  void compute_CdCu();

  const Eigen::ArrayXd z_, rho_, beta_, alpha_, mu_;
  const int nl_;
  const std::complex<double> angfreq_;
  const double c_;

  Eigen::ArrayXcd gamma_, nv_;
  std::vector<Eigen::Matrix2cd, Eigen::aligned_allocator<Eigen::Matrix2cd>>
      e11_, e12_, e21_, e22_,   //
      t_d_, r_ud_, r_du_, t_u_, //
      gt_d_, gr_ud_, gr_du_, gt_u_;
  std::vector<Eigen::Vector2cd, Eigen::aligned_allocator<Eigen::Vector2cd>> Cd_,
      Cu_;

  const Eigen::Matrix2cd matI_ = Eigen::Matrix2cd::Identity();
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

  double intker_us2_top(int id_layer);
  double intker_ur2_top(int id_layer);
  double intker_dus2_top(int id_layer);
  double intker_dur2_top(int id_layer);
  double intker_urdus_top(int id_layer);
  double intker_usdur_top(int id_layer);

  double intker_us2_bottom(int id_layer);
  double intker_ur2_bottom(int id_layer);
  double intker_dus2_bottom(int id_layer);
  double intker_dur2_bottom(int id_layer);
  double intker_urdus_bottom(int id_layer);
  double intker_usdur_bottom(int id_layer);

  void integrate_us2();
  void integrate_ur2();
  void integrate_dus2();
  void integrate_dur2();
  void integrate_usdur();
  void integrate_urdus();

  const std::unique_ptr<GRTCoeff> grtc_;
  const int nl_;
  const double k_;
  const double pvel_;
  Eigen::ArrayXd z_, alpha_, beta_, rho_, mu_, lamb_;
  Eigen::ArrayXcd gamma_, nv_;
  Eigen::ArrayXcd thickness_;

  std::vector<Eigen::Vector2cd, Eigen::aligned_allocator<Eigen::Vector2cd>> Cd_,
      Cu_;
  VecMat24cd matE_;
  VecMat4cd matP_u_u_;
  VecMat4cd matP_uc_u_;
  VecMat4cd matP_uc_uc_;
  VecMat4cd matP_du_du_;
  VecMat4cd matP_duc_du_;
  VecMat4cd matP_duc_duc_;
  VecMat4cd matP_u_du_;
  VecMat4cd matP_uc_du_;
  VecMat4cd matP_u_duc_;
  VecMat4cd matP_uc_duc_;
  // top is the upper limit of integral
  VecMat4cd sigma_x_sigma_top_;
  VecMat4cd sigmac_x_sigma_top_;
  VecMat4cd sigma_x_sigmac_top_;
  VecMat4cd sigmac_x_sigmac_top_;
  // bottom is the lower limit
  VecMat4cd sigma_x_sigma_bottom_;
  VecMat4cd sigmac_x_sigma_bottom_;
  VecMat4cd sigma_x_sigmac_bottom_;
  VecMat4cd sigmac_x_sigmac_bottom_;

  Eigen::ArrayXd int_us2_;
  Eigen::ArrayXd int_ur2_;
  Eigen::ArrayXd int_dus2_;
  Eigen::ArrayXd int_dur2_;
  Eigen::ArrayXd int_urdus_;
  Eigen::ArrayXd int_usdur_;
};

class GradientPSV : public Gradient {
public:
  using Gradient::Gradient;
  ~GradientPSV();
  Eigen::ArrayXd compute(const double freq, const double c) const override;
};

} // namespace grad_psv

#endif
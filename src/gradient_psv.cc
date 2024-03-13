#include "gradient_psv.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fstream>
#include <memory>
#include <utility>

using namespace Eigen;
using namespace std::complex_literals;

using std::exp;
using std::pow;
using std::sqrt;

using complex_d = std::complex<double>;
using Mat42cd = Matrix<complex_d, 4, 2>;
using Ary24cd = Array<complex_d, 2, 4>;

const double PI = 3.14159265358979323846;

namespace grad_psv {

GRTCoeff::GRTCoeff(const Ref<const ArrayXXd> model, const double freq,
                   const double c)
    : z_(model.col(1)), rho_(model.col(2)), beta_(model.col(3)),
      alpha_(model.col(4)), mu_(rho_ * beta_.pow(2)), nl_(model.rows()),
      angfreq_(2.0 * PI * freq), c_(c), gamma_(nl_), nv_(nl_),
      e11_(nl_, Matrix2cd::Zero()), e12_(nl_, Matrix2cd::Zero()),
      e21_(nl_, Matrix2cd::Zero()), e22_(nl_, Matrix2cd::Zero()),
      t_d_(nl_, Matrix2cd::Zero()), r_ud_(nl_, Matrix2cd::Zero()),
      r_du_(nl_, Matrix2cd::Zero()), t_u_(nl_, Matrix2cd::Zero()),
      gt_d_(nl_, Matrix2cd::Zero()), gr_ud_(nl_, Matrix2cd::Zero()),
      gr_du_(nl_ + 1, Matrix2cd::Zero()), gt_u_(nl_, Matrix2cd::Zero()),
      Cd_(nl_, Vector2cd::Zero()), Cu_(nl_, Vector2cd::Zero()) {

  initialize_gamma();
  initialize_nv();
  initialize_E();

  compute_rtc();
  compute_grtc();
  compute_CdCu();
}

void GRTCoeff::initialize_E() {
  complex_d k = angfreq_ / c_;
  for (auto i = 0; i < nl_; ++i) {
    complex_d gamma = gamma_(i);
    complex_d nv = nv_(i);
    complex_d xi = pow(k, 2) + pow(nv, 2);

    // E
    e11_[i] << alpha_(i) * k, beta_(i) * nv, alpha_(i) * gamma, beta_(i) * k;
    e11_[i] /= angfreq_;

    auto &tmp11 = e11_[i];
    e12_[i] << tmp11(0, 0), tmp11(0, 1), -tmp11(1, 0), -tmp11(1, 1);

    e21_[i] << -2.0 * alpha_(i) * mu_(i) * k * gamma, -beta_(i) * mu_(i) * xi,
        -alpha_(i) * mu_(i) * xi, -2.0 * beta_(i) * mu_(i) * k * nv;
    e21_[i] /= angfreq_;

    auto &tmp21 = e21_[i];
    e22_[i] << -tmp21(0, 0), -tmp21(0, 1), tmp21(1, 0), tmp21(1, 1);
  }
}

void GRTCoeff::initialize_gamma() {
  for (int i = 0; i < nl_; ++i) {
    complex_d val =
        std::sqrt(pow(angfreq_ / c_, 2) - pow(angfreq_ / alpha_(i), 2));
    if (val.real() < 0) {
      val = -val;
    }
    gamma_(i) = val;
  }
}

void GRTCoeff::initialize_nv() {
  for (int i = 0; i < nl_; ++i) {
    complex_d val =
        std::sqrt(pow(angfreq_ / c_, 2) - pow(angfreq_ / beta_(i), 2));
    if (val.real() < 0) {
      val = -val;
    }
    nv_(i) = val;
  }
}

Matrix2cd GRTCoeff::get_Ad(const double z, const int ind_layer) const {
  Matrix2cd Ad;
  Ad << exp(-gamma_(ind_layer) * (z - z_(ind_layer))), 0., 0.,
      exp(-nv_(ind_layer) * (z - z_(ind_layer)));
  return Ad;
}

Matrix2cd GRTCoeff::get_Au(const double z, const int ind_layer) const {
  Matrix2cd Au;
  if (ind_layer == nl_ - 1) {
    Au << 0., 0., 0., 0.;
  } else {
    Au << exp(-gamma_(ind_layer) * (z_(ind_layer + 1) - z)), 0., 0.,
        exp(-nv_(ind_layer) * (z_(ind_layer + 1) - z));
  }
  return Au;
}

Matrix2cd GRTCoeff::get_Ad_der(const double z, const int ind_layer) const {
  Matrix2cd Ad_der;
  Ad_der << -gamma_(ind_layer) * exp(-gamma_(ind_layer) * (z - z_(ind_layer))),
      0., 0., -nv_(ind_layer) * exp(-nv_(ind_layer) * (z - z_(ind_layer)));
  return Ad_der;
}

Matrix2cd GRTCoeff::get_Au_der(const double z, const int ind_layer) const {
  if (ind_layer == nl_ - 1) {
    return Matrix2cd::Zero();
  }
  Matrix2cd Au_der;
  Au_der << gamma_(ind_layer) *
                exp(-gamma_(ind_layer) * (z_(ind_layer + 1) - z)),
      0., 0., nv_(ind_layer) * exp(-nv_(ind_layer) * (z_(ind_layer + 1) - z));
  return Au_der;
}

void GRTCoeff::compute_rtc() {
  for (auto i = 1; i < nl_ - 1; ++i) {
    auto &e11_0 = e11_[i - 1];
    auto &e12_0 = e12_[i - 1];
    auto &e21_0 = e21_[i - 1];
    auto &e22_0 = e22_[i - 1];
    auto &e11_1 = e11_[i];
    auto &e12_1 = e12_[i];
    auto &e21_1 = e21_[i];
    auto &e22_1 = e22_[i];

    Matrix2cd exp_d00 = get_Ad(z_(i), i - 1);
    Matrix2cd exp_u10 = get_Au(z_(i), i);

    Matrix4cd mat1;
    mat1 << e11_1(0, 0), e11_1(0, 1), -e12_0(0, 0), -e12_0(0, 1), //
        e11_1(1, 0), e11_1(1, 1), -e12_0(1, 0), -e12_0(1, 1),     //
        e21_1(0, 0), e21_1(0, 1), -e22_0(0, 0), -e22_0(0, 1),     //
        e21_1(1, 0), e21_1(1, 1), -e22_0(1, 0), -e22_0(1, 1);

    Matrix4cd mat2;
    mat2 << e11_0(0, 0), e11_0(0, 1), -e12_1(0, 0), -e12_1(0, 1), //
        e11_0(1, 0), e11_0(1, 1), -e12_1(1, 0), -e12_1(1, 1),     //
        e21_0(0, 0), e21_0(0, 1), -e22_1(0, 0), -e22_1(0, 1),     //
        e21_0(1, 0), e21_0(1, 1), -e22_1(1, 0), -e22_1(1, 1);

    Matrix4cd mat3;
    mat3 << exp_d00(0, 0), exp_d00(0, 1), 0, 0, //
        exp_d00(1, 0), exp_d00(1, 1), 0, 0,     //
        0, 0, exp_u10(0, 0), exp_u10(0, 1),     //
        0, 0, exp_u10(1, 0), exp_u10(1, 1);

    Matrix4cd result = mat1.inverse() * (mat2 * mat3);

    t_d_[i] << result(0, 0), result(0, 1), result(1, 0), result(1, 1);

    r_ud_[i] << result(0, 2), result(0, 3), result(1, 2), result(1, 3);

    r_du_[i] << result(2, 0), result(2, 1), result(3, 0), result(3, 1);

    t_u_[i] << result(2, 2), result(2, 3), result(3, 2), result(3, 3);
  }

  // the last interface
  auto &e11_N = e11_[nl_ - 1];
  auto &e21_N = e21_[nl_ - 1];
  auto &e11_N_1 = e11_[nl_ - 2];
  auto &e12_N_1 = e12_[nl_ - 2];
  auto &e21_N_1 = e21_[nl_ - 2];
  auto &e22_N_1 = e22_[nl_ - 2];

  Matrix4cd mat1;
  mat1 << e11_N(0, 0), e11_N(0, 1), -e12_N_1(0, 0), -e12_N_1(0, 1), //
      e11_N(1, 0), e11_N(1, 1), -e12_N_1(1, 0), -e12_N_1(1, 1),     //
      e21_N(0, 0), e21_N(0, 1), -e22_N_1(0, 0), -e22_N_1(0, 1),     //
      e21_N(1, 0), e21_N(1, 1), -e22_N_1(1, 0), -e22_N_1(1, 1);

  Matrix2cd exp_dN_1N_1 = get_Ad(z_(nl_ - 1), nl_ - 2);
  Matrix2cd mat2_1 = e11_N_1 * exp_dN_1N_1;
  Matrix2cd mat2_2 = e21_N_1 * exp_dN_1N_1;

  Mat42cd mat2;
  mat2 << mat2_1(0, 0), mat2_1(0, 1), mat2_1(1, 0), mat2_1(1, 1), //
      mat2_2(0, 0), mat2_2(0, 1), mat2_2(1, 0), mat2_2(1, 1);

  Mat42cd result = mat1.inverse() * mat2;
  t_d_[nl_ - 1] << result(0, 0), result(0, 1), result(1, 0), result(1, 1);
  r_du_[nl_ - 1] << result(2, 0), result(2, 1), result(3, 0), result(3, 1);
}

void GRTCoeff::compute_grtc() {
  for (auto i = nl_ - 1; i >= 1; --i) {
    Matrix2cd mat1 = matI_ - r_ud_[i] * gr_du_[i + 1];
    gt_d_[i] = mat1.inverse() * t_d_[i];
    gr_du_[i] = r_du_[i] + t_u_[i] * gr_du_[i + 1] * gt_d_[i];
  }
  gr_ud_[0] = -e21_[0].inverse() * e22_[0] * get_Au(z_(0), 0);
  for (auto i = 1; i < nl_; ++i) {
    Matrix2cd mat1 = matI_ - r_du_[i] * gr_ud_[i - 1];
    gt_u_[i] = mat1.inverse() * t_u_[i];
    gr_ud_[i] = r_ud_[i] + t_d_[i] * gr_ud_[i - 1] * gt_u_[i];
  }
}

void GRTCoeff::compute_CdCu() {
  Matrix2cd mat1 = matI_ - gr_ud_[0] * gr_du_[1];

  complex_d norm = sqrt(pow(mat1(0, 0), 2) + pow(mat1(0, 1), 2));
  Cd_[0] << mat1(0, 1) / norm, -mat1(0, 0) / norm;
  Cu_[0] = gr_du_[1] * Cd_[0];

  for (auto i = 1; i < nl_ - 1; ++i) {
    Cd_[i] = gt_d_[i] * Cd_[i - 1];
    Cu_[i] = gr_du_[i + 1] * Cd_[i];
  }
  Cd_[nl_ - 1] = gt_d_[nl_ - 1] * Cd_[nl_ - 2];
}

IntegralLayer::IntegralLayer(const Ref<const ArrayXXd> model, const double freq,
                             const double c)
    : grtc_(std::make_unique<GRTCoeff>(model, freq, c)), nl_(model.rows()),
      k_(2.0 * PI * freq / c), pvel_(c), z_(model.col(1)), alpha_(model.col(4)),
      beta_(model.col(3)), rho_(model.col(2)), mu_(rho_ * beta_ * beta_),
      lamb_(rho_ * alpha_ * alpha_ - 2.0 * mu_), gamma_(grtc_->gamma_),
      nv_(grtc_->nv_), thickness_(nl_ - 1), Cd_(grtc_->Cd_), Cu_(grtc_->Cu_),
      matE_(nl_, MatrixXcd::Zero(2, 4)), matP_u_u_(nl_, Matrix4cd::Zero()),
      matP_uc_u_(nl_, Matrix4cd::Zero()), matP_uc_uc_(nl_, Matrix4cd::Zero()),
      matP_du_du_(nl_, Matrix4cd::Zero()), matP_duc_du_(nl_, Matrix4cd::Zero()),
      matP_duc_duc_(nl_, Matrix4cd::Zero()), matP_u_du_(nl_, Matrix4cd::Zero()),
      matP_uc_du_(nl_, Matrix4cd::Zero()), matP_u_duc_(nl_, Matrix4cd::Zero()),
      matP_uc_duc_(nl_, Matrix4cd::Zero()),
      sigma_x_sigma_top_(nl_, Matrix4cd::Zero()),
      sigmac_x_sigma_top_(nl_, Matrix4cd::Zero()),
      sigma_x_sigmac_top_(nl_, Matrix4cd::Zero()),
      sigmac_x_sigmac_top_(nl_, Matrix4cd::Zero()),
      sigma_x_sigma_bottom_(nl_, Matrix4cd::Zero()),
      sigmac_x_sigma_bottom_(nl_, Matrix4cd::Zero()),
      sigma_x_sigmac_bottom_(nl_, Matrix4cd::Zero()),
      sigmac_x_sigmac_bottom_(nl_, Matrix4cd::Zero()),
      int_us2_(ArrayXd::Zero(nl_)), int_ur2_(ArrayXd::Zero(nl_)),
      int_dus2_(ArrayXd::Zero(nl_)), int_dur2_(ArrayXd::Zero(nl_)),
      int_urdus_(ArrayXd::Zero(nl_)), int_usdur_(ArrayXd::Zero(nl_)) {
  for (int i = 0; i < nl_ - 1; ++i) {
    thickness_(i) = z_(i + 1) - z_(i);
  }
  for (int i = 0; i < nl_; ++i) {
    matE_[i] << grtc_->e11_[i], grtc_->e12_[i];
  }

  initialize_P();
  initialize_sigma();

  integrate_us2();
  integrate_dus2();
  integrate_ur2();
  integrate_dur2();
  integrate_usdur();
  integrate_urdus();
}

void IntegralLayer::initialize_P() {
  for (int i = 0; i < nl_; ++i) {
    const complex_d ga0 = gamma_(i);
    const complex_d nv0 = nv_(i);
    const complex_d ga0_2 = ga0 * ga0;
    const complex_d nv0_2 = nv0 * nv0;
    const complex_d ga1 = conj(ga0);
    const complex_d nv1 = conj(nv0);
    const complex_d ga1_2 = ga1 * ga1;
    const complex_d nv1_2 = nv1 * nv1;
    matP_u_u_[i] << 1. / (-ga0 - ga0), 1. / (-ga0 - nv0), 1., 1. / (-ga0 + nv0),
        1. / (-nv0 - ga0), 1. / (-nv0 - nv0), 1. / (-nv0 + ga0), 1., 1.,
        1. / (ga0 - nv0), 1. / (ga0 + ga0), 1. / (ga0 + nv0), 1. / (nv0 - ga0),
        1., 1. / (nv0 + ga0), 1. / (nv0 + nv0);
    matP_uc_u_[i] << 1. / (-ga1 - ga0), 1. / (-ga1 - nv0), 1. / (-ga1 + ga0),
        1. / (-ga1 + nv0), 1. / (-nv1 - ga0), 1. / (-nv1 - nv0),
        1. / (-nv1 + ga0), 1. / (-nv1 + nv0), 1. / (ga1 - ga0),
        1. / (ga1 - nv0), 1. / (ga1 + ga0), 1. / (ga1 + nv0), 1. / (nv1 - ga0),
        1. / (nv1 - nv0), 1. / (nv1 + ga0), 1. / (nv1 + nv0);
    matP_uc_uc_[i] << 1. / (-ga1 - ga1), 1. / (-ga1 - nv1), 1.,
        1. / (-ga1 + nv1), 1. / (-nv1 - ga1), 1. / (-nv1 - nv1),
        1. / (-nv1 + ga1), 1., 1., 1. / (ga1 - nv1), 1. / (ga1 + ga1),
        1. / (ga1 + nv1), 1. / (nv1 - ga1), 1., 1. / (nv1 + ga1),
        1. / (nv1 + nv1);
    matP_du_du_[i] << ga0_2 / (-ga0 - ga0), ga0 * nv0 / (-ga0 - nv0), -ga0_2,
        -ga0 * nv0 / (-ga0 + nv0), nv0 * ga0 / (-nv0 - ga0),
        nv0_2 / (-nv0 - nv0), -nv0 * ga0 / (-nv0 + ga0), -nv0_2, -ga0_2,
        -ga0 * nv0 / (ga0 - nv0), ga0_2 / (ga0 + ga0), ga0 * nv0 / (ga0 + nv0),
        -ga0 * nv0 / (nv0 - ga0), -nv0_2, ga0 * nv0 / (nv0 + ga0),
        nv0_2 / (nv0 + nv0);
    matP_duc_du_[i] << ga1 * ga0 / (-ga1 - ga0), ga1 * nv0 / (-ga1 - nv0),
        -ga1 * ga0 / (-ga1 + ga0), -ga1 * nv0 / (-ga1 + nv0),
        nv1 * ga0 / (-nv1 - ga0), nv1 * nv0 / (-nv1 - nv0),
        -nv1 * ga0 / (-nv1 + ga0), -nv1 * nv0 / (-nv1 + nv0),
        -ga1 * ga0 / (ga1 - ga0), -ga1 * nv0 / (ga1 - nv0),
        ga1 * ga0 / (ga1 + ga0), ga1 * nv0 / (ga1 + nv0),
        -nv1 * ga0 / (nv1 - ga0), -nv1 * nv0 / (nv1 - nv0),
        nv1 * ga0 / (nv1 + ga0), nv1 * nv0 / (nv1 + nv0);
    matP_duc_duc_[i] << ga1_2 / (-ga1 - ga1), ga1 * nv1 / (-ga1 - nv1), -ga1_2,
        -ga1 * nv1 / (-ga1 + nv1), nv1 * ga1 / (-nv1 - ga1),
        nv1_2 / (-nv1 - nv1), -nv1 * ga1 / (-nv1 + ga1), -nv1_2, -ga1_2,
        -ga1 * nv1 / (ga1 - nv1), ga1_2 / (ga1 + ga1), ga1 * nv1 / (ga1 + nv1),
        -ga1 * nv1 / (nv1 - ga1), -nv1_2, ga1 * nv1 / (nv1 + ga1),
        nv1_2 / (nv1 + nv1);
    matP_u_du_[i] << -ga0 / (-ga0 - ga0), -nv0 / (-ga0 - nv0), ga0,
        nv0 / (-ga0 + nv0), -ga0 / (-nv0 - ga0), -nv0 / (-nv0 - nv0),
        ga0 / (-nv0 + ga0), nv0, -ga0, -nv0 / (ga0 - nv0), ga0 / (ga0 + ga0),
        nv0 / (ga0 + nv0), -ga0 / (nv0 - ga0), -nv0, ga0 / (nv0 + ga0),
        nv0 / (nv0 + nv0);
    matP_uc_du_[i] << -ga0 / (-ga1 - ga0), -nv0 / (-ga1 - nv0),
        ga0 / (-ga1 + ga0), nv0 / (-ga1 + nv0), -ga0 / (-nv1 - ga0),
        -nv0 / (-nv1 - nv0), ga0 / (-nv1 + ga0), nv0 / (-nv1 + nv0),
        -ga0 / (ga1 - ga0), -nv0 / (ga1 - nv0), ga0 / (ga1 + ga0),
        nv0 / (ga1 + nv0), -ga0 / (nv1 - ga0), -nv0 / (nv1 - nv0),
        ga0 / (nv1 + ga0), nv0 / (nv1 + nv0);
    matP_u_duc_[i] << -ga1 / (-ga0 - ga1), -nv1 / (-ga0 - nv1),
        ga1 / (-ga0 + ga1), nv1 / (-ga0 + nv1), -ga1 / (-nv0 - ga1),
        -nv1 / (-nv0 - nv1), ga1 / (-nv0 + ga1), nv1 / (-nv0 + nv1),
        -ga1 / (ga0 - ga1), -nv1 / (ga0 - nv1), ga1 / (ga0 + ga1),
        nv1 / (ga0 + nv1), -ga1 / (nv0 - ga1), -nv1 / (nv0 - nv1),
        ga1 / (nv0 + ga1), nv1 / (nv0 + nv1);
    matP_uc_duc_[i] << -ga1 / (-ga1 - ga1), -nv1 / (-ga1 - nv1), ga1,
        nv1 / (-ga1 + nv1), -ga1 / (-nv1 - ga1), -nv1 / (-nv1 - nv1),
        ga1 / (-nv1 + ga1), nv1, -ga1, -nv1 / (ga1 - nv1), ga1 / (ga1 + ga1),
        nv1 / (ga1 + nv1), -ga1 / (nv1 - ga1), -nv1, ga1 / (nv1 + ga1),
        nv1 / (nv1 + nv1);
    if (pvel_ > alpha_(i)) {
      matP_uc_u_[i](0, 0) = 1.;
      matP_uc_u_[i](2, 2) = 1.;
      matP_duc_du_[i](0, 0) = ga1 * ga0;
      matP_duc_du_[i](2, 2) = ga1 * ga0;
      matP_uc_du_[i](0, 0) = -ga0;
      matP_uc_du_[i](2, 2) = ga0;
      matP_u_duc_[i](0, 0) = -ga1;
      matP_u_duc_[i](2, 2) = ga1;
    } else {
      matP_uc_u_[i](0, 2) = 1.;
      matP_uc_u_[i](2, 0) = 1.;
      matP_duc_du_[i](0, 2) = -ga1 * ga0;
      matP_duc_du_[i](2, 0) = -ga1 * ga0;
      matP_uc_du_[i](0, 2) = ga0;
      matP_uc_du_[i](2, 0) = -ga0;
      matP_u_duc_[i](0, 2) = ga1;
      matP_u_duc_[i](2, 0) = -ga1;
    }
    if (pvel_ > beta_(i)) {
      matP_uc_u_[i](1, 1) = 1.;
      matP_uc_u_[i](3, 3) = 1.;
      matP_duc_du_[i](1, 1) = nv1 * nv0;
      matP_duc_du_[i](3, 3) = nv1 * nv0;
      matP_uc_du_[i](1, 1) = -nv0;
      matP_uc_du_[i](3, 3) = nv0;
      matP_u_duc_[i](1, 1) = -nv1;
      matP_u_duc_[i](3, 3) = nv1;
    } else {
      matP_uc_u_[i](1, 3) = 1.;
      matP_uc_u_[i](3, 1) = 1.;
      matP_duc_du_[i](1, 3) = -nv1 * nv0;
      matP_duc_du_[i](3, 1) = -nv1 * nv0;
      matP_uc_du_[i](1, 3) = nv0;
      matP_uc_du_[i](3, 1) = -nv0;
      matP_u_duc_[i](1, 3) = nv1;
      matP_u_duc_[i](3, 1) = -nv1;
    }
  }
}

void IntegralLayer::initialize_sigma() {
  MatrixXcd sigma_bottom = MatrixXcd::Zero(4, nl_);
  MatrixXcd sigma_top = MatrixXcd::Zero(4, nl_);
  for (auto i = 0; i < nl_ - 1; ++i) {
    sigma_bottom.col(i) << Cd_[i](0), Cd_[i](1),
        Cu_[i](0) * exp(-gamma_(i) * thickness_(i)),
        Cu_[i](1) * exp(-nv_(i) * thickness_(i));
    sigma_top.col(i) << Cd_[i](0) * exp(-gamma_(i) * thickness_(i)),
        Cd_[i](1) * exp(-nv_(i) * thickness_(i)), Cu_[i](0), Cu_[i](1);
  }
  sigma_bottom.col(nl_ - 1) << Cd_[nl_ - 1](0), Cd_[nl_ - 1](1), 0, 0;
  sigma_top.col(nl_ - 1).fill(0);

  for (auto id_layer = 0; id_layer < nl_; ++id_layer) {
    auto cd = Cd_[id_layer];
    auto cu = Cu_[id_layer];
    for (auto i = 0; i < 4; ++i) {
      for (auto j = 0; j < 4; ++j) {
        sigma_x_sigma_top_[id_layer](i, j) =
            sigma_top(i, id_layer) * sigma_top(j, id_layer);
        sigmac_x_sigma_top_[id_layer](i, j) =
            conj(sigma_top(i, id_layer)) * sigma_top(j, id_layer);
        sigma_x_sigmac_top_[id_layer](i, j) =
            sigma_top(i, id_layer) * conj(sigma_top(j, id_layer));
        sigmac_x_sigmac_top_[id_layer](i, j) =
            conj(sigma_top(i, id_layer)) * conj(sigma_top(j, id_layer));

        sigma_x_sigma_bottom_[id_layer](i, j) =
            sigma_bottom(i, id_layer) * sigma_bottom(j, id_layer);
        sigmac_x_sigma_bottom_[id_layer](i, j) =
            conj(sigma_bottom(i, id_layer)) * sigma_bottom(j, id_layer);
        sigma_x_sigmac_bottom_[id_layer](i, j) =
            sigma_bottom(i, id_layer) * conj(sigma_bottom(j, id_layer));
        sigmac_x_sigmac_bottom_[id_layer](i, j) =
            conj(sigma_bottom(i, id_layer)) * conj(sigma_bottom(j, id_layer));
      }
    }
    if (id_layer != nl_ - 1) {
      sigma_x_sigma_top_[id_layer](0, 2) *= thickness_(id_layer);
      sigma_x_sigma_top_[id_layer](2, 0) *= thickness_(id_layer);
      sigma_x_sigma_top_[id_layer](1, 3) *= thickness_(id_layer);
      sigma_x_sigma_top_[id_layer](3, 1) *= thickness_(id_layer);
      sigmac_x_sigmac_top_[id_layer](0, 2) *= thickness_(id_layer);
      sigmac_x_sigmac_top_[id_layer](2, 0) *= thickness_(id_layer);
      sigmac_x_sigmac_top_[id_layer](1, 3) *= thickness_(id_layer);
      sigmac_x_sigmac_top_[id_layer](3, 1) *= thickness_(id_layer);
      if (pvel_ > alpha_(id_layer)) {
        sigmac_x_sigma_top_[id_layer](0, 0) =
            conj(cd(0)) * cd(0) * thickness_(id_layer);
        sigmac_x_sigma_top_[id_layer](2, 2) =
            conj(cu(0)) * cu(0) * thickness_(id_layer);
        sigma_x_sigmac_top_[id_layer](0, 0) =
            cd(0) * conj(cd(0)) * thickness_(id_layer);
        sigma_x_sigmac_top_[id_layer](2, 2) =
            cu(0) * conj(cu(0)) * thickness_(id_layer);
      } else {
        sigmac_x_sigma_top_[id_layer](0, 2) *= thickness_(id_layer);
        sigmac_x_sigma_top_[id_layer](2, 0) *= thickness_(id_layer);
        sigma_x_sigmac_top_[id_layer](0, 2) *= thickness_(id_layer);
        sigma_x_sigmac_top_[id_layer](2, 0) *= thickness_(id_layer);
      }
      if (pvel_ > beta_(id_layer)) {
        sigmac_x_sigma_top_[id_layer](1, 1) =
            conj(cd(1)) * cd(1) * thickness_(id_layer);
        sigmac_x_sigma_top_[id_layer](3, 3) =
            conj(cu(1)) * cu(1) * thickness_(id_layer);
        sigma_x_sigmac_top_[id_layer](1, 1) =
            cd(1) * conj(cd(1)) * thickness_(id_layer);
        sigma_x_sigmac_top_[id_layer](3, 3) =
            cu(1) * conj(cu(1)) * thickness_(id_layer);
      } else {
        sigmac_x_sigma_top_[id_layer](1, 3) *= thickness_(id_layer);
        sigmac_x_sigma_top_[id_layer](3, 1) *= thickness_(id_layer);
        sigma_x_sigmac_top_[id_layer](1, 3) *= thickness_(id_layer);
        sigma_x_sigmac_top_[id_layer](3, 1) *= thickness_(id_layer);
      }
      sigma_x_sigma_bottom_[id_layer](0, 2) = 0.;
      sigma_x_sigma_bottom_[id_layer](2, 0) = 0.;
      sigma_x_sigma_bottom_[id_layer](1, 3) = 0.;
      sigma_x_sigma_bottom_[id_layer](3, 1) = 0.;
      sigmac_x_sigmac_bottom_[id_layer](0, 2) = 0.;
      sigmac_x_sigmac_bottom_[id_layer](2, 0) = 0.;
      sigmac_x_sigmac_bottom_[id_layer](1, 3) = 0.;
      sigmac_x_sigmac_bottom_[id_layer](3, 1) = 0.;
      if (pvel_ > alpha_(id_layer)) {
        sigmac_x_sigma_bottom_[id_layer](0, 0) = 0.;
        sigmac_x_sigma_bottom_[id_layer](2, 2) = 0.;
        sigma_x_sigmac_bottom_[id_layer](0, 0) = 0.;
        sigma_x_sigmac_bottom_[id_layer](2, 2) = 0.;
      } else {
        sigmac_x_sigma_bottom_[id_layer](0, 2) = 0.;
        sigmac_x_sigma_bottom_[id_layer](2, 0) = 0.;
        sigma_x_sigmac_bottom_[id_layer](0, 2) = 0.;
        sigma_x_sigmac_bottom_[id_layer](2, 0) = 0.;
      }
      if (pvel_ > beta_(id_layer)) {
        sigmac_x_sigma_bottom_[id_layer](1, 1) = 0.;
        sigmac_x_sigma_bottom_[id_layer](3, 3) = 0.;
        sigma_x_sigmac_bottom_[id_layer](1, 1) = 0.;
        sigma_x_sigmac_bottom_[id_layer](3, 3) = 0.;
      } else {
        sigmac_x_sigma_bottom_[id_layer](1, 3) = 0.;
        sigmac_x_sigma_bottom_[id_layer](3, 1) = 0.;
        sigma_x_sigmac_bottom_[id_layer](1, 3) = 0.;
        sigma_x_sigmac_bottom_[id_layer](3, 1) = 0.;
      }
    }
  }
}

double IntegralLayer::intker_us2_top(int id_layer) {
  complex_d result = 0;

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      result += matP_u_u_[id_layer](i, j) * matE_[id_layer](0, i) *
                    matE_[id_layer](0, j) * sigma_x_sigma_top_[id_layer](i, j) +
                2. * matP_uc_u_[id_layer](i, j) * conj(matE_[id_layer](0, i)) *
                    matE_[id_layer](0, j) *
                    sigmac_x_sigma_top_[id_layer](i, j) +
                matP_uc_uc_[id_layer](i, j) * conj(matE_[id_layer](0, i)) *
                    conj(matE_[id_layer](0, j)) *
                    sigmac_x_sigmac_top_[id_layer](i, j);
    }
  }
  return std::real(result) / 4;
}

double IntegralLayer::intker_us2_bottom(int id_layer) {
  complex_d result = 0;

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      result +=
          matP_u_u_[id_layer](i, j) * matE_[id_layer](0, i) *
              matE_[id_layer](0, j) * sigma_x_sigma_bottom_[id_layer](i, j) +
          2. * matP_uc_u_[id_layer](i, j) * conj(matE_[id_layer](0, i)) *
              matE_[id_layer](0, j) * sigmac_x_sigma_bottom_[id_layer](i, j) +
          matP_uc_uc_[id_layer](i, j) * conj(matE_[id_layer](0, i)) *
              conj(matE_[id_layer](0, j)) *
              sigmac_x_sigmac_bottom_[id_layer](i, j);
    }
  }
  return std::real(result) / 4;
}

double IntegralLayer::intker_ur2_top(int id_layer) {
  complex_d result = 0;

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      result += matP_u_u_[id_layer](i, j) * matE_[id_layer](1, i) *
                    matE_[id_layer](1, j) * sigma_x_sigma_top_[id_layer](i, j) +
                2. * matP_uc_u_[id_layer](i, j) * conj(matE_[id_layer](1, i)) *
                    matE_[id_layer](1, j) *
                    sigmac_x_sigma_top_[id_layer](i, j) +
                matP_uc_uc_[id_layer](i, j) * conj(matE_[id_layer](1, i)) *
                    conj(matE_[id_layer](1, j)) *
                    sigmac_x_sigmac_top_[id_layer](i, j);
    }
  }
  return std::real(result) / 4;
}

double IntegralLayer::intker_ur2_bottom(int id_layer) {
  complex_d result = 0;

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      result +=
          matP_u_u_[id_layer](i, j) * matE_[id_layer](1, i) *
              matE_[id_layer](1, j) * sigma_x_sigma_bottom_[id_layer](i, j) +
          2. * matP_uc_u_[id_layer](i, j) * conj(matE_[id_layer](1, i)) *
              matE_[id_layer](1, j) * sigmac_x_sigma_bottom_[id_layer](i, j) +
          matP_uc_uc_[id_layer](i, j) * conj(matE_[id_layer](1, i)) *
              conj(matE_[id_layer](1, j)) *
              sigmac_x_sigmac_bottom_[id_layer](i, j);
    }
  }
  return std::real(result) / 4;
}

double IntegralLayer::intker_dus2_top(int id_layer) {
  complex_d result = 0;

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      result += matP_du_du_[id_layer](i, j) * matE_[id_layer](0, i) *
                    matE_[id_layer](0, j) * sigma_x_sigma_top_[id_layer](i, j) +
                2. * matP_duc_du_[id_layer](i, j) *
                    conj(matE_[id_layer](0, i)) * matE_[id_layer](0, j) *
                    sigmac_x_sigma_top_[id_layer](i, j) +
                matP_duc_duc_[id_layer](i, j) * conj(matE_[id_layer](0, i)) *
                    conj(matE_[id_layer](0, j)) *
                    sigmac_x_sigmac_top_[id_layer](i, j);
    }
  }
  return std::real(result) / 4.;
}

double IntegralLayer::intker_dus2_bottom(int id_layer) {
  complex_d result = 0;

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      result +=
          matP_du_du_[id_layer](i, j) * matE_[id_layer](0, i) *
              matE_[id_layer](0, j) * sigma_x_sigma_bottom_[id_layer](i, j) +
          2. * matP_duc_du_[id_layer](i, j) * conj(matE_[id_layer](0, i)) *
              matE_[id_layer](0, j) * sigmac_x_sigma_bottom_[id_layer](i, j) +
          matP_duc_duc_[id_layer](i, j) * conj(matE_[id_layer](0, i)) *
              conj(matE_[id_layer](0, j)) *
              sigmac_x_sigmac_bottom_[id_layer](i, j);
    }
  }
  return std::real(result) / 4.;
}

double IntegralLayer::intker_dur2_top(int id_layer) {
  complex_d result = 0;

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      result += matP_du_du_[id_layer](i, j) * matE_[id_layer](1, i) *
                    matE_[id_layer](1, j) * sigma_x_sigma_top_[id_layer](i, j) +
                2. * matP_duc_du_[id_layer](i, j) *
                    conj(matE_[id_layer](1, i)) * matE_[id_layer](1, j) *
                    sigmac_x_sigma_top_[id_layer](i, j) +
                matP_duc_duc_[id_layer](i, j) * conj(matE_[id_layer](1, i)) *
                    conj(matE_[id_layer](1, j)) *
                    sigmac_x_sigmac_top_[id_layer](i, j);
    }
  }
  return std::real(result) / 4;
}

double IntegralLayer::intker_dur2_bottom(int id_layer) {
  complex_d result = 0;

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      result +=
          matP_du_du_[id_layer](i, j) * matE_[id_layer](1, i) *
              matE_[id_layer](1, j) * sigma_x_sigma_bottom_[id_layer](i, j) +
          2. * matP_duc_du_[id_layer](i, j) * conj(matE_[id_layer](1, i)) *
              matE_[id_layer](1, j) * sigmac_x_sigma_bottom_[id_layer](i, j) +
          matP_duc_duc_[id_layer](i, j) * conj(matE_[id_layer](1, i)) *
              conj(matE_[id_layer](1, j)) *
              sigmac_x_sigmac_bottom_[id_layer](i, j);
    }
  }
  return std::real(result) / 4;
}

double IntegralLayer::intker_urdus_top(int id_layer) {
  complex_d result = 0;

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      result += matP_u_du_[id_layer](i, j) * matE_[id_layer](1, i) *
                    matE_[id_layer](0, j) * sigma_x_sigma_top_[id_layer](i, j) +
                matP_u_duc_[id_layer](i, j) * matE_[id_layer](1, i) *
                    conj(matE_[id_layer](0, j)) *
                    sigma_x_sigmac_top_[id_layer](i, j) +
                matP_uc_du_[id_layer](i, j) * conj(matE_[id_layer](1, i)) *
                    matE_[id_layer](0, j) *
                    sigmac_x_sigma_top_[id_layer](i, j) +
                matP_uc_duc_[id_layer](i, j) * conj(matE_[id_layer](1, i)) *
                    conj(matE_[id_layer](0, j)) *
                    sigmac_x_sigmac_top_[id_layer](i, j);
    }
  }
  return std::real(result) / 4.;
}

double IntegralLayer::intker_urdus_bottom(int id_layer) {
  complex_d result = 0;

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      result +=
          matP_u_du_[id_layer](i, j) * matE_[id_layer](1, i) *
              matE_[id_layer](0, j) * sigma_x_sigma_bottom_[id_layer](i, j) +
          matP_u_duc_[id_layer](i, j) * matE_[id_layer](1, i) *
              conj(matE_[id_layer](0, j)) *
              sigma_x_sigmac_bottom_[id_layer](i, j) +
          matP_uc_du_[id_layer](i, j) * conj(matE_[id_layer](1, i)) *
              matE_[id_layer](0, j) * sigmac_x_sigma_bottom_[id_layer](i, j) +
          matP_uc_duc_[id_layer](i, j) * conj(matE_[id_layer](1, i)) *
              conj(matE_[id_layer](0, j)) *
              sigmac_x_sigmac_bottom_[id_layer](i, j);
    }
  }
  return std::real(result) / 4.;
}

double IntegralLayer::intker_usdur_top(int id_layer) {
  complex_d result = 0;

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      result += matP_u_du_[id_layer](i, j) * matE_[id_layer](0, i) *
                    matE_[id_layer](1, j) * sigma_x_sigma_top_[id_layer](i, j) +
                matP_u_duc_[id_layer](i, j) * matE_[id_layer](0, i) *
                    conj(matE_[id_layer](1, j)) *
                    sigma_x_sigmac_top_[id_layer](i, j) +
                matP_uc_du_[id_layer](i, j) * conj(matE_[id_layer](0, i)) *
                    matE_[id_layer](1, j) *
                    sigmac_x_sigma_top_[id_layer](i, j) +
                matP_uc_duc_[id_layer](i, j) * conj(matE_[id_layer](0, i)) *
                    conj(matE_[id_layer](1, j)) *
                    sigmac_x_sigmac_top_[id_layer](i, j);
    }
  }
  return std::real(result) / 4.;
}

double IntegralLayer::intker_usdur_bottom(int id_layer) {
  complex_d result = 0;

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      result +=
          matP_u_du_[id_layer](i, j) * matE_[id_layer](0, i) *
              matE_[id_layer](1, j) * sigma_x_sigma_bottom_[id_layer](i, j) +
          matP_u_duc_[id_layer](i, j) * matE_[id_layer](0, i) *
              conj(matE_[id_layer](1, j)) *
              sigma_x_sigmac_bottom_[id_layer](i, j) +
          matP_uc_du_[id_layer](i, j) * conj(matE_[id_layer](0, i)) *
              matE_[id_layer](1, j) * sigmac_x_sigma_bottom_[id_layer](i, j) +
          matP_uc_duc_[id_layer](i, j) * conj(matE_[id_layer](0, i)) *
              conj(matE_[id_layer](1, j)) *
              sigmac_x_sigmac_bottom_[id_layer](i, j);
    }
  }
  return std::real(result) / 4.;
}

void IntegralLayer::integrate_us2() {
  for (int id_layer = 0; id_layer < nl_; ++id_layer) {
    int_us2_(id_layer) = intker_us2_top(id_layer) - intker_us2_bottom(id_layer);
  }
}

void IntegralLayer::integrate_ur2() {
  for (int id_layer = 0; id_layer < nl_; ++id_layer) {
    int_ur2_(id_layer) = intker_ur2_top(id_layer) - intker_ur2_bottom(id_layer);
  }
}

void IntegralLayer::integrate_dus2() {
  for (int id_layer = 0; id_layer < nl_; ++id_layer) {
    int_dus2_(id_layer) =
        intker_dus2_top(id_layer) - intker_dus2_bottom(id_layer);
  }
}

void IntegralLayer::integrate_dur2() {
  for (int id_layer = 0; id_layer < nl_; ++id_layer) {
    int_dur2_(id_layer) =
        intker_dur2_top(id_layer) - intker_dur2_bottom(id_layer);
  }
}

void IntegralLayer::integrate_usdur() {
  for (int id_layer = 0; id_layer < nl_; ++id_layer) {
    int_usdur_(id_layer) =
        intker_usdur_top(id_layer) - intker_usdur_bottom(id_layer);
  }
}

void IntegralLayer::integrate_urdus() {
  for (int id_layer = 0; id_layer < nl_; ++id_layer) {
    int_urdus_(id_layer) =
        intker_urdus_top(id_layer) - intker_urdus_bottom(id_layer);
  }
}

double IntegralLayer::compute_I1() {
  ArrayXd ker = rho_ * (int_us2_ + int_ur2_);
  double i1 = 0.5 * ker.sum();
  return i1;
}

double IntegralLayer::compute_I2() {
  ArrayXd ker = (lamb_ + 2.0 * mu_) * int_us2_ + mu_ * int_ur2_;
  double i2 = 0.5 * ker.sum();
  return i2;
}

double IntegralLayer::compute_I3() {
  ArrayXd ker = lamb_ * int_usdur_ - mu_ * int_urdus_;
  double i3 = 0.5 * ker.sum();
  return i3;
}

ArrayXd IntegralLayer::compute_kvs() {
  double k2 = std::pow(k_, 2);
  ArrayXd kvs(nl_);
  for (int i = 0; i < nl_; ++i) {
    kvs(i) = 0.5 * rho_(i) * beta_(i) *
             (int_ur2_(i) + 1.0 / k2 * int_dus2_(i) - 2.0 / k_ * int_urdus_(i) -
              4.0 / k_ * int_usdur_(i));
  }
  return kvs;
}

GradientPSV::~GradientPSV() = default;

ArrayXd GradientPSV::compute(const double freq, const double c) const {
  double k = 2.0 * PI * freq / c;
  IntegralLayer intl(model_, freq, c);
  double I2 = intl.compute_I2();
  double I3 = intl.compute_I3();
  ArrayXd kvs = intl.compute_kvs();
  kvs *= c / (I2 + I3 / (2.0*k));
  return kvs;
}

} // namespace grad_psv

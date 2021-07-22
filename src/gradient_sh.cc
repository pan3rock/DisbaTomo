#include "gradient_sh.hpp"

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

const double PI = 3.14159265358979323846;

namespace grad_sh {

GRTCoeff::GRTCoeff(const Ref<const ArrayXXd> model, const double freq,
                   const double c)
    : z_(model.col(1)), rho_(model.col(2)), beta_(model.col(3)),
      alpha_(model.col(4)), mu_(rho_ * beta_.pow(2)), nl_(model.rows()),
      angfreq_(2.0 * PI * freq), c_(c), nv_(VectorXcd::Zero(nl_)),
      e_(nl_, Matrix2cd::Zero()), t_d_(VectorXcd::Zero(nl_)),
      r_ud_(VectorXcd::Zero(nl_)), r_du_(VectorXcd::Zero(nl_)),
      t_u_(VectorXcd::Zero(nl_)), gt_d_(VectorXcd::Zero(nl_)),
      gr_ud_(VectorXcd::Zero(nl_)), gr_du_(VectorXcd::Zero(nl_ + 1)),
      gt_u_(VectorXcd::Zero(nl_)), cd_(VectorXcd::Zero(nl_)),
      cu_(VectorXcd::Zero(nl_)) {

  initialize_nv();
  initialize_E();

  compute_rtc();
  compute_grtc();
  compute_CdCu();
}

void GRTCoeff::initialize_E() {
  for (int i = 0; i < nl_; ++i) {
    complex_d e21 = -mu_(i) * nv_(i);
    e_[i] << 1.0, 1.0, e21, -e21;
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

complex_d GRTCoeff::get_Ad(const double z, const int ind_layer) const {
  return exp(-nv_(ind_layer) * (z - z_(ind_layer)));
}

complex_d GRTCoeff::get_Au(const double z, const int ind_layer) const {
  return exp(-nv_(ind_layer) * (z_(ind_layer + 1) - z));
}

complex_d GRTCoeff::get_Ad_der(const double z, const int ind_layer) const {
  return -nv_(ind_layer) * exp(-nv_(ind_layer) * (z - z_(ind_layer)));
}

complex_d GRTCoeff::get_Au_der(const double z, const int ind_layer) const {
  if (ind_layer == nl_ - 1) {
    return 0.0;
  }

  return nv_(ind_layer) * exp(-nv_(ind_layer) * (z_(ind_layer + 1) - z));
}

void GRTCoeff::compute_rtc() {
  for (int i = 1; i < nl_ - 1; ++i) {
    auto &e0 = e_[i - 1];
    auto &e1 = e_[i];

    Matrix2cd mat1;
    mat1 << e1(0, 0), -e0(0, 1), e1(1, 0), -e0(1, 1);

    // complex_d ad = exp(-nv_(i - 1) * (z_(i) - z_(i - 1)));
    // complex_d au = exp(-nv_(i) * (z_(i + 1) - z_(i)));
    complex_d ad = get_Ad(z_(i), i - 1);
    complex_d au = get_Au(z_(i), i);
    Matrix2cd mat2;
    mat2 << e0(0, 0) * ad, -e1(0, 1) * au, e0(1, 0) * ad, -e1(1, 1) * au;

    Matrix2cd result = mat1.inverse() * mat2;
    t_d_(i) = result(0, 0);
    r_ud_(i) = result(0, 1);
    r_du_(i) = result(1, 0);
    t_u_(i) = result(1, 1);
  }

  int i = nl_ - 1;
  auto &e0 = e_[i - 1];
  auto &e1 = e_[i];
  Matrix2cd mat1;
  mat1 << e1(0, 0), -e0(0, 1), e1(1, 0), -e0(1, 1);
  // complex_d ad = exp(-nv_(i - 1) * (z_(i) - z_(i - 1)));
  complex_d ad = get_Ad(z_(nl_ - 1), nl_ - 2);
  Matrix<complex_d, 2, 1> mat2;
  mat2 << e0(0, 0) * ad, e0(1, 0) * ad;

  Matrix<complex_d, 2, 1> result = mat1.inverse() * mat2;
  t_d_(i) = result(0, 0);
  r_du_(i) = result(1, 0);
}

void GRTCoeff::compute_grtc() {
  for (int i = nl_ - 1; i >= 1; --i) {
    gt_d_(i) = 1.0 / (1.0 - r_ud_(i) * gr_du_(i + 1)) * t_d_(i);
    gr_du_(i) = r_du_(i) + t_u_(i) * gr_du_(i + 1) * gt_d_(i);
  }
  auto &e0 = e_[0];
  // complex_d au = exp(-nv_(0) * (z_(1) - z_(0)));
  complex_d au = get_Au(z_(0), 0);
  gr_ud_(0) = -1.0 / e0(1, 0) * e0(1, 1) * au;
  for (int i = 1; i < nl_; ++i) {
    gt_u_(i) = 1.0 / (1.0 - r_du_(i) * gr_ud_(i - 1)) * t_u_(i);
    gr_ud_(i) = r_ud_(i) + t_d_(i) * gr_ud_(i - 1) * gt_u_(i);
  }
}

void GRTCoeff::compute_CdCu() {
  cd_(0) = 1.0;
  cu_(0) = gr_du_(1) * cd_(0);
  for (int i = 1; i < nl_ - 1; ++i) {
    cd_(i) = gt_d_(i) * cd_(i - 1);
    cu_(i) = gr_du_(i + 1) * cd_(i);
  }
  cd_(nl_ - 1) = gt_d_(nl_ - 1) * cd_(nl_ - 2);
}

IntegralLayer::IntegralLayer(const Ref<const ArrayXXd> model, const double freq,
                             const double c)
    : grtc_(std::make_unique<GRTCoeff>(model, freq, c)), nl_(model.rows()),
      k_(2.0 * PI * freq / c), pvel_(c), z_(model.col(1)), beta_(model.col(3)),
      rho_(model.col(2)), mu_(rho_ * beta_ * beta_), nv_(grtc_->nv_),
      thickness_(nl_ - 1), cd_(grtc_->cd_), cu_(grtc_->cu_),
      matP_u_u_(nl_, Array22cd::Zero()), matP_uc_u_(nl_, Array22cd::Zero()),
      matP_uc_uc_(nl_, Array22cd::Zero()), matP_du_du_(nl_, Array22cd::Zero()),
      matP_duc_du_(nl_, Array22cd::Zero()),
      matP_duc_duc_(nl_, Array22cd::Zero()),
      sigma_x_sigma_top_(nl_, Array22cd::Zero()),
      sigmac_x_sigma_top_(nl_, Array22cd::Zero()),
      sigma_x_sigmac_top_(nl_, Array22cd::Zero()),
      sigmac_x_sigmac_top_(nl_, Array22cd::Zero()),
      sigma_x_sigma_bottom_(nl_, Array22cd::Zero()),
      sigmac_x_sigma_bottom_(nl_, Array22cd::Zero()),
      sigma_x_sigmac_bottom_(nl_, Array22cd::Zero()),
      sigmac_x_sigmac_bottom_(nl_, Array22cd::Zero()),
      int_ut2_(ArrayXd::Zero(nl_)), int_dut2_(ArrayXd::Zero(nl_)) {
  for (int i = 0; i < nl_ - 1; ++i) {
    thickness_(i) = z_(i + 1) - z_(i);
  }

  initialize_P();
  initialize_sigma();

  integrate_ut2();
  integrate_dut2();
}

void IntegralLayer::initialize_P() {
  for (auto i = 0; i < nl_; ++i) {
    const complex_d nv = nv_(i);
    const complex_d nvc = std::conj(nv_(i));

    matP_u_u_[i] << 1.0 / (-nv - nv), 1.0, 1.0, 1.0 / (nv + nv);
    matP_uc_u_[i] << 1.0 / (-nvc - nv), 1.0 / (-nvc + nv), 1.0 / (nvc - nv),
        1.0 / (nvc + nv);
    matP_uc_uc_[i] << 1.0 / (-nvc - nvc), 1.0, 1.0, 1.0 / (nvc + nvc);
    matP_du_du_[i] << nv / (-2.0), -nv * nv, -nv * nv, nv / 2.0;
    matP_duc_du_[i] << nvc * nv / (-nvc - nv), -nvc * nv / (-nvc + nv),
        -nvc * nv / (nvc - nv), nvc * nv / (nvc + nv);
    matP_duc_duc_[i] << nvc / (-2.0), -nvc * nvc, -nvc * nvc, nvc / 2.0;

    if (pvel_ > beta_(i)) {
      matP_uc_u_[i](0, 0) = 1.0;
      matP_uc_u_[i](1, 1) = 1.0;
      matP_duc_du_[i](0, 0) = nvc * nv;
      matP_duc_du_[i](1, 1) = nvc * nv;
    } else {
      matP_uc_u_[i](0, 1) = 1.0;
      matP_uc_u_[i](1, 0) = 1.0;
      matP_duc_du_[i](0, 1) = -nvc * nv;
      matP_duc_du_[i](1, 0) = -nvc * nv;
    }
  }
}

void IntegralLayer::initialize_sigma() {
  MatrixXcd sigma_bottom = MatrixXcd::Zero(2, nl_);
  MatrixXcd sigma_top = MatrixXcd::Zero(2, nl_);
  for (auto i = 0; i < nl_ - 1; ++i) {
    sigma_bottom.col(i) << cd_(i), cu_(i) * exp(-nv_(i) * thickness_(i));
    sigma_top.col(i) << cd_(i) * exp(-nv_(i) * thickness_(i)), cu_(i);
  }
  sigma_bottom.col(nl_ - 1) << cd_(nl_ - 1), 0;
  sigma_top.col(nl_ - 1).fill(0);

  for (auto id_layer = 0; id_layer < nl_; ++id_layer) {
    for (auto i = 0; i < 2; ++i) {
      for (auto j = 0; j < 2; ++j) {
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
      sigma_x_sigma_top_[id_layer](0, 1) *= thickness_(id_layer);
      sigma_x_sigma_top_[id_layer](1, 0) *= thickness_(id_layer);
      sigmac_x_sigmac_top_[id_layer](0, 1) *= thickness_(id_layer);
      sigmac_x_sigmac_top_[id_layer](1, 0) *= thickness_(id_layer);
      if (pvel_ > beta_(id_layer)) {
        sigmac_x_sigma_top_[id_layer](0, 0) =
            conj(cd_(id_layer)) * cd_(id_layer) * thickness_(id_layer);
        sigmac_x_sigma_top_[id_layer](1, 1) =
            conj(cu_(id_layer)) * cu_(id_layer) * thickness_(id_layer);
        sigma_x_sigmac_top_[id_layer](0, 0) =
            cd_(id_layer) * conj(cd_(id_layer)) * thickness_(id_layer);
        sigmac_x_sigma_top_[id_layer](1, 1) =
            cu_(id_layer) * conj(cu_(id_layer)) * thickness_(id_layer);
      } else {
        sigmac_x_sigma_top_[id_layer](0, 1) *= thickness_(id_layer);
        sigmac_x_sigma_top_[id_layer](1, 0) *= thickness_(id_layer);
        sigma_x_sigmac_top_[id_layer](0, 1) *= thickness_(id_layer);
        sigma_x_sigmac_top_[id_layer](1, 0) *= thickness_(id_layer);
      }
      sigma_x_sigma_bottom_[id_layer](0, 1) = 0.;
      sigma_x_sigma_bottom_[id_layer](1, 0) = 0.;
      sigmac_x_sigmac_bottom_[id_layer](0, 1) = 0.;
      sigmac_x_sigmac_bottom_[id_layer](1, 0) = 0.;

      if (pvel_ > beta_(id_layer)) {
        sigmac_x_sigma_bottom_[id_layer](0, 0) = 0.;
        sigmac_x_sigma_bottom_[id_layer](1, 1) = 0.;
        sigma_x_sigmac_bottom_[id_layer](0, 0) = 0.;
        sigma_x_sigmac_bottom_[id_layer](1, 1) = 0.;
      } else {
        sigmac_x_sigma_bottom_[id_layer](0, 1) = 0.;
        sigmac_x_sigma_bottom_[id_layer](1, 0) = 0.;
        sigma_x_sigmac_bottom_[id_layer](0, 1) = 0.;
        sigma_x_sigmac_bottom_[id_layer](1, 0) = 0.;
      }
    }
  }
}

double IntegralLayer::intker_ut2_top(int id_layer) {
  Array22cd auu = matP_u_u_[id_layer] * sigma_x_sigma_top_[id_layer];
  Array22cd aucu = matP_uc_u_[id_layer] * sigmac_x_sigma_top_[id_layer];
  Array22cd aucuc = matP_uc_uc_[id_layer] * sigmac_x_sigmac_top_[id_layer];
  complex_d result = auu.sum() + 2.0 * aucu.sum() + aucuc.sum();
  return result.real() / 4.0;
}

double IntegralLayer::intker_ut2_bottom(int id_layer) {
  Array22cd auu = matP_u_u_[id_layer] * sigma_x_sigma_bottom_[id_layer];
  Array22cd aucu = matP_uc_u_[id_layer] * sigmac_x_sigma_bottom_[id_layer];
  Array22cd aucuc = matP_uc_uc_[id_layer] * sigmac_x_sigmac_bottom_[id_layer];
  complex_d result = auu.sum() + 2.0 * aucu.sum() + aucuc.sum();
  return result.real() / 4.0;
}

double IntegralLayer::intker_dut2_top(int id_layer) {
  Array22cd auu = matP_du_du_[id_layer] * sigma_x_sigma_top_[id_layer];
  Array22cd aucu = matP_duc_du_[id_layer] * sigmac_x_sigma_top_[id_layer];
  Array22cd aucuc = matP_duc_duc_[id_layer] * sigmac_x_sigmac_top_[id_layer];
  complex_d result = auu.sum() + 2.0 * aucu.sum() + aucuc.sum();
  return result.real() / 4.0;
}

double IntegralLayer::intker_dut2_bottom(int id_layer) {
  Array22cd auu = matP_du_du_[id_layer] * sigma_x_sigma_bottom_[id_layer];
  Array22cd aucu = matP_duc_du_[id_layer] * sigmac_x_sigma_bottom_[id_layer];
  Array22cd aucuc = matP_duc_duc_[id_layer] * sigmac_x_sigmac_bottom_[id_layer];
  complex_d result = auu.sum() + 2.0 * aucu.sum() + aucuc.sum();
  return result.real() / 4.0;
}

void IntegralLayer::integrate_ut2() {
  for (int id_layer = 0; id_layer < nl_; ++id_layer) {
    int_ut2_(id_layer) = intker_ut2_top(id_layer) - intker_ut2_bottom(id_layer);
  }
}

void IntegralLayer::integrate_dut2() {
  for (int id_layer = 0; id_layer < nl_; ++id_layer) {
    int_dut2_(id_layer) =
        intker_dut2_top(id_layer) - intker_dut2_bottom(id_layer);
  }
}

double IntegralLayer::compute_I1() {
  ArrayXd ker = rho_ * int_ut2_;
  double i2 = 0.5 * ker.sum();
  return i2;
}

double IntegralLayer::compute_I2() {
  ArrayXd ker = mu_ * int_ut2_;
  double i2 = 0.5 * ker.sum();
  return i2;
}

double IntegralLayer::compute_I3() {
  ArrayXd ker = mu_ * int_dut2_;
  double i2 = 0.5 * ker.sum();
  return i2;
}

ArrayXd IntegralLayer::compute_kvs() {
  ArrayXd kvs(nl_);
  for (int i = 0; i < nl_; ++i) {
    kvs(i) =
        0.5 * rho_(i) * beta_(i) * (int_ut2_(i) + int_dut2_(i) / pow(k_, 2));
  }
  return kvs;
}

GradientSH::~GradientSH() = default;

ArrayXd GradientSH::compute(const double freq, const double c) const {
  IntegralLayer intl(model_, freq, c);
  double I2 = intl.compute_I2();
  ArrayXd kvs = intl.compute_kvs();

  kvs *= c / I2;

  return kvs;
}

} // namespace grad_sh
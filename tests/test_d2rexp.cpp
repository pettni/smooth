// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>

#include "smooth/diff.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"

namespace smooth {

namespace detail {

template<typename At, typename dAt, typename Bt, typename dBt>
auto hessian_product(const At & A, const dAt & dA, const Bt & B, const dBt & dB)
{
  using Scalar = std::common_type_t<
    typename At::Scalar,
    typename dAt::Scalar,
    typename Bt::Scalar,
    typename dBt::Scalar>;

  static constexpr int N    = At::ColsAtCompileTime;
  static constexpr int K    = At::RowsAtCompileTime;
  static constexpr int M    = Bt::RowsAtCompileTime;
  static constexpr int Nvar = dAt::SizeAtCompileTime / (N * K);

  static_assert(K == Bt::ColsAtCompileTime);
  static_assert(Nvar == dBt::SizeAtCompileTime / (K * M));

  Eigen::Matrix<Scalar, N, M * Nvar> dAB = B.transpose() * dA;
  for (auto n = 0u; n < N; ++n) {
    for (auto m = 0u; m < M; ++m) {
      dAB.template middleCols<Nvar>(n * Nvar) += A(n, m) * dB.template middleCols<Nvar>(m * Nvar);
    }
  }
  return dAB;
}

template<typename Scalar>
std::pair<Eigen::Matrix3<Scalar>, Eigen::Matrix<Scalar, 3, 18>>
calculate_Q_dQ(const Tangent<SE3<Scalar>> & a_in)
{
  const Eigen::Vector3<Scalar> v = a_in.template head<3>();
  const Eigen::Vector3<Scalar> w = a_in.template tail<3>();
  const Scalar th2               = w.squaredNorm();

  const auto [A, B, C, dA_over_th, dB_over_th, dC_over_th] = [&]() -> std::array<Scalar, 6> {
    if (th2 < Scalar(eps2)) {
      return {
        Scalar(1) / Scalar(6) - th2 / Scalar(120),
        Scalar(1) / Scalar(24) - th2 / Scalar(720),
        -Scalar(1) / Scalar(120) + th2 / Scalar(5040),
        -Scalar(1) / 60,
        -Scalar(1) / 360,
        Scalar(1) / 2520,
      };
    } else {
      const Scalar th  = sqrt(th2);
      const Scalar th3 = th2 * th;
      const Scalar th4 = th2 * th2;
      const Scalar th5 = th3 * th2;
      const Scalar th6 = th3 * th3;
      const Scalar th7 = th4 * th3;
      const Scalar sTh = sin(th);
      const Scalar cTh = cos(th);
      return {
        (th - sTh) / (th3),
        (cTh - Scalar(1) + th2 / Scalar(2)) / th4,
        (th - sTh - th * th2 / Scalar(6)) / th5,
        -cTh / th4 - 2 / th4 + 3 * sTh / th5,
        -1 / th4 - sTh / th5 - 4 * cTh / th6 + 4 / th6,
        1 / (3 * th4) - cTh / th6 - 4 / th6 + 5 * sTh / th7,
      };
    }
  }();

  const Eigen::Matrix3<Scalar> W  = smooth::SO3d::hat(w);
  const Eigen::Matrix3<Scalar> V  = smooth::SO3d::hat(v);
  const Eigen::Matrix3<Scalar> WV = W * V, VW = V * W, WW = W * W;
  const Scalar vdw                = v.dot(w);
  const Eigen::Matrix3<Scalar> PA = WV + VW - vdw * W;
  const Eigen::Matrix3<Scalar> PB = W * WV + VW * W + vdw * (3 * W - WW);
  const Eigen::Matrix3<Scalar> PC = -3 * vdw * WW;

  const Eigen::Matrix3<Scalar> Q = V / 2 + A * PA + B * PB + C * PC;

  // part with derivatives from matrices
  // clang-format off
  Eigen:: Matrix<Scalar, 3, 18> dQ{
    { w.x()*(B + 3*C)*(w.y()*w.y() + w.z()*w.z()), w.y()*(-2*A + B*(w.y()*w.y() + w.z()*w.z()) + 3*C*(w.y()*w.y() + w.z()*w.z())), w.z()*(-2*A + B*(w.y()*w.y() + w.z()*w.z()) + 3*C*(w.y()*w.y() + w.z()*w.z())), v.x()*(B + 3*C)*(w.y()*w.y() + w.z()*w.z()), -2*A*v.y() + B*v.y()*(w.y()*w.y() + w.z()*w.z()) + 2*B*w.y()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()) + 3*C*(v.y()*w.y()*w.y() + v.y()*w.z()*w.z() + 2*w.y()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())), -2*A*v.z() + B*v.z()*(w.y()*w.y() + w.z()*w.z()) + 2*B*w.z()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()) + 3*C*(v.z()*w.y()*w.y() + v.z()*w.z()*w.z() + 2*w.z()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())), -A*(w.x()*w.z() - w.y()) - B*w.x()*(w.x()*w.y() - 2*w.z()) - 3*C*w.x()*w.x()*w.y(), A*(w.x() - w.y()*w.z()) - B*w.y()*(w.x()*w.y() - 2*w.z()) - 3*C*w.x()*w.y()*w.y(), -A*w.z()*w.z() - B*w.x()*w.x() - B*w.x()*w.y()*w.z() - B*w.y()*w.y() + B*w.z()*w.z() - 3*C*w.x()*w.y()*w.z() + 0.5, -A*(v.x()*w.z() - v.y()) - B*(v.x()*w.z() + v.x()*(w.x()*w.y() - 3*w.z()) + 2*v.z()*w.x() + w.y()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())) - 3*C*v.x()*w.x()*w.y() - 3*C*w.y()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()), A*(v.x() - v.y()*w.z()) - B*(v.y()*w.z() + v.y()*(w.x()*w.y() - 3*w.z()) + 2*v.z()*w.y() + w.x()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())) - 3*C*v.y()*w.x()*w.y() - 3*C*w.x()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()), -A*(v.x()*w.x() + v.y()*w.y() + 2*v.z()*w.z()) + B*(2*v.x()*w.x() + 2*v.y()*w.y() - v.z()*w.z() - v.z()*(w.x()*w.y() - 3*w.z())) - 3*C*v.z()*w.x()*w.y(), A*(w.x()*w.y() + w.z()) - B*w.x()*(w.x()*w.z() + 2*w.y()) - 3*C*w.x()*w.x()*w.z(), A*w.y()*w.y() + B*w.x()*w.x() - B*w.x()*w.y()*w.z() - B*w.y()*w.y() + B*w.z()*w.z() - 3*C*w.x()*w.y()*w.z() - 0.5, A*(w.x() + w.y()*w.z()) - B*w.z()*(w.x()*w.z() + 2*w.y()) - 3*C*w.x()*w.z()*w.z(), A*(v.x()*w.y() + v.z()) + B*(v.x()*w.y() - v.x()*(w.x()*w.z() + 3*w.y()) + 2*v.y()*w.x() - w.z()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())) - 3*C*v.x()*w.x()*w.z() - 3*C*w.z()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()), A*(v.x()*w.x() + 2*v.y()*w.y() + v.z()*w.z()) - B*(2*v.x()*w.x() - v.y()*w.y() + v.y()*(w.x()*w.z() + 3*w.y()) + 2*v.z()*w.z()) - 3*C*v.y()*w.x()*w.z(), A*(v.x() + v.z()*w.y()) + B*(2*v.y()*w.z() + v.z()*w.y() - v.z()*(w.x()*w.z() + 3*w.y()) - w.x()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())) - 3*C*v.z()*w.x()*w.z() - 3*C*w.x()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()) },
  { A*(w.x()*w.z() + w.y()) - B*w.x()*(w.x()*w.y() + 2*w.z()) - 3*C*w.x()*w.x()*w.y(), A*(w.x() + w.y()*w.z()) - B*w.y()*(w.x()*w.y() + 2*w.z()) - 3*C*w.x()*w.y()*w.y(), A*w.z()*w.z() + B*w.x()*w.x() - B*w.x()*w.y()*w.z() + B*w.y()*w.y() - B*w.z()*w.z() - 3*C*w.x()*w.y()*w.z() - 0.5, A*(v.x()*w.z() + v.y()) + B*(v.x()*w.z() - v.x()*(w.x()*w.y() + 3*w.z()) + 2*v.z()*w.x() - w.y()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())) - 3*C*v.x()*w.x()*w.y() - 3*C*w.y()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()), A*(v.x() + v.y()*w.z()) + B*(v.y()*w.z() - v.y()*(w.x()*w.y() + 3*w.z()) + 2*v.z()*w.y() - w.x()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())) - 3*C*v.y()*w.x()*w.y() - 3*C*w.x()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()), A*(v.x()*w.x() + v.y()*w.y() + 2*v.z()*w.z()) - B*(2*v.x()*w.x() + 2*v.y()*w.y() - v.z()*w.z() + v.z()*(w.x()*w.y() + 3*w.z())) - 3*C*v.z()*w.x()*w.y(), w.x()*(-2*A + B*(w.x()*w.x() + w.z()*w.z()) + 3*C*(w.x()*w.x() + w.z()*w.z())), w.y()*(B + 3*C)*(w.x()*w.x() + w.z()*w.z()), w.z()*(-2*A + B*(w.x()*w.x() + w.z()*w.z()) + 3*C*(w.x()*w.x() + w.z()*w.z())), -2*A*v.x() + B*v.x()*(w.x()*w.x() + w.z()*w.z()) + 2*B*w.x()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()) + 3*C*(v.x()*w.x()*w.x() + v.x()*w.z()*w.z() + 2*w.x()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())), v.y()*(B + 3*C)*(w.x()*w.x() + w.z()*w.z()), -2*A*v.z() + B*v.z()*(w.x()*w.x() + w.z()*w.z()) + 2*B*w.z()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()) + 3*C*(v.z()*w.x()*w.x() + v.z()*w.z()*w.z() + 2*w.z()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())), -A*w.x()*w.x() + B*w.x()*w.x() - B*w.x()*w.y()*w.z() - B*w.y()*w.y() - B*w.z()*w.z() - 3*C*w.x()*w.y()*w.z() + 0.5, -A*(w.x()*w.y() - w.z()) + B*w.y()*(2*w.x() - w.y()*w.z()) - 3*C*w.y()*w.y()*w.z(), -A*(w.x()*w.z() - w.y()) + B*w.z()*(2*w.x() - w.y()*w.z()) - 3*C*w.y()*w.z()*w.z(), -A*(2*v.x()*w.x() + v.y()*w.y() + v.z()*w.z()) + B*(-v.x()*w.x() + v.x()*(3*w.x() - w.y()*w.z()) + 2*v.y()*w.y() + 2*v.z()*w.z()) - 3*C*v.x()*w.y()*w.z(), -A*(v.y()*w.x() - v.z()) - B*(2*v.x()*w.y() + v.y()*w.x() - v.y()*(3*w.x() - w.y()*w.z()) + w.z()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())) - 3*C*v.y()*w.y()*w.z() - 3*C*w.z()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()), A*(v.y() - v.z()*w.x()) - B*(2*v.x()*w.z() + v.z()*w.x() - v.z()*(3*w.x() - w.y()*w.z()) + w.y()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())) - 3*C*v.z()*w.y()*w.z() - 3*C*w.y()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()) },
  { -A*(w.x()*w.y() - w.z()) - B*w.x()*(w.x()*w.z() - 2*w.y()) - 3*C*w.x()*w.x()*w.z(), -A*w.y()*w.y() - B*w.x()*w.x() - B*w.x()*w.y()*w.z() + B*w.y()*w.y() - B*w.z()*w.z() - 3*C*w.x()*w.y()*w.z() + 0.5, A*(w.x() - w.y()*w.z()) - B*w.z()*(w.x()*w.z() - 2*w.y()) - 3*C*w.x()*w.z()*w.z(), -A*(v.x()*w.y() - v.z()) - B*(v.x()*w.y() + v.x()*(w.x()*w.z() - 3*w.y()) + 2*v.y()*w.x() + w.z()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())) - 3*C*v.x()*w.x()*w.z() - 3*C*w.z()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()), -A*(v.x()*w.x() + 2*v.y()*w.y() + v.z()*w.z()) + B*(2*v.x()*w.x() - v.y()*w.y() - v.y()*(w.x()*w.z() - 3*w.y()) + 2*v.z()*w.z()) - 3*C*v.y()*w.x()*w.z(), A*(v.x() - v.z()*w.y()) - B*(2*v.y()*w.z() + v.z()*w.y() + v.z()*(w.x()*w.z() - 3*w.y()) + w.x()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())) - 3*C*v.z()*w.x()*w.z() - 3*C*w.x()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()), A*w.x()*w.x() - B*w.x()*w.x() - B*w.x()*w.y()*w.z() + B*w.y()*w.y() + B*w.z()*w.z() - 3*C*w.x()*w.y()*w.z() - 0.5, A*(w.x()*w.y() + w.z()) - B*w.y()*(2*w.x() + w.y()*w.z()) - 3*C*w.y()*w.y()*w.z(), A*(w.x()*w.z() + w.y()) - B*w.z()*(2*w.x() + w.y()*w.z()) - 3*C*w.y()*w.z()*w.z(), A*(2*v.x()*w.x() + v.y()*w.y() + v.z()*w.z()) - B*(-v.x()*w.x() + v.x()*(3*w.x() + w.y()*w.z()) + 2*v.y()*w.y() + 2*v.z()*w.z()) - 3*C*v.x()*w.y()*w.z(), A*(v.y()*w.x() + v.z()) + B*(2*v.x()*w.y() + v.y()*w.x() - v.y()*(3*w.x() + w.y()*w.z()) - w.z()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())) - 3*C*v.y()*w.y()*w.z() - 3*C*w.z()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()), A*(v.y() + v.z()*w.x()) + B*(2*v.x()*w.z() + v.z()*w.x() - v.z()*(3*w.x() + w.y()*w.z()) - w.y()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())) - 3*C*v.z()*w.y()*w.z() - 3*C*w.y()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()), w.x()*(-2*A + B*(w.x()*w.x() + w.y()*w.y()) + 3*C*(w.x()*w.x() + w.y()*w.y())), w.y()*(-2*A + B*(w.x()*w.x() + w.y()*w.y()) + 3*C*(w.x()*w.x() + w.y()*w.y())), w.z()*(B + 3*C)*(w.x()*w.x() + w.y()*w.y()), -2*A*v.x() + B*v.x()*(w.x()*w.x() + w.y()*w.y()) + 2*B*w.x()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()) + 3*C*(v.x()*w.x()*w.x() + v.x()*w.y()*w.y() + 2*w.x()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())), -2*A*v.y() + B*v.y()*(w.x()*w.x() + w.y()*w.y()) + 2*B*w.y()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z()) + 3*C*(v.y()*w.x()*w.x() + v.y()*w.y()*w.y() + 2*w.y()*(v.x()*w.x() + v.y()*w.y() + v.z()*w.z())), v.z()*(B + 3*C)*(w.x()*w.x() + w.y()*w.y()) },
  };
  // clang-format on

  // parts with dA, dB, dC
  for (auto i = 0u; i < 3; ++i) {
    const Scalar dA_dwi = dA_over_th * w(i);
    const Scalar dB_dwi = dB_over_th * w(i);
    const Scalar dC_dwi = dC_over_th * w(i);
    for (auto j = 0u; j < 3; ++j) {
      dQ.col(3 + i + 6 * j) += dA_dwi * PA.row(j).transpose() + dB_dwi * PB.row(j).transpose()
                             + dC_dwi * PC.row(j).transpose();
    }
  }

  return {Q, dQ};
}

}  // namespace detail

// SO2
template<typename Scalar>
Eigen::Matrix<Scalar, 1, 1> d2r_exp_so2(const Tangent<SO2<Scalar>> &)
{
  return Eigen::Matrix<Scalar, 1, 1>::Zero();
}

template<typename Scalar>
Eigen::Matrix<Scalar, 1, 1> d2r_expinv_so2(const Tangent<SO2<Scalar>> &)
{
  return Eigen::Matrix<Scalar, 1, 1>::Zero();
}

// SE2
template<typename Scalar>
Eigen::Matrix<Scalar, 3, 9> d2r_exp_se2(const Tangent<SE2<Scalar>> & a_in)
{
  const auto [A, B, dA_dwz, dB_dwz] = [&]() -> std::array<Scalar, 4> {
    const Scalar wz  = a_in.z();
    const Scalar wz2 = wz * wz;

    if (wz2 < Scalar(eps2)) {
      return {
        Scalar(0.5) - wz2 / 24,
        Scalar(1. / 6) - wz2 / 120,
        -wz / 48,
        -wz / 60,
      };
    } else {
      const Scalar sTh = sin(wz);
      const Scalar cTh = cos(wz);
      const Scalar wz3 = wz2 * wz;
      const Scalar wz4 = wz2 * wz2;
      return {
        (Scalar(1) - cTh) / wz2,
        (wz - sTh) / wz3,
        sTh / wz2 + 2 * cTh / wz3 - 2 / wz3,
        -cTh / wz3 - 2 / wz3 + 3 * sTh / wz4,
      };
    }
  }();

  // -A * d(ad) + B * d(ad^2)
  // clang-format off
  Eigen::Matrix<Scalar, 3, 9> ret{
    { 0, 0, -2*B*a_in.z(), 0, 0, -A, 0, 0, 0 },
    { 0, 0, A, 0, 0, -2*B*a_in.z(), 0, 0, 0 },
    { B*a_in.z(), -A, B*a_in.x(), A, B*a_in.z(), B*a_in.y(), 0, 0, 0 }
  };
  // clang-format on

  // add -dA * ad + dB * ad^2
  const Eigen::Matrix3<Scalar> ad  = smooth::SE2d::ad(a_in);
  const Eigen::Matrix3<Scalar> ad2 = ad * ad;

  for (auto j = 0u; j < 3; ++j) {
    ret.col(2 + 3 * j) -= dA_dwz * ad.row(j).transpose();
    ret.col(2 + 3 * j) += dB_dwz * ad2.row(j).transpose();
  }

  return ret;
}

template<typename Scalar>
Eigen::Matrix<Scalar, 3, 9> d2r_expinv_se2(const Tangent<SE2<Scalar>> & a_in)
{
  const auto [A, dA_dwz] = [&]() -> std::array<Scalar, 2> {
    const Scalar wz  = a_in.z();
    const Scalar wz2 = wz * wz;

    if (wz2 < Scalar(eps2)) {
      return {
        Scalar(1) / Scalar(12) + wz2 / Scalar(720),
        Scalar(1) / Scalar(360),
      };
    } else {
      const Scalar sTh = sin(wz);
      const Scalar cTh = cos(wz);
      const Scalar wz3 = wz2 * wz;
      return {
        Scalar(1) / wz2 - (Scalar(1) + cTh) / (Scalar(2) * wz * sTh),
        1 / (2 * wz) + cTh * cTh / (2 * wz * sTh * sTh) + cTh / (2 * wz * sTh * sTh)
          + cTh / (2 * wz2 * sTh) + 1 / (2 * wz2 * sTh) - 2 / wz3,
      };
    }
  }();

  // -A * d(ad) + B * d(ad^2)
  // clang-format off
  Eigen::Matrix<Scalar, 3, 9> ret{
    { 0, 0, -2*A*a_in.z(), 0, 0, 0.5, 0, 0, 0 },
    { 0, 0, -0.5, 0, 0, -2*A*a_in.z(), 0, 0, 0 },
    { A*a_in.z(), 0.5, A*a_in.x(), -0.5, A*a_in.z(), A*a_in.y(), 0, 0, 0 },
  };
  // clang-format on

  // add -dA * ad + dB * ad^2
  const Eigen::Matrix3<Scalar> ad  = smooth::SE2d::ad(a_in);
  const Eigen::Matrix3<Scalar> ad2 = ad * ad;

  for (auto j = 0u; j < 3; ++j) { ret.col(2 + 3 * j) += dA_dwz * ad2.row(j).transpose(); }

  return ret;
}

// SO3
template<typename Scalar>
Eigen::Matrix<Scalar, 3, 9> d2r_exp_so3(const Tangent<SO3<Scalar>> & a_in)
{
  const auto [A, B, dA_over_th, dB_over_th] = [&]() -> std::array<Scalar, 4> {
    using std::sqrt;

    const Scalar th2 = a_in.squaredNorm();
    const Scalar th  = sqrt(th2);

    if (th2 < Scalar(eps2)) {
      return {
        Scalar(0.5) - th2 / 24,
        Scalar(1. / 6) - th2 / 120,
        -Scalar(1) / 48,
        -Scalar(1) / 60,
      };
    } else {
      const Scalar sTh = sin(th);
      const Scalar cTh = cos(th);
      const Scalar th3 = th2 * th;
      const Scalar th4 = th2 * th2;
      const Scalar th5 = th3 * th2;
      return {
        (Scalar(1) - cTh) / th2,
        (th - sTh) / th3,
        sTh / th3 + 2 * cTh / th4 - 2 / th4,
        -cTh / th4 - 2 / th4 + 3 * sTh / th5,
      };
    }
  }();

  // -A * d(ad) + B * d(ad^2)
  // clang-format off
  Eigen::Matrix<Scalar, 3, 9> ret{
    {0., -2. * B * a_in.y(), -2. * B * a_in.z(), B * a_in.y(), B * a_in.x(), -A, B * a_in.z(), A, B * a_in.x()},
    {B * a_in.y(), B * a_in.x(), A, -2. * B * a_in.x(), 0., -2. * B * a_in.z(), -A, B * a_in.z(), B * a_in.y()},
    {B * a_in.z(), -A, B * a_in.x(), A, B * a_in.z(), B * a_in.y(), -2. * B * a_in.x(), -2 * B * a_in.y(), 0.},
  };
  // clang-format on

  // add -dA * ad + dB * ad^2
  const Eigen::Matrix3<Scalar> ad  = smooth::SO3d::ad(a_in);
  const Eigen::Matrix3<Scalar> ad2 = ad * ad;
  for (auto i = 0u; i < 3; ++i) {
    const Scalar dA_dxi = dA_over_th * a_in(i);
    const Scalar dB_dxi = dB_over_th * a_in(i);
    for (auto j = 0u; j < 3; ++j) {
      ret.col(i + 3 * j) -= dA_dxi * ad.row(j).transpose();
      ret.col(i + 3 * j) += dB_dxi * ad2.row(j).transpose();
    }
  }

  return ret;
}

template<typename Scalar>
Eigen::Matrix<Scalar, 3, 9> d2r_expinv_so3(const Tangent<SO3<Scalar>> & a_in)
{
  const auto [A, dA_over_th] = [&]() -> std::array<Scalar, 2> {
    using std::sqrt;

    const Scalar th2 = a_in.squaredNorm();
    const Scalar th  = sqrt(th2);
    if (th2 < Scalar(eps2)) {
      return {
        Scalar(1) / Scalar(12) + th2 / Scalar(720),
        Scalar(1) / Scalar(360),
      };
    } else {
      const Scalar th3 = th2 * th;
      const Scalar th4 = th2 * th2;
      const Scalar sTh = sin(th);
      const Scalar cTh = cos(th);
      return {
        Scalar(1) / th2 - (Scalar(1) + cTh) / (Scalar(2) * th * sTh),
        1 / (2 * th2) + cTh * cTh / (2 * th2 * sTh * sTh) + cTh / (2 * th2 * sTh * sTh)
          + cTh / (2 * th3 * sTh) + 1 / (2 * th3 * sTh) - 2 / th4,
      };
    }
  }();

  // A * d(ad^2)
  // clang-format off
  Eigen::Matrix<Scalar, 3, 9> ret{
    {0, -2 * A * a_in.y(), -2 * A * a_in.z(), A * a_in.y(), A * a_in.x(), 0.5, A * a_in.z(), -0.5, A * a_in.x()},
    {A * a_in.y(), A * a_in.x(), -0.5, -2 * A * a_in.x(), 0, -2 * A * a_in.z(), 0.5, A * a_in.z(), A * a_in.y()},
    {A * a_in.z(), 0.5, A * a_in.x(), -0.5, A * a_in.z(), A * a_in.y(), -2 * A * a_in.x(), -2 * A * a_in.y(), 0},
  };
  // clang-format on

  // add dA * ad^2
  const Eigen::Matrix3<Scalar> ad  = smooth::SO3d::hat(a_in);
  const Eigen::Matrix3<Scalar> ad2 = ad * ad;
  for (auto i = 0u; i < 3; ++i) {
    const Scalar dA_dxi = dA_over_th * a_in(i);
    for (auto j = 0u; j < 3; ++j) { ret.col(i + 3 * j) += dA_dxi * ad2.row(j).transpose(); }
  }

  return ret;
}

// SE3
template<typename Scalar>
Eigen::Matrix<Scalar, 6, 36> d2r_exp_se3(const Tangent<SE3<Scalar>> & a_in)
{
  Eigen::Matrix<Scalar, 6, 36> ret = Eigen::Matrix<Scalar, 6, 36>::Zero();

  // DERIVATIVES OF SO3 JACOBIAN
  const Eigen::Matrix<Scalar, 3, 9> Hso3 = d2r_exp_so3<Scalar>(a_in.template tail<3>());

  for (auto i = 0u; i < 3; ++i) {
    ret.template block<3, 3>(0, 6 * i + 3)      = Hso3.template block<3, 3>(0, 3 * i);
    ret.template block<3, 3>(3, 18 + 6 * i + 3) = Hso3.template block<3, 3>(0, 3 * i);
  }

  // DERIVATIVE OF Q TERM
  const auto [Q, dQ]              = detail::calculate_Q_dQ<Scalar>(-a_in);
  ret.template block<3, 18>(3, 0) = -dQ;

  return ret;
}

template<typename Scalar>
Eigen::Matrix<Scalar, 6, 36> d2r_expinv_se3(const Tangent<SE3<Scalar>> & a_in)
{
  Eigen::Matrix<Scalar, 6, 36> ret = Eigen::Matrix<Scalar, 6, 36>::Zero();

  // DERIVATIVES OF SO3 JACOBIAN
  const Eigen::Matrix<Scalar, 3, 9> Hso3 = d2r_expinv_so3<Scalar>(a_in.template tail<3>());

  for (auto i = 0u; i < 3; ++i) {
    ret.template block<3, 3>(0, 6 * i + 3)      = Hso3.template block<3, 3>(0, 3 * i);
    ret.template block<3, 3>(3, 18 + 6 * i + 3) = Hso3.template block<3, 3>(0, 3 * i);
  }

  // DERIVATIVE OF -J Q J TERM
  auto [Q, dQ] = detail::calculate_Q_dQ<Scalar>(-a_in);
  dQ *= -1;  // account for -a_in

  const Eigen::Matrix3<Scalar> Jso3 = dr_expinv<SO3<Scalar>>(a_in.template tail<3>());
  // Hso3 contains derivatives w.r.t. w, we extend for derivatives w.r.t. [v, w]
  Eigen::Matrix<Scalar, 3, 18> Hso3_exp = Eigen::Matrix<Scalar, 3, 18>::Zero();
  for (auto i = 0u; i < 3; ++i) {
    Hso3_exp.template middleCols<3>(6 * i + 3) = Hso3.template middleCols<3>(3 * i);
  }

  const Eigen::Matrix3<Scalar> Jtmp       = Jso3 * Q;
  const Eigen::Matrix<Scalar, 3, 18> Htmp = detail::hessian_product(Jso3, Hso3_exp, Q, dQ);
  ret.template block<3, 18>(3, 0)         = -detail::hessian_product(Jtmp, Htmp, Jso3, Hso3_exp);

  return ret;
}

template<LieGroup G>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> d2r_exp(const Tangent<G> & a)
{
  if constexpr (std::is_base_of_v<SO2Base<G>, G>) {
    return d2r_exp_so2<Scalar<G>>(a);
  } else if constexpr (std::is_base_of_v<SE2Base<G>, G>) {
    return d2r_exp_se2<Scalar<G>>(a);
  } else if constexpr (std::is_base_of_v<SO3Base<G>, G>) {
    return d2r_exp_so3<Scalar<G>>(a);
  } else if constexpr (std::is_base_of_v<SE3Base<G>, G>) {
    return d2r_exp_se3<Scalar<G>>(a);
  }
}

template<LieGroup G>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> d2r_expinv(const Tangent<G> & a)
{
  if constexpr (std::is_base_of_v<SO2Base<G>, G>) {
    return d2r_exp_so2<Scalar<G>>(a);
  } else if constexpr (std::is_base_of_v<SE2Base<G>, G>) {
    return d2r_expinv_se2<Scalar<G>>(a);
  } else if constexpr (std::is_base_of_v<SO3Base<G>, G>) {
    return d2r_expinv_so3<Scalar<G>>(a);
  } else if constexpr (std::is_base_of_v<SE3Base<G>, G>) {
    return d2r_expinv_se3<Scalar<G>>(a);
  }
}
}  // namespace smooth

namespace smooth {

template<LieGroup G>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> d2r_exp_autodiff(const Tangent<G> & a)
{
  static constexpr auto N   = Dof<G>;
  const auto [expa, drexpa] = diff::dr<1>(
    []<typename T>(const CastT<T, Tangent<G>> & var) -> Eigen::Vector<T, N * N> {
      return dr_exp<CastT<T, G>>(var).transpose().reshaped();
    },
    wrt(a));

  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> ret;
  for (auto i = 0u; i < Dof<G>; ++i) {
    ret.template block<Dof<G>, Dof<G>>(0, Dof<G> * i) =
      drexpa.template block<Dof<G>, Dof<G>>(Dof<G> * i, 0);
  }
  return ret;
}

template<LieGroup G>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> d2r_expinv_autodiff(const Tangent<G> & a)
{
  static constexpr auto N   = Dof<G>;
  const auto [expa, drexpa] = diff::dr<1>(
    []<typename T>(const CastT<T, Tangent<G>> & var) -> Eigen::Vector<T, N * N> {
      return dr_expinv<CastT<T, G>>(var).transpose().reshaped();
    },
    wrt(a));

  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> ret;
  for (auto i = 0u; i < Dof<G>; ++i) {
    ret.template block<Dof<G>, Dof<G>>(0, Dof<G> * i) =
      drexpa.template block<Dof<G>, Dof<G>>(Dof<G> * i, 0);
  }
  return ret;
}

}  // namespace smooth

template<smooth::LieGroup G>
class SecondDerivatives : public ::testing::Test
{};

using TestGroups = ::testing::Types<smooth::SO2d, smooth::SE2d, smooth::SO3d, smooth::SE3d>;
TYPED_TEST_SUITE(SecondDerivatives, TestGroups);

TYPED_TEST(SecondDerivatives, d2rexp)
{
  for (auto i = 0u; i < 5; ++i) {
    smooth::Tangent<TypeParam> a;
    a.setRandom();

    if (i == 0) { a *= 1e-6; }

    const auto d2rexp_ana  = smooth::d2r_exp<TypeParam>(a);
    const auto d2rexp_diff = smooth::d2r_exp_autodiff<TypeParam>(a);

    ASSERT_TRUE(d2rexp_ana.isApprox(d2rexp_diff, 1e-4));
  }
}

TYPED_TEST(SecondDerivatives, d2rexpinv)
{
  for (auto i = 0u; i < 5; ++i) {
    smooth::Tangent<TypeParam> a;
    a.setRandom();

    if (i == 0) { a *= 1e-6; }

    const auto d2rexp_ana  = smooth::d2r_expinv<TypeParam>(a);
    const auto d2rexp_diff = smooth::d2r_expinv_autodiff<TypeParam>(a);

    ASSERT_TRUE(d2rexp_ana.isApprox(d2rexp_diff, 1e-4));
  }
}

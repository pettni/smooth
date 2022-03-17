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

#ifndef SMOOTH__INTERNAL__SE3_HPP_
#define SMOOTH__INTERNAL__SE3_HPP_

#include <Eigen/Core>

#include "common.hpp"
#include "smooth/diff.hpp"
#include "so3.hpp"

namespace smooth {

/**
 * @brief SE(3) Lie Group represented as S^3 ⋉ R3
 *
 * Memory layout
 * -------------
 * Group:    x y z qx qy qz qw
 * Tangent:  vx vy vz Ωx Ωy Ωz
 *
 * Lie group Matrix form
 * ---------------------
 * [ R T ]
 * [ 0 1 ]
 *
 * where R ∈ SO(3) and T = [x y z] ∈ R3
 *
 * Lie algebra Matrix form
 * -----------------------
 * [  0 -Ωz  Ωy vx]
 * [  Ωz  0 -Ωx vy]
 * [ -Ωy Ωx   0 vz]
 * [   0  0   0  0]
 *
 * Constraints
 * -----------
 * Group:   qx * qx + qy * qy + qz * qz + qw * qw = 1
 * Tangent: -pi < Ωx Ωy Ωz <= pi
 */
template<typename _Scalar>
class SE3Impl
{
public:
  using Scalar = _Scalar;

  static constexpr Eigen::Index RepSize = 7;
  static constexpr Eigen::Index Dim     = 4;
  static constexpr Eigen::Index Dof     = 6;

  SMOOTH_DEFINE_REFS;

  static void setIdentity(GRefOut g_out)
  {
    g_out.template head<6>().setZero();
    g_out(6) = Scalar(1);
  }

  static void setRandom(GRefOut g_out)
  {
    g_out.template head<3>().setRandom();
    SO3Impl<Scalar>::setRandom(g_out.template tail<4>());
  }

  static void matrix(GRefIn g_in, MRefOut m_out)
  {
    m_out.setIdentity();
    SO3Impl<Scalar>::matrix(g_in.template tail<4>(), m_out.template topLeftCorner<3, 3>());
    m_out.template topRightCorner<3, 1>() = g_in.template head<3>();
  }

  static void composition(GRefIn g_in1, GRefIn g_in2, GRefOut g_out)
  {
    SO3Impl<Scalar>::composition(
      g_in1.template tail<4>(), g_in2.template tail<4>(), g_out.template tail<4>());
    Eigen::Matrix<Scalar, 3, 3> R1;
    SO3Impl<Scalar>::matrix(g_in1.template tail<4>(), R1);
    g_out.template head<3>().noalias() = R1 * g_in2.template head<3>() + g_in1.template head<3>();
  }

  static void inverse(GRefIn g_in, GRefOut g_out)
  {
    Eigen::Matrix<Scalar, 4, 1> so3inv;
    SO3Impl<Scalar>::inverse(g_in.template tail<4>(), so3inv);

    Eigen::Matrix<Scalar, 3, 3> Rinv;
    SO3Impl<Scalar>::matrix(so3inv, Rinv);

    g_out.template head<3>().noalias() = -Rinv * g_in.template head<3>();
    g_out.template tail<4>()           = so3inv;
  }

  static void log(GRefIn g_in, TRefOut a_out)
  {
    using SO3TangentMap = Eigen::Matrix<Scalar, 3, 3>;

    SO3Impl<Scalar>::log(g_in.template tail<4>(), a_out.template tail<3>());

    SO3TangentMap M_dr_expinv, M_ad;
    SO3Impl<Scalar>::dr_expinv(a_out.template tail<3>(), M_dr_expinv);
    SO3Impl<Scalar>::ad(a_out.template tail<3>(), M_ad);
    a_out.template head<3>().noalias() = (-M_ad + M_dr_expinv) * g_in.template head<3>();
  }

  static void Ad(GRefIn g_in, TMapRefOut A_out)
  {
    SO3Impl<Scalar>::matrix(g_in.template tail<4>(), A_out.template topLeftCorner<3, 3>());
    SO3Impl<Scalar>::hat(g_in.template head<3>(), A_out.template topRightCorner<3, 3>());
    A_out.template topRightCorner<3, 3>() *= A_out.template topLeftCorner<3, 3>();
    A_out.template bottomRightCorner<3, 3>() = A_out.template topLeftCorner<3, 3>();
    A_out.template bottomLeftCorner<3, 3>().setZero();
  }

  static void exp(TRefIn a_in, GRefOut g_out)
  {
    using SO3TangentMap = Eigen::Matrix<Scalar, 3, 3>;

    SO3Impl<Scalar>::exp(a_in.template tail<3>(), g_out.template tail<4>());

    SO3TangentMap M_dr_exp, M_Ad;
    SO3Impl<Scalar>::dr_exp(a_in.template tail<3>(), M_dr_exp);
    SO3Impl<Scalar>::Ad(g_out.template tail<4>(), M_Ad);

    g_out.template head<3>().noalias() = M_Ad * M_dr_exp * a_in.template head<3>();
  }

  static void hat(TRefIn a_in, MRefOut A_out)
  {
    A_out.setZero();
    SO3Impl<Scalar>::hat(a_in.template tail<3>(), A_out.template topLeftCorner<3, 3>());
    A_out.template topRightCorner<3, 1>() = a_in.template head<3>();
  }

  static void vee(MRefIn A_in, TRefOut a_out)
  {
    SO3Impl<Scalar>::vee(A_in.template topLeftCorner<3, 3>(), a_out.template tail<3>());
    a_out.template head<3>() = A_in.template topRightCorner<3, 1>();
  }

  static void ad(TRefIn a_in, TMapRefOut A_out)
  {
    SO3Impl<Scalar>::hat(a_in.template tail<3>(), A_out.template topLeftCorner<3, 3>());
    SO3Impl<Scalar>::hat(a_in.template head<3>(), A_out.template topRightCorner<3, 3>());
    A_out.template bottomRightCorner<3, 3>() = A_out.template topLeftCorner<3, 3>();
    A_out.template bottomLeftCorner<3, 3>().setZero();
  }

  static Eigen::Matrix<Scalar, 3, 3> calculate_q(TRefIn a)
  {
    using std::abs, std::sqrt, std::cos, std::sin;

    const Scalar th2 = a.template tail<3>().squaredNorm();

    const auto [A, B, C] = [&]() -> std::array<Scalar, 3> {
      if (th2 < Scalar(eps2)) {
        return {
          // https://www.wolframalpha.com/input/?i=series+%28x+-+sin+x%29+%2F+x%5E3+at+x%3D0
          Scalar(1) / Scalar(6) - th2 / Scalar(120),
          // https://www.wolframalpha.com/input/?i=series+%28cos+x+-+1+%2B+x%5E2%2F2%29+%2F+x%5E4+at+x%3D0
          Scalar(1) / Scalar(24) - th2 / Scalar(720),
          // https://www.wolframalpha.com/input/?i=series+%28x+-+sin+x+-+x%5E3%2F6%29+%2F+x%5E5+at+x%3D0
          -Scalar(1) / Scalar(120) + th2 / Scalar(5040),
        };
      } else {
        const Scalar th = sqrt(th2), th_4 = th2 * th2, cTh = cos(th), sTh = sin(th);
        return {
          (th - sTh) / (th * th2),
          (cTh - Scalar(1) + th2 / Scalar(2)) / th_4,
          (th - sTh - th * th2 / Scalar(6)) / (th_4 * th),
        };
      }
    }();

    Eigen::Matrix<Scalar, 3, 3> V, W;
    SO3Impl<Scalar>::hat(a.template head<3>(), V);
    SO3Impl<Scalar>::hat(a.template tail<3>(), W);

    const Scalar vdw                     = a.template tail<3>().dot(a.template head<3>());
    const Eigen::Matrix<Scalar, 3, 3> WV = W * V, VW = V * W, WW = W * W;

    // clang-format off
    return Scalar(0.5) * V + A * (WV + VW - vdw * W) + B * (W * WV + VW * W + vdw * (Scalar(3) * W - WW)) - C * Scalar(3) * vdw * WW;
    // clang-format on
  }

  static std::pair<Eigen::Matrix3<Scalar>, Eigen::Matrix<Scalar, 3, 18>> calculate_Q_dQ(TRefIn a)
  {
    const Eigen::Vector3<Scalar> v = a.template head<3>();
    const Eigen::Vector3<Scalar> w = a.template tail<3>();
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

    Eigen::Matrix<Scalar, 3, 3> V, W;
    SO3Impl<Scalar>::hat(a.template head<3>(), V);
    SO3Impl<Scalar>::hat(a.template tail<3>(), W);
    const Scalar vdw = v.dot(w);

    const Eigen::Matrix3<Scalar> WV = W * V, VW = V * W, WW = W * W, PA = WV + VW - vdw * W,
                                 PB = W * WV + VW * W + vdw * (3 * W - WW), PC = -3 * vdw * WW;

    Eigen::Matrix3<Scalar> Q = V / 2 + A * PA + B * PB + C * PC;

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

    return std::make_pair(std::move(Q), std::move(dQ));
  }

  static void dr_exp(TRefIn a_in, TMapRefOut A_out)
  {
    SO3Impl<Scalar>::dr_exp(a_in.template tail<3>(), A_out.template topLeftCorner<3, 3>());
    A_out.template topRightCorner<3, 3>()    = calculate_q(-a_in);
    A_out.template bottomRightCorner<3, 3>() = A_out.template topLeftCorner<3, 3>();
    A_out.template bottomLeftCorner<3, 3>().setZero();
  }

  static void dr_expinv(TRefIn a_in, TMapRefOut A_out)
  {
    SO3Impl<Scalar>::dr_expinv(a_in.template tail<3>(), A_out.template topLeftCorner<3, 3>());
    A_out.template topRightCorner<3, 3>().noalias() = -A_out.template topLeftCorner<3, 3>()
                                                    * calculate_q(-a_in)
                                                    * A_out.template topLeftCorner<3, 3>();
    A_out.template bottomRightCorner<3, 3>() = A_out.template topLeftCorner<3, 3>();
    A_out.template bottomLeftCorner<3, 3>().setZero();
  }

  static void d2r_exp(TRefIn a_in, THessRefOut H_out)
  {
    H_out.setZero();

    // DERIVATIVES OF SO3 JACOBIAN
    Eigen::Matrix<Scalar, 3, 9> Hso3;
    SO3Impl<Scalar>::d2r_exp(a_in.template tail<3>(), Hso3);

    for (auto i = 0u; i < 3; ++i) {
      H_out.template block<3, 3>(0, 6 * i + 3)      = Hso3.template block<3, 3>(0, 3 * i);
      H_out.template block<3, 3>(3, 18 + 6 * i + 3) = Hso3.template block<3, 3>(0, 3 * i);
    }

    // DERIVATIVE OF Q TERM
    const auto [Q, dQ]                = calculate_Q_dQ(-a_in);
    H_out.template block<3, 18>(3, 0) = -dQ;
  }

  static void d2r_expinv(TRefIn a_in, THessRefOut H_out)
  {
    H_out.setZero();

    // DERIVATIVES OF SO3 JACOBIAN
    Eigen::Matrix<Scalar, 3, 9> Hso3;
    SO3Impl<Scalar>::d2r_expinv(a_in.template tail<3>(), Hso3);

    for (auto i = 0u; i < 3; ++i) {
      H_out.template block<3, 3>(0, 6 * i + 3)      = Hso3.template block<3, 3>(0, 3 * i);
      H_out.template block<3, 3>(3, 18 + 6 * i + 3) = Hso3.template block<3, 3>(0, 3 * i);
    }

    // DERIVATIVE OF -J Q J TERM
    auto [Q, dQ] = calculate_Q_dQ(-a_in);
    dQ *= -1;  // account for -a_in

    Eigen::Matrix3<Scalar> Jso3;
    SO3Impl<Scalar>::dr_expinv(a_in.template tail<3>(), Jso3);
    // Hso3 contains derivatives w.r.t. w, we extend for derivatives w.r.t. [v, w]
    Eigen::Matrix<Scalar, 3, 18> Hso3_exp = Eigen::Matrix<Scalar, 3, 18>::Zero();
    for (auto i = 0u; i < 3; ++i) {
      Hso3_exp.template middleCols<3>(6 * i + 3) = Hso3.template middleCols<3>(3 * i);
    }

    const Eigen::Matrix3<Scalar> Jtmp       = Jso3 * Q;
    const Eigen::Matrix<Scalar, 3, 18> Htmp = diff::matrix_product(Jso3, Hso3_exp, Q, dQ);
    H_out.template block<3, 18>(3, 0)       = -diff::matrix_product(Jtmp, Htmp, Jso3, Hso3_exp);
  }
};

}  // namespace smooth

#endif  // SMOOTH__INTERNAL__SE3_HPP_

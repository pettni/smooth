// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Core>

#include "common.hpp"
#include "trig.hpp"

namespace smooth {

/**
 * @brief SO(3) Lie group represented as S^3
 *
 * Memory layout
 * -------------
 * Group:    qx qy qz qw  (same as Eigen quaternion)
 * Tangent:  Ωx Ωy Ωz
 *
 * Lie group Matrix form
 * ---------------------
 * 3x3 rotation matrix
 *
 * Lie algebra Matrix form
 * -----------------------
 * [  0 -Ωz  Ωy ]
 * [  Ωz  0 -Ωx ]
 * [ -Ωy Ωx   0 ]
 *
 * Constraints
 * -----------
 * Group:   qx * qx + qy * qy + qz * qz + qw * qw = 1
 * Tangent: -pi < Ωx, Ωy, Ωz <= pi
 */
template<typename _Scalar>
class SO3Impl
{
public:
  using Scalar = _Scalar;

  static constexpr int RepSize        = 4;
  static constexpr int Dim            = 3;
  static constexpr int Dof            = 3;
  static constexpr bool IsCommutative = false;

  SMOOTH_DEFINE_REFS;

  /// @brief Compute sum S1 = \sum \hat w^k / (k+1)!
  static Eigen::Matrix3<Scalar> calc_S1(TRefIn a_in)
  {
    using detail::cos_2, detail::sin_3;

    const Scalar th2 = a_in.squaredNorm();
    Eigen::Matrix3<Scalar> M;
    hat(a_in, M);
    return Eigen::Matrix3<Scalar>::Identity() - cos_2(th2) * M - sin_3(th2) * M * M;
  }

  /// @brief Compute sum S2 = \sum \hat w^k / (k+2)!
  static Eigen::Matrix3<Scalar> calc_S2(TRefIn a_in)
  {
    using detail::cos_4, detail::sin_3;

    const Scalar th2 = a_in.squaredNorm();
    Eigen::Matrix3<Scalar> M;
    hat(a_in, M);
    return Eigen::Matrix3<Scalar>::Identity() / Scalar(2) - sin_3(th2) * M + cos_4(th2) * M * M;
  }

  /**
   * @brief Compute matrix inverse of sum
   * \f[
   *   \sum_{k \geq 0} \frac {\hat \omega}{(k+1)!}.
   * \f]
   */
  static Eigen::Matrix3<Scalar> calc_S1inv(TRefIn a_in)
  {
    using std::sqrt, std::sin, std::cos;
    const Scalar th2 = a_in.squaredNorm();

    const auto A = [&]() -> Scalar {
      if (th2 < Scalar(eps2)) {
        // https://www.wolframalpha.com/input/?i=series+1%2Fx%5E2-%281%2Bcos+x%29%2F%282*x*sin+x%29+at+x%3D0
        return Scalar(1) / Scalar(12) + th2 / Scalar(720);
      } else {
        const Scalar th = sqrt(th2);
        return Scalar(1) / th2 - (Scalar(1) + cos(th)) / (Scalar(2) * th * sin(th));
      }
    }();
    Eigen::Matrix3<Scalar> M;
    hat(a_in, M);

    return Eigen::Matrix3<Scalar>::Identity() - M / Scalar(2) + A * M * M;
  }

  static void setIdentity(GRefOut g_out) { g_out << Scalar(0), Scalar(0), Scalar(0), Scalar(1); }

  static void setRandom(GRefOut g_out)
  {
    g_out = Eigen::Quaternion<Scalar>::UnitRandom().coeffs();
    if (g_out[3] < Scalar(0)) { g_out *= Scalar(-1); }
  }

  static void matrix(GRefIn g_in, MRefOut m_out)
  {
    Eigen::Map<const Eigen::Quaternion<Scalar>> q(g_in.data());
    m_out = q.toRotationMatrix();
  }

  static void composition(GRefIn g_in1, GRefIn g_in2, GRefOut g_out)
  {
    Eigen::Map<const Eigen::Quaternion<Scalar>> q1(g_in1.data());
    Eigen::Map<const Eigen::Quaternion<Scalar>> q2(g_in2.data());
    g_out = (q1 * q2).coeffs();
    if (g_out[3] < Scalar(0)) { g_out *= Scalar(-1); }
  }

  static void inverse(GRefIn g_in, GRefOut g_out)
  {
    Eigen::Map<const Eigen::Quaternion<Scalar>> q(g_in.data());
    g_out = q.inverse().coeffs();
  }

  static void log(GRefIn g_in, TRefOut a_out)
  {
    using std::atan2, std::sqrt;
    const Scalar xyz2 = g_in[0] * g_in[0] + g_in[1] * g_in[1] + g_in[2] * g_in[2];

    const auto phi = [&]() -> Scalar {
      if (xyz2 < Scalar(eps2)) {
        // https://www.wolframalpha.com/input/?i=series+atan%28y%2Fx%29+%2F+y+at+y%3D0
        return Scalar(2) / g_in[3] - Scalar(2) * xyz2 / (Scalar(3) * g_in[3] * g_in[3] * g_in[3]);
      } else {
        const Scalar xyz = sqrt(xyz2);
        return Scalar(2) * atan2(xyz, g_in[3]) / xyz;
      }
    }();

    a_out << g_in[0], g_in[1], g_in[2];
    a_out *= phi;
  }

  static void Ad(GRefIn g_in, TMapRefOut A_out)
  {
    Eigen::Map<const Eigen::Quaternion<Scalar>> q(g_in.data());
    A_out = q.toRotationMatrix();
  }

  static void exp(TRefIn a_in, GRefOut g_out)
  {
    using std::sqrt, std::cos, std::sin;

    const Scalar th2 = a_in.squaredNorm();

    const auto [A, B] = [&]() -> std::array<Scalar, 2> {
      if (th2 < Scalar(eps2)) {
        return {
          // https://www.wolframalpha.com/input/?i=series+sin%28x%2F2%29%2Fx+at+x%3D0
          Scalar(1) / Scalar(2) - th2 / Scalar(48),
          // https://www.wolframalpha.com/input/?i=series+cos%28x%2F2%29+at+x%3D0
          Scalar(1) - th2 / Scalar(8),
        };
      } else {
        const Scalar th = sqrt(th2);
        return {
          sin(th / Scalar(2)) / th,
          cos(th / Scalar(2)),
        };
      }
    }();

    g_out << A * a_in.x(), A * a_in.y(), A * a_in.z(), B;
    if (g_out[3] < Scalar(0)) { g_out *= Scalar(-1); }
  }

  static void hat(TRefIn a_in, MRefOut A_out)
  {
    A_out << Scalar(0), -a_in(2), a_in(1), a_in(2), Scalar(0), -a_in(0), -a_in(1), a_in(0),
      Scalar(0);
  }

  static void vee(MRefIn A_in, TRefOut a_out)
  {
    a_out << (A_in(2, 1) - A_in(1, 2)) / Scalar(2), (A_in(0, 2) - A_in(2, 0)) / Scalar(2),
      (A_in(1, 0) - A_in(0, 1)) / Scalar(2);
  }

  static void ad(TRefIn a_in, TMapRefOut A_out) { hat(a_in, A_out); }

  static void dr_exp(TRefIn a_in, TMapRefOut A_out) { A_out.noalias() = calc_S1(-a_in); }

  static void dr_expinv(TRefIn a_in, TMapRefOut A_out)
  {
    Eigen::Matrix3<Scalar> ad_a;
    ad(a_in, ad_a);
    A_out.noalias() = calc_S1inv(a_in) + ad_a;
  }

  static void d2r_exp(TRefIn a_in, THessRefOut H_out)
  {
    const auto [A, B, dA_over_th, dB_over_th] = [&]() -> std::array<Scalar, 4> {
      using std::sqrt, std::sin, std::cos;

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
    H_out <<
      0., -2 * B * a_in.y(), -2 * B * a_in.z(), B * a_in.y(), B * a_in.x(), -A, B * a_in.z(), A, B * a_in.x(),
      B * a_in.y(), B * a_in.x(), A, -2 * B * a_in.x(), 0., -2 * B * a_in.z(), -A, B * a_in.z(), B * a_in.y(),
      B * a_in.z(), -A, B * a_in.x(), A, B * a_in.z(), B * a_in.y(), -2 * B * a_in.x(), -2 * B * a_in.y(), 0.;
    // clang-format on

    // add -dA * ad + dB * ad^2
    Eigen::Matrix3<Scalar> ad_a;
    ad(a_in, ad_a);
    const Eigen::Matrix3<Scalar> ad_a2 = ad_a * ad_a;

    for (auto i = 0u; i < 3; ++i) {
      const Scalar dA_dxi = dA_over_th * a_in(i);
      const Scalar dB_dxi = dB_over_th * a_in(i);
      for (auto j = 0u; j < 3; ++j) {
        H_out.col(i + 3 * j) -= dA_dxi * ad_a.row(j).transpose();
        H_out.col(i + 3 * j) += dB_dxi * ad_a2.row(j).transpose();
      }
    }
  }

  static void d2r_expinv(TRefIn a_in, THessRefOut H_out)
  {
    const auto [A, dA_over_th] = [&]() -> std::array<Scalar, 2> {
      using std::sqrt, std::cos, std::sin;

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
    H_out <<
      0, -2 * A * a_in.y(), -2 * A * a_in.z(), A * a_in.y(), A * a_in.x(), 0.5, A * a_in.z(), -0.5, A * a_in.x(),
      A * a_in.y(), A * a_in.x(), -0.5, -2 * A * a_in.x(), 0, -2 * A * a_in.z(), 0.5, A * a_in.z(), A * a_in.y(),
      A * a_in.z(), 0.5, A * a_in.x(), -0.5, A * a_in.z(), A * a_in.y(), -2 * A * a_in.x(), -2 * A * a_in.y(), 0;
    // clang-format on

    // add dA * ad^2
    Eigen::Matrix3<Scalar> ad_a;
    ad(a_in, ad_a);
    const Eigen::Matrix3<Scalar> ad_a2 = ad_a * ad_a;
    for (auto i = 0u; i < 3; ++i) {
      const Scalar dA_dxi = dA_over_th * a_in(i);
      for (auto j = 0u; j < 3; ++j) { H_out.col(i + 3 * j) += dA_dxi * ad_a2.row(j).transpose(); }
    }
  }
};

}  // namespace smooth

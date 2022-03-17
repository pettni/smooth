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

#ifndef SMOOTH__INTERNAL__SE2_HPP_
#define SMOOTH__INTERNAL__SE2_HPP_

#include <Eigen/Core>

#include "common.hpp"
#include "so2.hpp"

namespace smooth {

/**
 * @brief SE(2) Lie Group represented as C^1 ⋉ R^2
 *
 * Memory layout
 * -------------
 * Group:    x y qz qw
 * Tangent:  vx vy Ωz
 *
 * Lie group Matrix form
 * ---------------------
 * [ qw -qz x ]
 * [ qz  qw y ]
 * [  0   0 1 ]
 *
 * Lie algebra Matrix form
 * -----------------------
 * [ 0 -Ωz vx ]
 * [ Ωz  0 vy ]
 * [  0  0  0 ]
 *
 * Constraints
 * -----------
 * Group:   qz * qz + qw * qw = 1
 * Tangent: -pi < Ωz <= pi
 */
template<typename _Scalar>
class SE2Impl
{
public:
  using Scalar = _Scalar;

  static constexpr Eigen::Index RepSize = 4;
  static constexpr Eigen::Index Dim     = 3;
  static constexpr Eigen::Index Dof     = 3;

  SMOOTH_DEFINE_REFS;

  static void setIdentity(GRefOut g_out) { g_out << Scalar(0), Scalar(0), Scalar(0), Scalar(1); }

  static void setRandom(GRefOut g_out)
  {
    g_out.template head<2>().setRandom();
    SO2Impl<Scalar>::setRandom(g_out.template tail<2>());
  }

  static void matrix(GRefIn g_in, MRefOut m_out)
  {
    m_out.setIdentity();
    SO2Impl<Scalar>::matrix(g_in.template tail<2>(), m_out.template topLeftCorner<2, 2>());
    m_out.template topRightCorner<2, 1>() = g_in.template head<2>();
  }

  static void composition(GRefIn g_in1, GRefIn g_in2, GRefOut g_out)
  {
    SO2Impl<Scalar>::composition(
      g_in1.template tail<2>(), g_in2.template tail<2>(), g_out.template tail<2>());
    Eigen::Matrix<Scalar, 2, 2> R1;
    SO2Impl<Scalar>::matrix(g_in1.template tail<2>(), R1);
    g_out.template head<2>().noalias() = R1 * g_in2.template head<2>() + g_in1.template head<2>();
  }

  static void inverse(GRefIn g_in, GRefOut g_out)
  {
    Eigen::Matrix<Scalar, 2, 1> so2inv;
    SO2Impl<Scalar>::inverse(g_in.template tail<2>(), so2inv);

    Eigen::Matrix<Scalar, 2, 2> Rinv;
    SO2Impl<Scalar>::matrix(so2inv, Rinv);

    g_out.template head<2>().noalias() = -Rinv * g_in.template head<2>();
    g_out.template tail<2>()           = so2inv;
  }

  static void log(GRefIn g_in, TRefOut a_out)
  {
    using std::tan;

    Eigen::Matrix<Scalar, 1, 1> so2_log;
    SO2Impl<Scalar>::log(g_in.template tail<2>(), so2_log);
    const Scalar th  = so2_log(0);
    const Scalar th2 = th * th;

    const Scalar B = th / Scalar(2);
    const auto A   = [&]() -> Scalar {
      if (th2 < Scalar(eps2)) {
        // https://www.wolframalpha.com/input/?i=series+x+%2F+tan+x+at+x%3D0
        return Scalar(1) - th2 / Scalar(12);
      } else {
        return B / tan(B);
      }
    }();

    const Eigen::Matrix<Scalar, 2, 2> Sinv{{A, B}, {-B, A}};

    a_out.template head<2>().noalias() = Sinv * g_in.template head<2>();
    a_out(2)                           = th;
  }

  static void Ad(GRefIn g_in, TMapRefOut A_out)
  {
    SO2Impl<Scalar>::matrix(g_in.template tail<2>(), A_out.template topLeftCorner<2, 2>());
    A_out(0, 2) = g_in(1);
    A_out(1, 2) = -g_in(0);
    A_out(2, 0) = Scalar(0);
    A_out(2, 1) = Scalar(0);
    A_out(2, 2) = Scalar(1);
  }

  static void exp(TRefIn a_in, GRefOut g_out)
  {
    using std::cos, std::sin;

    const Scalar th  = a_in.z();
    const Scalar th2 = th * th;

    const auto [A, B] = [&]() -> std::array<Scalar, 2> {
      if (th2 < Scalar(eps2)) {
        return {
          // https://www.wolframalpha.com/input/?i=series+sin+x+%2F+x+at+x%3D0
          Scalar(1) - th2 / Scalar(6),
          // https://www.wolframalpha.com/input/?i=series+%28cos+x+-+1%29+%2F+x+at+x%3D0
          -th / Scalar(2) + th * th2 / Scalar(24),
        };
      } else {
        return {
          sin(th) / th,
          (cos(th) - Scalar(1)) / th,
        };
      }
    }();

    const Eigen::Matrix<Scalar, 2, 2> S{{A, B}, {-B, A}};

    g_out.template head<2>().noalias() = S * a_in.template head<2>();
    SO2Impl<Scalar>::exp(a_in.template tail<1>(), g_out.template tail<2>());
  }

  static void hat(TRefIn a_in, MRefOut A_out)
  {
    A_out.setZero();
    SO2Impl<Scalar>::hat(a_in.template tail<1>(), A_out.template topLeftCorner<2, 2>());
    A_out.template topRightCorner<2, 1>() = a_in.template head<2>();
  }

  static void vee(MRefIn A_in, TRefOut a_out)
  {
    SO2Impl<Scalar>::vee(A_in.template topLeftCorner<2, 2>(), a_out.template tail<1>());
    a_out.template head<2>() = A_in.template topRightCorner<2, 1>();
  }

  static void ad(TRefIn a_in, TMapRefOut A_out)
  {
    A_out.setZero();
    SO2Impl<Scalar>::hat(a_in.template tail<1>(), A_out.template topLeftCorner<2, 2>());
    A_out(0, 2) = a_in.y();
    A_out(1, 2) = -a_in.x();
  }

  static void dr_exp(TRefIn a_in, TMapRefOut A_out)
  {
    using std::sin, std::cos;

    const Scalar th2 = a_in.z() * a_in.z();

    const auto [A, B] = [&]() -> std::array<Scalar, 2> {
      if (th2 < Scalar(eps2)) {
        return {
          // https://www.wolframalpha.com/input/?i=series+%281-cos+x%29+%2F+x%5E2+at+x%3D0
          Scalar(1) / Scalar(2) - th2 / Scalar(24),
          // https://www.wolframalpha.com/input/?i=series+%28x+-+sin%28x%29%29+%2F+x%5E3+at+x%3D0
          Scalar(1) / Scalar(6) - th2 / Scalar(120),
        };
      } else {
        const Scalar th = a_in.z();
        return {
          (Scalar(1) - cos(th)) / th2,
          (th - sin(th)) / (th2 * th),
        };
      }
    }();

    Eigen::Matrix3<Scalar> ad_a;
    ad(a_in, ad_a);
    A_out = Eigen::Matrix3<Scalar>::Identity() - A * ad_a + B * ad_a * ad_a;
  }

  static void dr_expinv(TRefIn a_in, TMapRefOut A_out)
  {
    using std::sin, std::cos;

    const Scalar th  = a_in.z();
    const Scalar th2 = th * th;

    const auto A = [&]() -> Scalar {
      if (th2 < Scalar(eps2)) {
        // https://www.wolframalpha.com/input/?i=series+1%2Fx%5E2+-+%281+%2B+cos+x%29+%2F+%282+*+x+*+sin+x%29+at+x%3D0
        return Scalar(1) / Scalar(12) + th2 / Scalar(720);
      } else {
        return (Scalar(1) / th2) - (Scalar(1) + cos(th)) / (Scalar(2) * th * sin(th));
      }
    }();

    Eigen::Matrix3<Scalar> ad_a;
    ad(a_in, ad_a);
    A_out = Eigen::Matrix3<Scalar>::Identity() + ad_a / 2 + A * ad_a * ad_a;
  }

  static void d2r_exp(TRefIn a_in, THessRefOut H_out)
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
    H_out <<
      0, 0, -2*B*a_in.z(), 0, 0, -A, 0, 0, 0,
      0, 0, A, 0, 0, -2*B*a_in.z(), 0, 0, 0 ,
      B*a_in.z(), -A, B*a_in.x(), A, B*a_in.z(), B*a_in.y(), 0, 0, 0;
    // clang-format on

    // add -dA * ad + dB * ad^2
    Eigen::Matrix3<Scalar> ad_a;
    ad(a_in, ad_a);
    const Eigen::Matrix3<Scalar> ad_a2 = ad_a * ad_a;

    for (auto j = 0u; j < 3; ++j) {
      H_out.col(2 + 3 * j) -= dA_dwz * ad_a.row(j).transpose();
      H_out.col(2 + 3 * j) += dB_dwz * ad_a2.row(j).transpose();
    }
  }

  static void d2r_expinv(TRefIn a_in, THessRefOut H_out)
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
    H_out <<
      0, 0, -2*A*a_in.z(), 0, 0, 0.5, 0, 0, 0,
      0, 0, -0.5, 0, 0, -2*A*a_in.z(), 0, 0, 0,
      A*a_in.z(), 0.5, A*a_in.x(), -0.5, A*a_in.z(), A*a_in.y(), 0, 0, 0;
    // clang-format on

    // add -dA * ad + dB * ad^2
    Eigen::Matrix3<Scalar> ad_a;
    ad(a_in, ad_a);
    const Eigen::Matrix3<Scalar> ad_a2 = ad_a * ad_a;

    for (auto j = 0u; j < 3; ++j) { H_out.col(2 + 3 * j) += dA_dwz * ad_a2.row(j).transpose(); }
  }
};

}  // namespace smooth

#endif  // SMOOTH__INTERNAL__SE2_HPP_

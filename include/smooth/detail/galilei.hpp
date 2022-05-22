// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Core>

#include "common.hpp"
#include "smooth/derivatives.hpp"
#include "so3.hpp"

namespace smooth {

/**
 * @brief Galilei Lie Group represented as S^3 ⋉ R3 ⋉ R3 ⋉ R
 *
 * Memory layout
 * -------------
 * Group:    vx vy vz px py pz tau qx qy qz qw
 * Tangent:  bx by bz qx qy qz s Ωx Ωy Ωz
 *
 * Lie group Matrix form
 * ---------------------
 * [ R v p   ]
 * [ 0 1 tau ]
 * [ 0 0 1   ]
 *
 * where R ∈ SO(3), v, p ∈ R3, and t ∈ R
 *
 * Lie algebra Matrix form
 * -----------------------
 * [  0 -Ωz  Ωy bx qx ]
 * [  Ωz  0 -Ωx by qy ]
 * [ -Ωy Ωx   0 bz qz ]
 * [   0  0   0  0  s ]
 * [   0  0   0  0  1 ]
 *
 * Constraints
 * -----------
 * Group:   qx * qx + qy * qy + qz * qz + qw * qw = 1
 * Tangent: -pi < Ωx Ωy Ωz <= pi
 */
template<typename _Scalar>
class GalileiImpl
{
public:
  using Scalar = _Scalar;

  static constexpr int RepSize        = 11;
  static constexpr int Dim            = 5;
  static constexpr int Dof            = 10;
  static constexpr bool IsCommutative = false;

  SMOOTH_DEFINE_REFS;

  static void setIdentity(GRefOut g_out)
  {
    g_out.template head<7>().setZero();
    SO3Impl<Scalar>::setIdentity(g_out.template tail<4>());
  }

  static void setRandom(GRefOut g_out)
  {
    g_out.template segment<7>(0).setRandom();
    SO3Impl<Scalar>::setRandom(g_out.template tail<4>());
  }

  static void matrix(GRefIn g_in, MRefOut m_out)
  {
    m_out.setIdentity();
    SO3Impl<Scalar>::matrix(g_in.template tail<4>(), m_out.template topLeftCorner<3, 3>());
    m_out.template block<3, 1>(0, 3) = g_in.template segment<3>(0);
    m_out.template block<4, 1>(0, 4) = g_in.template segment<4>(3);
  }

  static void composition(GRefIn g_in1, GRefIn g_in2, GRefOut g_out)
  {
    SO3Impl<Scalar>::composition(
      g_in1.template tail<4>(), g_in2.template tail<4>(), g_out.template tail<4>());
    Eigen::Matrix<Scalar, 3, 3> R1;
    SO3Impl<Scalar>::matrix(g_in1.template tail<4>(), R1);
    g_out.template segment<3>(0).noalias() =
      R1 * g_in2.template segment<3>(0) + g_in1.template segment<3>(0);
    g_out.template segment<3>(3).noalias() = R1 * g_in2.template segment<3>(3)
                                           + g_in1.template segment<3>(0) * g_in2(6)
                                           + g_in1.template segment<3>(3);
    g_out(6) = g_in1(6) + g_in2(6);
  }

  static void inverse(GRefIn g_in, GRefOut g_out)
  {
    Eigen::Matrix<Scalar, 4, 1> so3inv;
    SO3Impl<Scalar>::inverse(g_in.template tail<4>(), so3inv);

    Eigen::Matrix<Scalar, 3, 3> Rinv;
    SO3Impl<Scalar>::matrix(so3inv, Rinv);

    g_out.template segment<3>(0).noalias() = -Rinv * g_in.template segment<3>(0);
    g_out.template segment<3>(3).noalias() =
      Rinv * (-g_in.template segment<3>(3) + g_in(6) * g_in.template segment<3>(0));
    g_out(6)                 = -g_in(6);
    g_out.template tail<4>() = so3inv;
  }

  static void log(GRefIn g_in, TRefOut a_out)
  {
    SO3Impl<Scalar>::log(g_in.template tail<4>(), a_out.template tail<3>());

    const Eigen::Matrix3<Scalar> S1inv = SO3Impl<Scalar>::calc_S1inv(a_out.template tail<3>());
    const Eigen::Matrix3<Scalar> S2    = SO3Impl<Scalar>::calc_S2(a_out.template tail<3>());

    a_out.template segment<3>(0).noalias() = S1inv * g_in.template segment<3>(0);
    a_out.template segment<3>(3).noalias() =
      S1inv * (g_in.template segment<3>(3) - S2 * a_out.template segment<3>(0) * g_in(6));
    a_out(6) = g_in(6);
  }

  static void Ad(GRefIn g_in, TMapRefOut A_out)
  {
    Eigen::Matrix<Scalar, 3, 3> R;
    SO3Impl<Scalar>::matrix(g_in.template tail<4>(), R);

    const Eigen::Vector3<Scalar> v = g_in.template segment<3>(0);
    const Eigen::Vector3<Scalar> p = g_in.template segment<3>(3);
    const double t                 = g_in(6);

    A_out.setZero();

    // block row 1 (b)
    A_out.template block<3, 3>(0, 0) = R;
    SO3Impl<Scalar>::hat(v, A_out.template block<3, 3>(0, 7));
    A_out.template block<3, 3>(0, 7) *= R;

    // block row 2 (q)
    A_out.template block<3, 3>(3, 0) = -R * t;
    A_out.template block<3, 3>(3, 3) = R;
    A_out.template block<3, 1>(3, 6) = v;
    SO3Impl<Scalar>::hat(p - v * t, A_out.template block<3, 3>(3, 7));
    A_out.template block<3, 3>(3, 7) *= R;

    // block row 3 (s)
    A_out(6, 6) = Scalar(1);

    // block row 4 (omega)
    A_out.template block<3, 3>(7, 7) = R;
  }

  static void exp(TRefIn a_in, GRefOut g_out)
  {
    using std::cos, std::sin;

    SO3Impl<Scalar>::exp(a_in.template tail<3>(), g_out.template tail<4>());

    const Eigen::Matrix3<Scalar> S1 = SO3Impl<Scalar>::calc_S1(a_in.template tail<3>());
    const Eigen::Matrix3<Scalar> S2 = SO3Impl<Scalar>::calc_S2(a_in.template tail<3>());

    g_out.template segment<3>(0).noalias() = S1 * a_in.template segment<3>(0);
    g_out.template segment<3>(3).noalias() =
      S1 * a_in.template segment<3>(3) + S2 * a_in.template segment<3>(0) * a_in(6);
    g_out(6) = a_in(6);
  }

  static void hat(TRefIn a_in, MRefOut A_out)
  {
    A_out.setZero();
    SO3Impl<Scalar>::hat(a_in.template tail<3>(), A_out.template topLeftCorner<3, 3>());
    A_out.template block<3, 1>(0, 3) = a_in.template segment<3>(0);
    A_out.template block<4, 1>(0, 4) = a_in.template segment<4>(3);
  }

  static void vee(MRefIn A_in, TRefOut a_out)
  {
    SO3Impl<Scalar>::vee(A_in.template topLeftCorner<3, 3>(), a_out.template tail<3>());
    a_out.template segment<3>(0) = A_in.template block<3, 1>(0, 3);
    a_out.template segment<4>(3) = A_in.template block<4, 1>(0, 4);
  }

  static void ad(TRefIn a_in, TMapRefOut A_out)
  {
    // [ W  0 0 B ]
    // [ sI W b Q ]
    // [ 0  0 0 0 ]
    // [ 0  0 0 W ]

    Eigen::Ref<const Eigen::Vector3<Scalar>> b = a_in.template segment<3>(0);
    Eigen::Ref<const Eigen::Vector3<Scalar>> q = a_in.template segment<3>(3);
    const Scalar s                             = a_in(6);
    Eigen::Ref<const Eigen::Vector3<Scalar>> w = a_in.template segment<3>(7);

    Eigen::Matrix3<Scalar> B, Q, W;
    SO3Impl<Scalar>::hat(b, B);
    SO3Impl<Scalar>::hat(q, Q);
    SO3Impl<Scalar>::hat(w, W);

    A_out.setZero();

    A_out.template block<3, 3>(0, 0) = W;
    A_out.template block<3, 3>(0, 7) = B;

    A_out.template block<3, 3>(3, 0) = -s * Eigen::Matrix3<Scalar>::Identity();
    A_out.template block<3, 3>(3, 3) = W;
    A_out.template block<3, 1>(3, 6) = b;
    A_out.template block<3, 3>(3, 7) = Q;

    A_out.template block<3, 3>(7, 7) = W;
  }

  static void dr_exp(TRefIn, TMapRefOut)
  {
    throw std::runtime_error("Galilei jacobians not implemented");
  }

  static void dr_expinv(TRefIn, TMapRefOut)
  {
    throw std::runtime_error("Galilei jacobians not implemented");
  }
};

}  // namespace smooth

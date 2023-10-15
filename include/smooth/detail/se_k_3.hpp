// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Core>

#include "../derivatives.hpp"
#include "common.hpp"
#include "so3.hpp"

SMOOTH_BEGIN_NAMESPACE

/**
 * @brief SE_k(3) Lie Group represented as S^3 ⋉ R3 ... ⋉ R3
 *
 * Memory layout
 * -------------
 * Group:    p1 ... pk qx qy qz qw
 * Tangent:  v1 ... vk Ωx Ωy Ωz
 *
 * Lie group Matrix form
 * ---------------------
 * [ R P1 P2 ... Pk ]
 * [ 0 1   0 ...  0 ]
 * [       ...      ]
 * [ 0 0   0 ...  1 ]
 *
 * where R ∈ SO(3) and Pk = [xk yk zk] ∈ R3
 *
 * Lie algebra Matrix form
 * -----------------------
 * [  0 -Ωz  Ωy vx1 vx2 ... vxk]
 * [  Ωz  0 -Ωx vy1 vy2 ... vyk]
 * [ -Ωy Ωx   0 vz1 vz2 ... vzk]
 * [   0  0   0  0   0  ...   0]
 * [             ...           ]
 * [   0  0   0  0   0  ...   0]
 *
 * Constraints
 * -----------
 * Group:   qx * qx + qy * qy + qz * qz + qw * qw = 1
 * Tangent: -pi < Ωx, Ωy, Ωz <= pi
 */
template<typename _Scalar, int _K>
  requires(_K >= 1)
class SE_K_3Impl
{
public:
  using Scalar = _Scalar;

  static constexpr int K = _K;

  static constexpr int RepSize        = 4 + 3 * K;
  static constexpr int Dim            = 3 + K;
  static constexpr int Dof            = 3 + 3 * K;
  static constexpr bool IsCommutative = false;

  SMOOTH_DEFINE_REFS;

  static void setIdentity(GRefOut g_out)
  {
    g_out.setZero();
    g_out(RepSize - 1) = Scalar(1);
  }

  static void setRandom(GRefOut g_out)
  {
    g_out.setRandom();
    SO3Impl<Scalar>::setRandom(g_out.template tail<4>());
  }

  static void matrix(GRefIn g_in, MRefOut m_out)
  {
    m_out.setIdentity();

    SO3Impl<Scalar>::matrix(g_in.template tail<4>(), m_out.template topLeftCorner<3, 3>());
    for (auto i = 0u; i < K; ++i) { m_out.template block<3, 1>(0, 3 + i) = g_in.template segment<3>(3 * i); }
  }

  static void composition(GRefIn g_in1, GRefIn g_in2, GRefOut g_out)
  {
    SO3Impl<Scalar>::composition(g_in1.template tail<4>(), g_in2.template tail<4>(), g_out.template tail<4>());
    Eigen::Matrix<Scalar, 3, 3> R1;
    SO3Impl<Scalar>::matrix(g_in1.template tail<4>(), R1);
    for (auto i = 0u; i < K; ++i) {
      g_out.template segment<3>(3 * i).noalias() =
        R1 * g_in2.template segment<3>(3 * i) + g_in1.template segment<3>(3 * i);
    }
  }

  static void inverse(GRefIn g_in, GRefOut g_out)
  {
    Eigen::Matrix<Scalar, 4, 1> so3inv;
    SO3Impl<Scalar>::inverse(g_in.template tail<4>(), so3inv);

    Eigen::Matrix<Scalar, 3, 3> Rinv;
    SO3Impl<Scalar>::matrix(so3inv, Rinv);

    for (auto i = 0u; i < K; ++i) {
      g_out.template segment<3>(3 * i).noalias() = -Rinv * g_in.template segment<3>(3 * i);
    }
    g_out.template tail<4>() = so3inv;
  }

  static void log(GRefIn g_in, TRefOut a_out)
  {
    using SO3TangentMap = Eigen::Matrix<Scalar, 3, 3>;

    SO3Impl<Scalar>::log(g_in.template tail<4>(), a_out.template tail<3>());

    SO3TangentMap M_dr_expinv, M_ad;
    SO3Impl<Scalar>::dr_expinv(a_out.template tail<3>(), M_dr_expinv);
    SO3Impl<Scalar>::ad(a_out.template tail<3>(), M_ad);
    for (auto i = 0u; i < K; ++i) {
      a_out.template segment<3>(3 * i).noalias() = (-M_ad + M_dr_expinv) * g_in.template segment<3>(3 * i);
    }
  }

  static void Ad(GRefIn g_in, TMapRefOut A_out)
  {
    A_out.setZero();

    SO3Impl<Scalar>::matrix(g_in.template tail<4>(), A_out.template topLeftCorner<3, 3>());

    for (auto i = 0u; i < K; ++i) {
      SO3Impl<Scalar>::hat(g_in.template segment<3>(3 * i), A_out.template block<3, 3>(3 * i, 3 * K));
      A_out.template block<3, 3>(3 * i, 3 * K) *= A_out.template topLeftCorner<3, 3>();
      A_out.template block<3, 3>(3 + 3 * i, 3 + 3 * i) = A_out.template topLeftCorner<3, 3>();
    }
  }

  static void exp(TRefIn a_in, GRefOut g_out)
  {
    using SO3TangentMap = Eigen::Matrix<Scalar, 3, 3>;

    SO3Impl<Scalar>::exp(a_in.template tail<3>(), g_out.template tail<4>());

    SO3TangentMap M_dr_exp, M_Ad;
    SO3Impl<Scalar>::dr_exp(a_in.template tail<3>(), M_dr_exp);
    SO3Impl<Scalar>::Ad(g_out.template tail<4>(), M_Ad);
    SO3TangentMap M_prod = M_Ad * M_dr_exp;

    for (auto i = 0u; i < K; ++i) {
      g_out.template segment<3>(3 * i).noalias() = M_prod * a_in.template segment<3>(3 * i);
    }
  }

  static void hat(TRefIn a_in, MRefOut A_out)
  {
    A_out.setZero();
    SO3Impl<Scalar>::hat(a_in.template tail<3>(), A_out.template topLeftCorner<3, 3>());

    for (auto i = 0u; i < K; ++i) { A_out.template block<3, 1>(0, 3 + i) = a_in.template segment<3>(3 * i); }
  }

  static void vee(MRefIn A_in, TRefOut a_out)
  {
    SO3Impl<Scalar>::vee(A_in.template topLeftCorner<3, 3>(), a_out.template tail<3>());
    for (auto i = 0u; i < K; ++i) { a_out.template segment<3>(3 * i) = A_in.template block<3, 1>(0, 3 + i); }
  }

  static void ad(TRefIn a_in, TMapRefOut A_out)
  {
    A_out.setZero();

    SO3Impl<Scalar>::hat(a_in.template tail<3>(), A_out.template topLeftCorner<3, 3>());

    for (auto i = 0u; i < K; ++i) {
      SO3Impl<Scalar>::hat(a_in.template segment<3>(3 * i), A_out.template block<3, 3>(3 * i, 3 * K));
      A_out.template block<3, 3>(3 + 3 * i, 3 + 3 * i) = A_out.template topLeftCorner<3, 3>();
    }
  }

  static Eigen::Matrix<Scalar, 3, 3>
  calculate_q(Eigen::Ref<const Eigen::Vector3<Scalar>> v, Eigen::Ref<const Eigen::Vector3<Scalar>> w)
  {
    using detail::sin_3, detail::cos_4, detail::sin_5;
    const Scalar th2 = w.squaredNorm();

    Eigen::Matrix<Scalar, 3, 3> V, W;
    SO3Impl<Scalar>::hat(v, V);
    SO3Impl<Scalar>::hat(w, W);

    const Scalar vdw                     = v.dot(w);
    const Eigen::Matrix<Scalar, 3, 3> WV = W * V, VW = V * W, WW = W * W;

    // clang-format off
    return Scalar(0.5) * V
      + sin_3(th2) * (-WV - VW + vdw * W)
      + cos_4(th2) * (W * WV + VW * W + vdw * (Scalar(3) * W - WW))
      + sin_5(th2) * Scalar(3) * vdw * WW;
    // clang-format on
  }

  static void dr_exp(TRefIn a_in, TMapRefOut A_out)
  {
    A_out.setZero();

    SO3Impl<Scalar>::dr_exp(a_in.template tail<3>(), A_out.template topLeftCorner<3, 3>());
    for (auto i = 0u; i < K; ++i) {
      A_out.template block<3, 3>(3 * i, 3 * K) =
        calculate_q(-a_in.template segment<3>(3 * i), -a_in.template tail<3>());
      A_out.template block<3, 3>(3 + 3 * i, 3 + 3 * i) = A_out.template topLeftCorner<3, 3>();
    }
  }

  static void dr_expinv(TRefIn a_in, TMapRefOut A_out)
  {
    A_out.setZero();

    SO3Impl<Scalar>::dr_expinv(a_in.template tail<3>(), A_out.template topLeftCorner<3, 3>());
    for (auto i = 0u; i < K; ++i) {
      A_out.template block<3, 3>(3 * i, 3 * K).noalias() =
        -A_out.template topLeftCorner<3, 3>() * calculate_q(-a_in.template segment<3>(3 * i), -a_in.template tail<3>())
        * A_out.template topLeftCorner<3, 3>();
      A_out.template block<3, 3>(3 + 3 * i, 3 + 3 * i) = A_out.template topLeftCorner<3, 3>();
    }
  }
};

SMOOTH_END_NAMESPACE

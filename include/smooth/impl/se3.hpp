#ifndef SMOOTH__IMPL__SE3_HPP_
#define SMOOTH__IMPL__SE3_HPP_

#include <Eigen/Core>

#include "common.hpp"
#include "so3.hpp"

namespace smooth {

/**
 * @brief SE3 Lie Group represented as S3 ⋉ R3
 *
 * Memory layout
 * =============
 * Group:    x y z qx qy qz qw
 * Tangent:  vx vy vz Ωx Ωy Ωz
 *
 * Lie group Matrix form
 * =====================
 *
 * [ R T ]
 * [ 0 1 ]
 *
 * where R ∈ SO(3) and T ∈ R3
 *
 * Lie algebra Matrix form
 * =======================
 *
 * [  0 -Ωz  Ωy vx]
 * [  Ωz  0 -Ωx vy]
 * [ -Ωy Ωx   0 vz]
 * [   0  0   0  1]
 *
 * Constraints
 * ===========
 * Group:   qx * qx + qy * qy + qz * qz + qw * qw = 1
 * Tangent: -pi < Ωx Ωy Ωz <= pi, 0 <= Ωw <= pi
 */
template<typename _Scalar>
class SE3Impl
{
public:
  using Scalar = _Scalar;

  static constexpr Eigen::Index RepSize = 7;
  static constexpr Eigen::Index Dim     = 4;
  static constexpr Eigen::Index Dof     = 6;

  SMOOTH_DEFINE_REFS

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
    g_out.template head<3>() = R1 * g_in2.template head<3>() + g_in1.template head<3>();
  }

  static void inverse(GRefIn g_in, GRefOut g_out)
  {
    Eigen::Matrix<Scalar, 4, 1> so3inv;
    SO3Impl<Scalar>::inverse(g_in.template tail<4>(), so3inv);

    Eigen::Matrix<Scalar, 3, 3> Rinv;
    SO3Impl<Scalar>::matrix(so3inv, Rinv);

    g_out.template head<3>() = -Rinv * g_in.template head<3>();
    g_out.template tail<4>() = so3inv;
  }

  static void log(GRefIn g_in, TRefOut a_out)
  {
    using SO3TangentMap = Eigen::Matrix<Scalar, 3, 3>;

    SO3Impl<Scalar>::log(g_in.template tail<4>(), a_out.template tail<3>());

    SO3TangentMap M_dr_expinv, M_ad;
    SO3Impl<Scalar>::dr_expinv(a_out.template tail<3>(), M_dr_expinv);
    SO3Impl<Scalar>::ad(a_out.template tail<3>(), M_ad);
    a_out.template head<3>() = (-M_ad + M_dr_expinv) * g_in.template head<3>();
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

    g_out.template head<3>() = M_Ad * M_dr_exp * a_in.template head<3>();
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

    Scalar A, B, C;
    if (th2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+%28x+-+sin+x%29+%2F+x%5E3+at+x%3D0
      A = Scalar(1) / Scalar(6) - th2 / Scalar(120);
      // https://www.wolframalpha.com/input/?i=series+%28cos+x+-+1+%2B+x%5E2%2F2%29+%2F+x%5E4+at+x%3D0
      B = Scalar(1) / Scalar(24) - th2 / Scalar(720);
      // https://www.wolframalpha.com/input/?i=series+%28x+-+sin+x+-+x%5E3%2F6%29+%2F+x%5E5+at+x%3D0
      C = -Scalar(1) / Scalar(120) + th2 / Scalar(5040);
    } else {
      const Scalar th = sqrt(th2), th_4 = th2 * th2, cTh = cos(th), sTh = sin(th);
      A = (th - sTh) / (th * th2);
      B = (cTh - Scalar(1) + th2 / Scalar(2)) / th_4;
      C = (th - sTh - th * th2 / Scalar(6)) / (th_4 * th);
    }

    Eigen::Matrix<Scalar, 3, 3> V, W;
    SO3Impl<Scalar>::hat(a.template head<3>(), V);
    SO3Impl<Scalar>::hat(a.template tail<3>(), W);

    const Scalar vdw                     = a.template tail<3>().dot(a.template head<3>());
    const Eigen::Matrix<Scalar, 3, 3> WV = W * V, VW = V * W, WW = W * W;

    return Scalar(0.5) * V + A * (WV + VW - vdw * W)
         + B * (W * WV + VW * W + vdw * (Scalar(3) * W - WW)) - C * Scalar(3) * vdw * WW;
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
    A_out.template topRightCorner<3, 3>() = -A_out.template topLeftCorner<3, 3>()
                                          * calculate_q(-a_in)
                                          * A_out.template topLeftCorner<3, 3>();
    A_out.template bottomRightCorner<3, 3>() = A_out.template topLeftCorner<3, 3>();
    A_out.template bottomLeftCorner<3, 3>().setZero();
  }
};

}  // namespace smooth

#endif  // SMOOTH__IMPL__SE3_HPP_

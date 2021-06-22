#ifndef SE2IMPL_HPP_
#define SE2IMPL_HPP_

#include <Eigen/Core>

#include "so2.hpp"
#include "common.hpp"

namespace smooth {

/**
 * @brief SE2 Lie Group represented as U(1) ⋉ R2
 *
 * Memory layout
 * =============
 * Group:    x y qz qw
 * Tangent:  vx vy wz
 *
 * Constraints
 * ===========
 * Group:   qz * qz + qw * qw = 1
 * Tangent: -pi < wz <= pi
 */
template<typename _Scalar>
class SE2Impl
{
public:
  using Scalar = _Scalar;

  static constexpr Eigen::Index RepSize = 4;
  static constexpr Eigen::Index Dim     = 4;
  static constexpr Eigen::Index Dof     = 3;

  SMOOTH_DEFINE_REFS

  static void setIdentity(GRefOut g_out) {
    g_out << Scalar(0), Scalar(0), Scalar(0), Scalar(1);
  }

  static void setRandom(GRefOut g_out)
  {
    g_out.template head<2>().setRandom();
    SO2Impl<Scalar>::setRandom(g_out.template tail<2>());
  }

  static void matrix(GRefIn g_in, MRefOut m_out) {
    m_out.setIdentity();
    SO2Impl<Scalar>::matrix(
      g_in.template tail<2>(),
      m_out.template topLeftCorner<2, 2>()
    );
    m_out.template topRightCorner<2, 1>() = g_in.template head<2>();
  }

  static void composition(GRefIn g_in1, GRefIn g_in2, GRefOut g_out)
  {
    SO2Impl<Scalar>::composition(
      g_in1.template tail<2>(),
      g_in2.template tail<2>(),
      g_out.template tail<2>()
    );
    Eigen::Matrix<Scalar, 2, 2> R1;
    SO2Impl<Scalar>::matrix(
      g_in1.template tail<2>(),
      R1
    );
    g_out.template head<2>() = R1 * g_in2.template head<2>() + g_in1.template head<2>();
  }

  static void inverse(GRefIn g_in, GRefOut g_out) {
    Eigen::Matrix<Scalar, 2, 1> so2inv;
    SO2Impl<Scalar>::inverse(g_in.template tail<2>(), so2inv);

    Eigen::Matrix<Scalar, 2, 2> Rinv;
    SO2Impl<Scalar>::matrix(so2inv, Rinv);

    g_out.template head<2>() = -Rinv * g_in.template head<2>();
    g_out.template tail<2>() = so2inv;
  }

  static void log(GRefIn g_in, TRefOut a_out)
  {
    using std::tan;

    Eigen::Matrix<Scalar, 1, 1> so2_log;
    SO2Impl<Scalar>::log(g_in.template tail<2>(), so2_log);
    const Scalar th = so2_log(0);
    const Scalar th2 = th * th;

    const Scalar B = th / Scalar(2);
    Scalar A;
    if (th2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+x+%2F+tan+x+at+x%3D0
      A = Scalar(1) - th2 / Scalar(12);
    } else {
      A = B / tan(B);
    }

    Eigen::Matrix<Scalar, 2, 2> Sinv;
    Sinv(0, 0) = A;
    Sinv(1, 1) = A;
    Sinv(0, 1) = B;
    Sinv(1, 0) = -B;

    a_out.template head<2>() = Sinv * g_in.template head<2>();
    a_out(2) = th;
  }

  static void Ad(GRefIn g_in, TMapRefOut A_out) {

    SO2Impl<Scalar>::matrix(g_in.template tail<2>(), A_out.template topLeftCorner<2, 2>());
    A_out(0, 2) = g_in(1);
    A_out(1, 2) = -g_in(0);
    A_out.template bottomRows<1>().setZero();
  }

  static void exp(TRefIn a_in, GRefOut g_out)
  {
    using std::cos, std::sin;

    const Scalar th = a_in.z();
    const Scalar th2 = th * th;

    Scalar A, B;
    if (th2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+sin+x+%2F+x+at+x%3D0
      A = Scalar(1) - th2 / Scalar(6);
      // https://www.wolframalpha.com/input/?i=series+%28cos+x+-+1%29+%2F+x+at+x%3D0
      B = -th / Scalar(2) + th * th2 / Scalar(24);
    } else {
      A = sin(th) / th;
      B = (cos(th) - Scalar(1)) / th;
    }

    Eigen::Matrix<Scalar, 2, 2> S;
    S(0, 0) = A;
    S(1, 1) = A;
    S(0, 1) = B;
    S(1, 0) = -B;

    g_out.template head<2>() = S * a_in.template head<2>();
    SO2Impl<Scalar>::exp(a_in.template tail<1>(), g_out.template tail<2>());
  }

  static void hat(TRefIn a_in, MRefOut A_out) {
    A_out.setZero();
    SO2Impl<Scalar>::hat(a_in.template tail<1>(), A_out.template topLeftCorner<2, 2>());
    A_out.template topRightCorner<2, 1>() = a_in.template head<2>();
  }

  static void vee(MRefIn A_in, TRefOut a_out) {
    SO2Impl<Scalar>::vee(A_in.template topLeftCorner<2, 2>(), a_out.template tail<1>());
    a_out.template head<2>() = A_in.template topRightCorner<2, 1>();
  }

  static void ad(TRefIn a_in, TMapRefOut A_out) {
    A_out.setZero();
    SO2Impl<Scalar>::hat(a_in.template tail<1>(), A_out.template topLeftCorner<2, 2>());
    A_out(0, 2) = a_in.y();
    A_out(1, 2) = -a_in.x();
  }

  static void dr_exp(TRefIn a_in, TMapRefOut A_out) {
    using TangentMap = Eigen::Matrix<Scalar, 3, 3>;
    using std::sin, std::cos;

    const Scalar th2 = a_in.z() * a_in.z();

    Scalar A, B;
    if (th2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+%281-cos+x%29+%2F+x%5E2+at+x%3D0
      A = Scalar(1) / Scalar(2) - th2 / Scalar(24);
      // https://www.wolframalpha.com/input/?i=series+%28x+-+sin%28x%29%29+%2F+x%5E3+at+x%3D0
      B = Scalar(1) / Scalar(6) - th2 / Scalar(120);
    } else {
      const Scalar th = a_in.z();
      A = (Scalar(1) - cos(th)) / th2;
      B = (th - sin(th)) / (th2 * th);
    }

    TangentMap ad_a;
    ad(a_in, ad_a);
    A_out = TangentMap::Identity() - A * ad_a + B * ad_a * ad_a;
  }

  static void dr_expinv(TRefIn a_in, TMapRefOut A_out) {
    using TangentMap = Eigen::Matrix<Scalar, 3, 3>;
    using std::sin, std::cos;

    const Scalar th = a_in.z();
    const Scalar th2 = th * th;

    Scalar A;
    if (th2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+1%2Fx%5E2+-+%281+%2B+cos+x%29+%2F+%282+*+x+*+sin+x%29+at+x%3D0
      A = Scalar(1) / Scalar(12) + th2 / Scalar(720);
    } else {
      A = (Scalar(1) / th2) - (Scalar(1) + cos(th)) / (Scalar(2) * th * sin(th));
    }

    TangentMap ad_a;
    ad(a_in, ad_a);
    A_out = TangentMap::Identity() + ad_a / 2 + A * ad_a * ad_a;
  }
};

}  // namespace smooth

#endif  // SE2IMPL_HPP_

#ifndef SO3IMPL_HPP_
#define SO3IMPL_HPP_

#include <Eigen/Core>

#include "common.hpp"

namespace smooth {

template<typename _Scalar>
class SO3Impl
{
public:
  using Scalar = _Scalar;

  static constexpr Eigen::Index RepSize = 4;
  static constexpr Eigen::Index Dim     = 3;
  static constexpr Eigen::Index Dof     = 3;

  SMOOTH_DEFINE_REFS

  static void setIdentity(GRefOut g_out) { g_out << Scalar(0), Scalar(0), Scalar(0), Scalar(1); }

  static void setRandom(GRefOut g_out)
  {
    g_out = Eigen::Quaternion<Scalar>::UnitRandom().coeffs();
    if (g_out[3] < 0) { g_out *= Scalar(-1); }
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

    Scalar phi;
    if (xyz2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+atan%28y%2Fx%29+%2F+y+at+y%3D0
      phi = Scalar(2) / g_in[3] - Scalar(2) * xyz2 / (Scalar(3) * g_in[3] * g_in[3] * g_in[3]);
    } else {
      Scalar xyz = sqrt(xyz2);
      phi        = Scalar(2) * atan2(xyz, g_in[3]) / xyz;
    }

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

    Scalar A, B;
    if (th2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+sin%28x%2F2%29%2Fx+at+x%3D0
      A = Scalar(1) / Scalar(2) - th2 / Scalar(48);
      // https://www.wolframalpha.com/input/?i=series+cos%28x%2F2%29+at+x%3D0
      B = Scalar(1) - th2 / Scalar(8);
    } else {
      const Scalar th = sqrt(th2);
      A               = sin(th / Scalar(2)) / th;
      B               = cos(th / Scalar(2));
    }

    g_out << A * a_in.x(), A * a_in.y(), A * a_in.z(), B;
  }

  static void hat(TRefIn a_in, MRefOut A_out)
  {
    A_out << Scalar(0), -a_in(2), a_in(1),
          a_in(2), Scalar(0), -a_in(0),
          -a_in(1), a_in(0), Scalar(0);
  }

  static void vee(MRefIn A_in, TRefOut a_out)
  {
    a_out << (A_in(2, 1) - A_in(1, 2)) / Scalar(2),
          (A_in(0, 2) - A_in(2, 0)) / Scalar(2),
          (A_in(1, 0) - A_in(0, 1)) / Scalar(2);
  }

  static void ad(TRefIn a_in, TMapRefOut A_out)
  {
    A_out.setZero();
  }

  static void dr_exp(TRefIn a_in, TMapRefOut A_out)
  {
    A_out.setZero();
  }

  static void dr_expinv(TRefIn a_in, TMapRefOut A_out)
  {
    A_out.setZero();
  }
};

}  // namespace smooth

#endif  // SO3IMPL_HPP_

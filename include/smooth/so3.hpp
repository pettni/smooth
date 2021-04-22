#ifndef SMOOTH__SO3_HPP_
#define SMOOTH__SO3_HPP_

#include <Eigen/Geometry>
#include <random>

#include "common.hpp"
#include "concepts.hpp"
#include "lie_group_base.hpp"
#include "macro.hpp"

namespace smooth
{

/**
 * @brief SO3 Lie Group
 *
 * Memory layout
 * =============
 * Group:    qx qy qz qw  (same as Eigen quaternion)
 * Tangent:  wx wy wz
 *
 * Constraints
 * ===========
 * Group:   qx * qx + qy * qy + qz * qz + qw * qw = 1
 * Tangent: -pi < wx, wy, wz <= pi
 */
template<typename _Scalar, StorageLike _Storage = DefaultStorage<_Scalar, 4>>
requires(_Storage::SizeAtCompileTime == 4 && std::is_same_v<typename _Storage::Scalar, _Scalar>)
class SO3 : public LieGroupBase<SO3<_Scalar, _Storage>, 4>
{
private:
  _Storage s_;

public:
  // REQUIRED CONSTANTS

  static constexpr uint32_t lie_size = 4;
  static constexpr uint32_t lie_dof = 3;
  static constexpr uint32_t lie_dim = 3;
  static constexpr uint32_t lie_actdim = 3;

  // CONSTRUCTORS AND OPERATORS

  SMOOTH_GROUP_BOILERPLATE(SO3)

  // SO3-SPECIFIC API

  /**
   * @brief Construct from quaternion
   */
  template<typename Derived>
  SO3(const Eigen::QuaternionBase<Derived> & qin)
  requires ModifiableStorageLike<Storage>&& std::is_same_v<typename Derived::Scalar, Scalar>
  {
    s_[0] = qin.x(); s_[1] = qin.y(); s_[2] = qin.z(); s_[3] = qin.w();
    normalize();
  }

  /**
   * @brief Access as Eigen quaternion by Map
   *
   * Warning: if the quaternion is modified through this function
   * the user must ensure the new value is a unit quaternion.
   */
  Eigen::Map<Eigen::Quaternion<Scalar>> quat()
  requires ModifiableStorageLike<Storage>
  {
    return Eigen::Map<Eigen::Quaternion<Scalar>>(s_.data());
  }

  /**
   * @brief Access as Eigen quaternion by const Map
   */
  Eigen::Map<const Eigen::Quaternion<Scalar>> quat() const
  requires MappableStorageLike<Storage>
  {
    return Eigen::Map<const Eigen::Quaternion<Scalar>>(s_.data());
  }

  /**
   * @brief Return a quaternion by copy for constant storage
   */
  Eigen::Quaternion<Scalar> quat() const
  requires(!MappableStorageLike<Storage>)
  {
    return Eigen::Quaternion<Scalar>(s_[3], s_[0], s_[1], s_[2]);
  }

private:
  /**
   * @brief Construct from coefficients (does not normalize)
   */
  template<typename Scalar>
  explicit SO3(const Scalar & qx, const Scalar & qy, const Scalar & qz, const Scalar & qw)
  {
    s_[0] = qx; s_[1] = qy; s_[2] = qz, s_[3] = qw;
  }

  /**
   * @brief Normalize quaternion and set qw >= 0
   */
  void normalize()
  requires ModifiableStorageLike<Storage>
  {
    using std::abs, std::sqrt;

    Scalar mul = Scalar(1) /
      sqrt(s_[0] * s_[0] + s_[1] * s_[1] + s_[2] * s_[2] + s_[3] * s_[3]);
    if (s_[3] < Scalar(0)) {
      mul *= Scalar(-1);
    }
    if (abs(mul - Scalar(1)) > Scalar(eps)) {
      meta::static_for<lie_size>([&](auto i) {s_[i] *= mul;});
    }
  }

  // REQUIRED GROUP API

public:
  /**
   * @brief Set to identity element
   */
  void setIdentity() requires ModifiableStorageLike<Storage>
  {
    s_[0] = Scalar(0); s_[1] = Scalar(0); s_[2] = Scalar(0); s_[3] = Scalar(1);
  }

  /**
   * @brief Set to a random element
   */
  template<typename RNG>
  void setRandom(RNG & rng)
  requires ModifiableStorageLike<Storage>&& std::is_floating_point_v<Scalar>
  {
    const Scalar u1 = filler<Scalar>(rng, 0);
    const Scalar u2 = Scalar(2 * M_PI) * filler<Scalar>(rng, 0);
    const Scalar u3 = Scalar(2 * M_PI) * filler<Scalar>(rng, 0);
    Scalar a = sqrt(1. - u1), b = sqrt(u1);

    const Scalar su2 = sin(u2);
    if (su2 < 0) {
      a *= -1;
      b *= -1;
    }
    s_[0] = a * cos(u2);  s_[1] = b * sin(u3); s_[2] = b * cos(u3); s_[3] = a * su2;
  }

  /**
   * @brief Matrix lie group element
   */
  MatrixGroup matrix_group() const
  {
    return quat().toRotationMatrix();
  }

  /**
   * @brief Group action
   */
  template<typename Derived>
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_actdim)
  Vector operator*(const Eigen::MatrixBase<Derived> & x) const
  {
    return quat() * x;
  }

  /**
   * @brief Group composition
   */
  template<StorageLike OS>
  Group operator*(const SO3<Scalar, OS> & r) const
  {
    return Group(quat() * r.quat());
  }

  /**
   * @brief Group inverse
   */
  Group inverse() const
  {
    return Group(quat().inverse());
  }

  /**
   * @brief Group logarithm
   */
  Tangent log() const
  {
    using std::atan2, std::sqrt;
    const Scalar xyz2 = s_[0] * s_[0] + s_[1] * s_[1] + s_[2] * s_[2];

    Scalar phi;
    if (xyz2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+atan%28y%2Fx%29+%2F+y+at+y%3D0
      phi = Scalar(1) / s_[3] - xyz2 / (Scalar(3) * s_[3] * s_[3] * s_[3]);
    } else {
      Scalar xyz = sqrt(xyz2);
      phi = Scalar(2) * atan2(xyz, s_[3]) / xyz;
    }
    return phi * Tangent(s_[0], s_[1], s_[2]);
  }

  /**
   * @brief Group adjoint
   */
  TangentMap Ad() const
  {
    return quat().toRotationMatrix();
  }

  // REQUIRED TANGENT API

  /**
   * @brief Group exponential
   *
   * Valid arguments are (-pi, pi]
   */
  template<typename Derived>
  static Group exp(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    using std::sqrt, std::cos, std::sin;

    const Scalar th2 = a.squaredNorm();

    Scalar A, B;
    if (th2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+sin%28x%2F2%29%2Fx+at+x%3D0
      A = Scalar(1) / Scalar(2) - th2 / Scalar(48);
      // https://www.wolframalpha.com/input/?i=series+cos%28x%2F2%29+at+x%3D0
      B = Scalar(1) - th2 / Scalar(8);
    } else {
      const Scalar th = sqrt(th2);
      A = sin(th / Scalar(2)) / th;
      B = cos(th / Scalar(2));
    }

    return Group(A * a.x(), A * a.y(), A * a.z(), B);
  }

  /**
   * @brief Algebra adjoint
   */
  template<typename Derived>
  static TangentMap ad(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    return hat(a);
  }

  /**
   * @brief Algebra hat
   */
  template<typename Derived>
  static MatrixGroup hat(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    return (MatrixGroup() <<
           Scalar(0), -a.z(), a.y(),
           a.z(), Scalar(0), -a.x(),
           -a.y(), a.x(), Scalar(0)
    ).finished();
  }

  /**
   * @brief Algebra vee
   */
  template<typename Derived>
  static Tangent vee(const Eigen::MatrixBase<Derived> & A)
  requires(Derived::RowsAtCompileTime == lie_dim && Derived::ColsAtCompileTime == lie_dim)
  {
    return Tangent(A(2, 1) - A(1, 2), A(0, 2) - A(2, 0), A(1, 0) - A(0, 1)) / Scalar(2);
  }

  /**
   * @brief Right jacobian of the exponential map
   */
  template<typename Derived>
  static TangentMap dr_exp(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    using std::sqrt, std::sin, std::cos;
    const Scalar th2 = a.squaredNorm();

    Scalar A, B;
    if (th2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+%281-cos+x%29+%2F+x%5E2+at+x%3D0
      A = Scalar(1) / Scalar(2) - th2 / Scalar(24);
      // https://www.wolframalpha.com/input/?i=series+%28x+-+sin%28x%29%29+%2F+x%5E3+at+x%3D0
      B = Scalar(1) / Scalar(6) - th2 / Scalar(120);
    } else {
      const Scalar th = sqrt(th2);
      A = (Scalar(1) - cos(th)) / th2;
      B = (th - sin(th)) / (th2 * th);
    }

    const TangentMap ad = SO3<Scalar>::ad(a);
    return TangentMap::Identity() - A * ad + B * ad * ad;
  }

  /**
   * @brief Inverse of the right jacobian of the exponential map
   */
  template<typename Derived>
  static TangentMap dr_expinv(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    using std::sqrt, std::sin, std::cos;
    const Scalar th2 = a.squaredNorm();

    Scalar A;
    if (th2 < Scalar(eps2)) {
      // https://www.wolframalpha.com/input/?i=series+1%2Fx%5E2-%281%2Bcos+x%29%2F%282*x*sin+x%29+at+x%3D0
      A = Scalar(1) / Scalar(12) + th2 / Scalar(720);
    } else {
      const Scalar th = sqrt(th2);
      A = Scalar(1) / th2 - (Scalar(1) + cos(th)) / (Scalar(2) * th * sin(th));
    }

    const TangentMap ad = SO3<Scalar>::ad(a);
    return TangentMap::Identity() + ad / Scalar(2) + A * ad * ad;
  }
};

using SO3f = SO3<float>;
using SO3d = SO3<double>;

}  // namespace smooth

#endif  // SMOOTH__SO3_HPP_

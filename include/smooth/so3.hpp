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
    const Scalar xyz_n = sqrt(s_[0] * s_[0] + s_[1] * s_[1] + s_[2] * s_[2]);
    if (xyz_n < Scalar(eps)) {
      // TODO: small angle approx
      return Tangent::Zero();
    }
    const Scalar p = Scalar(2) * atan2(xyz_n, s_[3]) / xyz_n;
    return p * Tangent(s_[0], s_[1], s_[2]);
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

    Scalar cth, sth_over_th;
    if (th2 >= Scalar(eps2)) {
      // exact formula
      const Scalar th = sqrt(th2);
      cth = cos(th / Scalar(2));
      sth_over_th = sin(th / Scalar(2)) / th;
    } else {
      // small-angle approximations:
      //   cos(th / 2) = 1 - th2 / 8
      //   sin(th / 2) / th = 1 - th^2 / 48
      cth = Scalar(1) - th2 / Scalar(8);
      sth_over_th = Scalar(0.5) - th2 / Scalar(48);
    }
    return Group(a.x() * sth_over_th, a.y() * sth_over_th, a.z() * sth_over_th, cth);
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
    const Scalar th = sqrt(th2);

    if (th < Scalar(eps)) {
      // TODO: small angle approximation
      return TangentMap::Identity();
    }

    const TangentMap ad = SO3<Scalar>::ad(a);

    return TangentMap::Identity() -
           (Scalar(1) - cos(th)) / th2 * ad +
           (th - sin(th)) / (th2 * th) * ad * ad;
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
    const Scalar th = sqrt(th2);
    const TangentMap ad = SO3<Scalar>::ad(a);

    if (th < Scalar(eps)) {
      // TODO: small angle approximation
      return TangentMap::Identity() + ad / Scalar(2);
    }

    return TangentMap::Identity() +
           ad / Scalar(2) +
           ( (Scalar(1) / th2) - (Scalar(1) + cos(th)) / (Scalar(2) * th * sin(th))) * ad * ad;
  }
};

using SO3f = SO3<float>;
using SO3d = SO3<double>;

}  // namespace smooth

#endif  // SMOOTH__SO3_HPP_

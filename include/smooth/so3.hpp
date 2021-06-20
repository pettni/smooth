#ifndef SMOOTH__SO3_HPP_
#define SMOOTH__SO3_HPP_

#include <Eigen/Geometry>

#include "common.hpp"
#include "concepts.hpp"
#include "macro.hpp"
#include "storage.hpp"


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
requires(_Storage::Size == 4 && std::is_same_v<typename _Storage::Scalar, _Scalar>)
class SO3
{
private:
  _Storage s_;

  template<typename Scalar, StorageLike OS>
  requires(OS::Size == 4 && std::is_same_v<typename OS::Scalar, Scalar>)
  friend class SO3;

public:
  // REQUIRED CONSTANTS

  static constexpr Eigen::Index RepSize = 4;
  static constexpr Eigen::Index Dof = 3;
  static constexpr Eigen::Index Dim = 3;
  static constexpr Eigen::Index ActDim = 3;

  // REQUIRED TYPES

  SMOOTH_COMMON_TYPES(SO3)

  // CONSTRUCTORS AND OPERATORS

  SMOOTH_COMMON_API(SO3)

  // SO3-SPECIFIC API

  /**
   * @brief Construct from quaternion
   * @param q unit quaternion
   */
  template<typename Derived>
  SO3(const Eigen::QuaternionBase<Derived> & q)
  requires ModifiableStorageLike<Storage>
    && std::is_same_v<typename Derived::Scalar, Scalar>
  {
    quat() = q.normalized();
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
  requires (!MappableStorageLike<Storage>)
  {
    return Eigen::Quaternion<Scalar>(s_[3], s_[0], s_[1], s_[2]);
  }

  /**
   * @brief Return euler angles
   * @param i1, i2, i3 euler angle axis convention (0=x, 1=y, 2=z).
   *        Default values correspond to ZYX rotation.
   *
   * Returned angles a1, a2, a3 are s.t. rotation is described by
   * Rot_i1(a1) * Rot_i2(a2) * Rot_i3(a3)
   */
  Eigen::Matrix<Scalar, 3, 1> eulerAngles(
    Eigen::Index i1 = 2,
    Eigen::Index i2 = 1,
    Eigen::Index i3 = 0) const
  {
    return quat().toRotationMatrix().eulerAngles(i1, i2, i3);
  }

  // REQUIRED GROUP API

public:
  /**
   * @brief Set to identity element
   */
  void setIdentity() requires ModifiableStorageLike<Storage>
  {
    quat().setIdentity();
  }

  /**
   * @brief Set to a random quaternion with positive w component
   */
  void setRandom()
  requires ModifiableStorageLike<Storage>
  {
    quat() = Eigen::Quaternion<Scalar>::UnitRandom();

    if (s_[3] < 0) {
      for (auto i = 0u; i != 4; ++i) {
        s_[i] *= -1;
      }
    }
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
  Vector operator*(const Eigen::MatrixBase<Derived> & x) const
  {
    return quat() * x;
  }

  /**
   * @brief Group composition
   */
  template<StorageLike OS>
  PlainObject operator*(const SO3<Scalar, OS> & r) const
  {
    return PlainObject(quat() * r.quat());
  }

  /**
   * @brief Group inverse
   */
  PlainObject inverse() const
  {
    return PlainObject(quat().inverse());
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
      phi = Scalar(2) / s_[3] - Scalar(2) * xyz2 / (Scalar(3) * s_[3] * s_[3] * s_[3]);
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
  static PlainObject exp(const Eigen::MatrixBase<Derived> & a)
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

    return PlainObject(Eigen::Quaternion<Scalar>(B, A * a.x(), A * a.y(), A * a.z()));
  }

  /**
   * @brief Algebra adjoint
   */
  template<typename Derived>
  static TangentMap ad(const Eigen::MatrixBase<Derived> & a)
  {
    return hat(a);
  }

  /**
   * @brief Algebra hat
   */
  template<typename Derived>
  static MatrixGroup hat(const Eigen::MatrixBase<Derived> & a)
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
  {
    return (Tangent() << A.coeff(2, 1) - A.coeff(1, 2),
             A.coeff(0, 2) - A.coeff(2, 0),
             A.coeff(1, 0) - A.coeff(0, 1)
           ).finished() / Scalar(2);
  }

  /**
   * @brief Right jacobian of the exponential map
   */
  template<typename Derived>
  static TangentMap dr_exp(const Eigen::MatrixBase<Derived> & a)
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

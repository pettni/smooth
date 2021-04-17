#ifndef SMOOTH__SO3_HPP_
#define SMOOTH__SO3_HPP_

#include <Eigen/Geometry>
#include <random>

#include "common.hpp"
#include "concepts.hpp"
#include "lie_group_base.hpp"

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
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template<typename OtherScalar, StorageLike OS>
  requires std::is_same_v<_Scalar, OtherScalar>
  friend class SO3;

  friend class LieGroupBase<SO3<_Scalar, _Storage>, 4>;

  /**
   * @brief Construct from coefficients (does not normalize)
   */
  template<typename Scalar>
  explicit SO3(const Scalar & qx, const Scalar & qy, const Scalar & qz, const Scalar & qw)
  {
    s_[0] = qx; s_[1] = qy; s_[2] = qz, s_[3] = qw;
  }

public:
  // REQUIRED CONSTANTS

  static constexpr uint32_t lie_size = 4;
  static constexpr uint32_t lie_dof = 3;
  static constexpr uint32_t lie_dim = 3;
  static constexpr uint32_t lie_actdim = 3;

  // REQUIRED TYPES

  using Storage = _Storage;
  using Scalar = _Scalar;

  using Group = SO3<Scalar, DefaultStorage<Scalar, lie_size>>;
  using Tangent = Eigen::Matrix<Scalar, lie_dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, lie_dof, lie_dof>;
  using Vector = Eigen::Matrix<Scalar, lie_actdim, 1>;
  using MatrixGroup = Eigen::Matrix<Scalar, lie_dim, lie_dim>;

  // CONSTRUCTOR AND OPERATOR BOILERPLATE

  SO3() = default;
  SO3(const SO3 & o) = default;
  SO3(SO3 && o) = default;
  SO3 & operator=(const SO3 & o) = default;
  SO3 & operator=(SO3 && o) = default;
  ~SO3() = default;

  /**
   * @brief Copy constructor from other storage types
   */
  template<StorageLike OS>
  SO3(const SO3<Scalar, OS> & o)
  requires ModifiableStorageLike<Storage>
  {
    static_for<lie_size>([&](auto i) {s_[i] = o.coeffs()[i];});
  }

  /**
   * @brief Forwarding constructor to storage for map types
   */
  template<typename T>
  requires std::is_constructible_v<Storage, T *>
  explicit SO3(T * ptr)
  : s_(ptr) {}

  /**
   * @brief Forwarding constructor to storage for const map types
   */
  template<typename T>
  requires std::is_constructible_v<Storage, const T *>
  explicit SO3(const T * ptr)
  : s_(ptr) {}

  /**
   * @brief Copy assignment from other SO3
   */
  template<StorageLike OS>
  SO3 & operator=(const SO3<Scalar, OS> & o)
  requires ModifiableStorageLike<Storage>
  {
    static_for<lie_size>([&](auto i) {s_[i] = o.s_[i];});
    return *this;
  }

  // SO3-SPECIFIC API

public:
  /**
   * @brief Construct from quaternion
   */
  template<typename Derived>
  SO3(const Eigen::QuaternionBase<Derived> & qin)
  requires ModifiableStorageLike<Storage> && std::is_same_v<typename Derived::Scalar, Scalar>
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
   * @brief Normalize quaternion and set qw >= 0
   */
  void normalize()
  {
    Scalar mul = Scalar(1) /
      std::sqrt(s_[0] * s_[0] + s_[1] * s_[1] + s_[2] * s_[2] + s_[3] * s_[3]);
    if (s_[3] < 0) {
      mul *= Scalar(-1);
    }
    if (std::abs(mul - Scalar(1)) > eps<Scalar>) {
      static_for<lie_size>([&](auto i) {s_[i] *= mul;});
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
  requires(Derived::SizeAtCompileTime == lie_actdim)
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
    if (xyz_n < eps<Scalar>) {
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
  template<typename TangentDerived>
  static Group exp(const Eigen::MatrixBase<TangentDerived> & a)
  {
    using std::cos, std::sin;

    const Scalar th = a.norm();
    const Scalar cth = cos(th / 2);

    if (th < eps<Scalar>) {
      // small-angle approximation sin(th) / th = 1 - th^2/2
      const Scalar app = 1 - th * th;
      return Eigen::Quaternion<Scalar>(cth, a.x() * app, a.y() * app, a.z() * app);
    }

    const Scalar sth_over_th = sin(th / 2) / th;
    return Group(a.x() * sth_over_th, a.y() * sth_over_th, a.z() * sth_over_th, cth);
  }

  /**
   * @brief Algebra adjoint
   */
  template<typename TangentDerived>
  static TangentMap ad(const Eigen::MatrixBase<TangentDerived> & a)
  {
    return hat(a);
  }

  /**
   * @brief Algebra hat
   */
  template<typename TangentDerived>
  static MatrixGroup hat(const Eigen::MatrixBase<TangentDerived> & a)
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
  template<typename AlgebraDerived>
  static Tangent vee(const Eigen::MatrixBase<AlgebraDerived> & A)
  {
    return Tangent(A(2, 1) - A(1, 2), A(0, 2) - A(2, 0), A(1, 0) - A(0, 1)) / Scalar(2);
  }

  /**
   * @brief Right jacobian of the exponential map
   */
  template<typename TangentDerived>
  static TangentMap dr_exp(const Eigen::MatrixBase<TangentDerived> & a)
  {
    using std::sqrt, std::sin, std::cos;
    const Scalar th2 = a.squaredNorm();
    const Scalar th = sqrt(th2);

    if (th < eps<Scalar>) {
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
  template<typename TangentDerived>
  static TangentMap dr_expinv(const Eigen::MatrixBase<TangentDerived> & a)
  {
    using std::sqrt, std::sin, std::cos;
    const Scalar th2 = a.squaredNorm();
    const Scalar th = sqrt(th2);
    const TangentMap ad = SO3<Scalar>::ad(a);

    if (th < eps<Scalar>) {
      // TODO: small angle approximation
      return TangentMap::Identity() + ad / 2;
    }

    return TangentMap::Identity() +
           ad / 2 +
           ( (Scalar(1) / th2) - (Scalar(1) + cos(th)) / (Scalar(2) * th * sin(th))) * ad * ad;
  }
};

using SO3f = SO3<float>;
using SO3d = SO3<double>;

}  // namespace smooth

#endif  // SMOOTH__SO3_HPP_

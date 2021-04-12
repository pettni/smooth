#ifndef SMOOTH__SO3_HPP_
#define SMOOTH__SO3_HPP_

#include <Eigen/Geometry>

#include <random>

#include "concepts.hpp"
#include "common.hpp"
#include "lie_group_base.hpp"


namespace smooth
{

/**
 * @brief SO3 Lie Group
 *
 * Memory layout: qx qy qz qw  (same as Eigen quaternion)
 */
template<typename _Scalar, typename _Storage = DefaultStorage<_Scalar, 4>>
requires StorageLike<_Storage, _Scalar, 4>
class SO3 : public LieGroupBase<SO3<_Scalar, _Storage>, 4>
{
private:
  _Storage s_;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template<typename OtherScalar, typename OS>
  requires std::is_same_v<_Scalar, OtherScalar>
  friend class SO3;

  friend class LieGroupBase<SO3<_Scalar, _Storage>, 4>;

  // Helper for coefficient constructor
  template<typename ... Scalars, uint32_t ... idx>
  SO3(std::integer_sequence<uint32_t, idx...>, Scalars && ... args)
  {
    ((s_[idx] = std::forward<Scalars>(args)), ...);
  }

public:
  // REQUIRED CONSTANTS

  static constexpr uint32_t size = 4;
  static constexpr uint32_t dof = 3;
  static constexpr uint32_t dim = 3;

  // REQUIRED TYPES

  using Storage = _Storage;
  using Scalar = _Scalar;

  using Group = SO3<Scalar, DefaultStorage<Scalar, size>>;
  using MatrixGroup = Eigen::Matrix<Scalar, dim, dim>;
  using Tangent = Eigen::Matrix<Scalar, dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, dof, dof>;
  using Algebra = Eigen::Matrix<Scalar, dim, dim>;
  using Vector = Eigen::Matrix<Scalar, dim, 1>;

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
  template<typename OS>
  requires StorageLike<OS, Scalar, size>
  SO3(const SO3<Scalar, OS> & o)
  {
    static_for<size>([&](auto i) {s_[i] = o.coeffs()[i];});
  }

  /**
   * @brief Construct from coefficients
   */
  template<typename ... Scalars>
  explicit SO3(Scalars && ... args)
  requires std::conjunction_v<std::is_same<Scalar, Scalars>...>
  : SO3(std::make_integer_sequence<uint32_t, size>{}, std::forward<Scalars>(args)...)
  {}

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
  template<typename OS>
  requires StorageLike<OS, Scalar, size>
  SO3 & operator=(const SO3<Scalar, OS> & o)
  {
    static_for<size>([&](auto i) {s_[i] = o.s_[i];});
    return *this;
  }

  // SO3-SPECIFIC API

  /**
   * @brief Construct from quaternion
   */
  template<typename Derived>
  SO3(const Eigen::QuaternionBase<Derived> & qin)
  requires std::is_same_v<typename Derived::Scalar, Scalar>
  {
    const Eigen::Quaternion<typename Derived::Scalar> qin_norm = qin.normalized();
    s_[0] = qin_norm.x(); s_[1] = qin_norm.y(); s_[2] = qin_norm.z(); s_[3] = qin_norm.w();
  }

  /**
   * @brief Access as Eigen quaternion by Map
   *
   * Only available for ordered storage
   */
  Eigen::Map<Eigen::Quaternion<Scalar>> unit_quaternion()
  requires OrderedModifiableStorageLike<Storage, Scalar, size>
  {
    return Eigen::Map<Eigen::Quaternion<Scalar>>(s_.data());
  }

  /**
   * @brief Access as Eigen quaternion by const Map
   *
   * Only available for ordered storage
   */
  Eigen::Map<const Eigen::Quaternion<Scalar>> unit_quaternion() const
  requires OrderedStorageLike<Storage, Scalar, size>
  {
    return Eigen::Map<const Eigen::Quaternion<Scalar>>(s_.data());
  }

  /**
   * @brief Access as Eigen quaternion by copy
   *
   * Only for non-ordered storage (for ordered storage use map versions)
   */
  Eigen::Quaternion<Scalar> unit_quaternion() const
  requires UnorderedStorageLike<Storage, Scalar, size>
  {
    return Eigen::Quaternion<Scalar>(s_[3], s_[0], s_[1], s_[2]);
  }

  // REQUIRED GROUP API

  /**
   * @brief Set to identity element
   */
  void setIdentity() requires ModifiableStorageLike<Storage, Scalar, 4>
  {
    s_[0] = Scalar(0); s_[1] = Scalar(0); s_[2] = Scalar(0); s_[3] = Scalar(1);
  }

  /**
   * @brief Set to a random element
   */
  template<typename RNG>
  void setRandom(RNG & rng)
  requires ModifiableStorageLike<Storage, Scalar, 4>&& std::is_floating_point_v<Scalar>
  {
    const Scalar u1 = filler<Scalar>(rng, 0);
    const Scalar u2 = Scalar(2 * M_PI) * filler<Scalar>(rng, 0);
    const Scalar u3 = Scalar(2 * M_PI) * filler<Scalar>(rng, 0);
    const Scalar a = sqrt(1. - u1), b = sqrt(u1);

    // x y z w
    s_[0] = a * cos(u2);  s_[1] = b * sin(u3); s_[2] = b * cos(u3); s_[3] = a * sin(u2);
  }

  /**
   * @brief Matrix lie group element
   */
  MatrixGroup matrix() const
  {
    return unit_quaternion().toRotationMatrix();
  }

  /**
   * @brief Group action
   */
  template<typename Derived>
  requires(Derived::SizeAtCompileTime == dim)
  Vector operator*(const Eigen::MatrixBase<Derived> & x) const
  {
    return unit_quaternion() * x;
  }

  /**
   * @brief Group composition
   */
  template<typename OS>
  Group operator*(const SO3<Scalar, OS> & r) const
  {
    return Group(unit_quaternion() * r.unit_quaternion());
  }

  /**
   * @brief Group inverse
   */
  Group inverse() const
  {
    return Group(unit_quaternion().inverse());
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
    return unit_quaternion().toRotationMatrix();
  }

  // REQUIRED TANGENT API

  /**
   * @brief Group exponential
   */
  template<typename TangentDerived>
  static Group exp(const Eigen::MatrixBase<TangentDerived> & t)
  {
    using std::cos, std::sin;

    const Scalar th = t.norm();
    const Scalar cth = cos(th / 2);

    if (th < eps<Scalar>) {
      // small-angle approximation sin(th) / th = 1 - th^2/2
      const Scalar app = 1 - th * th;
      return Eigen::Quaternion<Scalar>(cth, t.x() * app, t.y() * app, t.z() * app);
    }

    const Scalar sth_over_th = sin(th / 2) / th;
    Group ret;
    ret.s_[0] = t.x() * sth_over_th;
    ret.s_[1] = t.y() * sth_over_th;
    ret.s_[2] = t.z() * sth_over_th;
    ret.s_[3] = cth;
    return ret;
  }

  /**
   * @brief Algebra adjoint
   */
  template<typename TangentDerived>
  static TangentMap ad(const Eigen::MatrixBase<TangentDerived> & t)
  {
    return hat(t);
  }

  /**
   * @brief Algebra hat
   */
  template<typename TangentDerived>
  static Algebra hat(const Eigen::MatrixBase<TangentDerived> & t)
  {
    return (Algebra() <<
           Scalar(0), -t.z(), t.y(),
           t.z(), Scalar(0), -t.x(),
           -t.y(), t.x(), Scalar(0)
    ).finished();
  }

  /**
   * @brief Algebra vee
   */
  template<typename AlgebraDerived>
  static Tangent vee(const Eigen::MatrixBase<AlgebraDerived> & a)
  {
    return Tangent(a(2, 1) - a(1, 2), a(0, 2) - a(2, 0), a(1, 0) - a(0, 1)) / Scalar(2);
  }

  /**
   * @brief Right jacobian of the exponential map
   */
  template<typename TangentDerived>
  static TangentMap dr_exp(const Eigen::MatrixBase<TangentDerived> & t)
  {
    using std::sqrt, std::sin, std::cos;
    const Scalar th2 = t.squaredNorm();
    if (th2 < eps<Scalar>) {
      // TODO: small angle approximation
      return TangentMap::Identity();
    }

    const Scalar th = sqrt(th2);
    const TangentMap ad = SO3<Scalar>::ad(t);

    return TangentMap::Identity() -
           (Scalar(1) - cos(th)) / th2 * ad +
           (th - sin(th)) / (th2 * th) * ad * ad;
  }

  /**
   * @brief Inverse of the right jacobian of the exponential map
   */
  template<typename TangentDerived>
  static TangentMap dr_expinv(const Eigen::MatrixBase<TangentDerived> & t)
  {
    using std::sqrt, std::sin, std::cos;
    const Scalar th2 = t.squaredNorm();
    const TangentMap ad = SO3<Scalar>::ad(t);
    if (th2 < eps<Scalar>) {
      // TODO: small angle approximation
      return TangentMap::Identity() + ad / 2;
    }

    const Scalar th = sqrt(th2);

    return TangentMap::Identity() +
           ad / 2 +
           ( (Scalar(1) / th2) - (Scalar(1) + cos(th)) / (Scalar(2) * th * sin(th))) * ad * ad;
  }
};

using SO3f = SO3<float>;
using SO3d = SO3<double>;

}  // namespace smooth

#endif  // SMOOTH__SO3_HPP_

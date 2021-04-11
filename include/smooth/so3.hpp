#ifndef SMOOTH__SO3_HPP_
#define SMOOTH__SO3_HPP_

#include <Eigen/Geometry>

#include <random>

#include "concepts.hpp"
#include "common.hpp"


namespace smooth
{

/**
 * @brief SO3 Lie Group
 *
 * Memory layout: qx qy qz qw
 */
template<typename _Scalar, typename _S = DefaultStorage<_Scalar, 4>>
requires ConstStorageLike<_S, _Scalar, 4>
class SO3
{
public:
  static constexpr uint32_t size = 4;
  static constexpr uint32_t dof = 3;
  static constexpr uint32_t dim = 3;

  /////////////////
  // GENERIC API //
  /////////////////

  using Storage = _S;
  using Scalar = _Scalar;

  using Group = SO3<Scalar, DefaultStorage<Scalar, size>>;
  using Tangent = Eigen::Matrix<Scalar, dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, dof, dof>;
  using Algebra = Eigen::Matrix<Scalar, dim, dim>;
  using Vector = Eigen::Matrix<Scalar, dim, 1>;

  SO3() = default;
  SO3(const SO3 & o) = default;
  SO3(SO3 && o) = default;
  // need copy constructor to copy between maps
  SO3 & operator=(const SO3 & o)
  {
    static_for<size>([&](auto i) {s_[i] = o.s_[i];});
    return *this;
  }
  SO3 & operator=(SO3 && o) = default;

  /**
   * @brief Copy constructor from other SO3
   */
  template<typename OS>
  requires ConstStorageLike<OS, Scalar, size>
  SO3(const SO3<Scalar, OS> & o)
  {
    static_for<size>([&](auto i) {s_[i] = o.s_[i];});
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
  template<typename OS>
  requires ConstStorageLike<OS, Scalar, size>
  SO3 & operator=(const SO3<Scalar, OS> & o)
  {
    static_for<size>([&](auto i) {s_[i] = o.s_[i];});
    return *this;
  }

  /**
   * @brief Cast to different datatype
   */
  template<typename NewScalar>
  SO3<NewScalar, DefaultStorage<NewScalar, 4>> cast() const
  {
    SO3<NewScalar, DefaultStorage<NewScalar, 4>> ret;
    ret.coeffs() = s_.template cast<NewScalar>();
    return ret;
  }

  /**
   * @brief Access group storage
   */
  Storage & coeffs()
  {
    return s_;
  }

  /**
   * @brief Const access group storage
   */
  const Storage & coeffs() const
  {
    return s_;
  }

private:
  Storage s_;

  template<typename OtherScalar, typename OS>
  requires std::is_same_v<Scalar, OtherScalar>
  friend class SO3;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

////////////////////
// SPECIFIC STUFF //
////////////////////

public:
  /**
   * @brief Construct from quaternion
   */
  template<typename Derived>
  SO3(const Eigen::QuaternionBase<Derived> & qin)
  requires std::is_same_v<typename Derived::Scalar, Scalar>
  {
    unit_quaternion() = qin.normalized();
  }

  // REQUIRED API

  /**
   * @brief Set to identity element
   */
  void setIdentity() requires StorageLike<Storage, Scalar, 4>
  {
    unit_quaternion().setIdentity();
  }

  // TODO: move to crtp
  static SO3<Scalar> Identity()
  {
    SO3<Scalar> ret;
    ret.setIdentity();
    return ret;
  }

  /**
   * @brief Set to a random element
   */
  template<typename RNG>
  void setRandom(RNG & rng)
  requires StorageLike<Storage, Scalar, 4>&& std::is_arithmetic_v<Scalar>
  {
    const Scalar u1 = filler<Scalar>(rng, 0);
    const Scalar u2 = Scalar(2 * M_PI) * filler<Scalar>(rng, 0);
    const Scalar u3 = Scalar(2 * M_PI) * filler<Scalar>(rng, 0);

    const Scalar a = sqrt(1. - u1), b = sqrt(u1);

    s_[0] = a * cos(u2);
    s_[1] = b * sin(u3);
    s_[2] = b * cos(u3);
    s_[3] = a * sin(u2);
  }

  // TODO: move to crtp
  template<typename RNG>
  static SO3<Scalar> Random(RNG & rng)
  {
    SO3<Scalar> ret;
    ret.setRandom(rng);
    return ret;
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
  requires ConstStorageLike<OS, Scalar, 4>
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
    using std::atan2;
    const Scalar xyz_n = Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>>(s_.data()).norm();

    if (xyz_n < eps<Scalar>) {
      return Tangent::Zero();
    }

    const Scalar p = Scalar(2) * atan2(xyz_n, unit_quaternion().w()) / xyz_n;

    return p * Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>>(s_.data());
  }

  /**
   * @brief Group adjoint
   */
  TangentMap Ad() const
  {
    return unit_quaternion().toRotationMatrix();
  }

  /**
   * @brief Group exponential
   */
  static Group exp(const Tangent & t)
  {
    using std::cos, std::sin;

    const Scalar th = t.norm();
    const Scalar cth = cos(th / 2);

    if (th < eps<Scalar>) {
      // small-angle approximation sin(th) / th = 1 - th^2/2
      const Scalar app = 1 - th * th;
      return Eigen::Quaternion<Scalar>(
        cth, t.x() * app, t.y() * app, t.z() * app
      );
    }

    const Scalar sth_over_th = sin(th / 2) / th;

    return Group(
      Eigen::Quaternion<Scalar>(
        cth, t.x() * sth_over_th, t.y() * sth_over_th, t.z() * sth_over_th
    ));
  }

  /**
   * @brief Algebra adjoint
   */
  static TangentMap ad(const Tangent & t)
  {
    return hat(t);
  }

  /**
   * @brief Algebra hat
   */
  static Algebra hat(const Tangent & t)
  {
    return (Algebra() <<
           Scalar(0), -t.z(), t.y(),
           t.z(), Scalar(0), -t.x(),
           -t.y(), t.x(), Scalar(0)
    ).finished();
  }

  /**
   * @brief Right jacobian of the exponential map
   */
  static TangentMap dr_exp(const Tangent & t)
  {
    using std::sqrt, std::sin, std::cos;
    const Scalar th2 = t.squaredNorm();
    if (th2 < eps<Scalar>) {
      // TODO: small angle approximation
      return TangentMap::Identity();
    }

    const Scalar th = sqrt(th2);
    const TangentMap ad = SO3<Scalar>::ad(t);

    return TangentMap::Identity()
      - (Scalar(1) - cos(th)) / th2 * ad
      + (th - sin(th)) / (th2 * th) * ad * ad;
  }

  /**
   * @brief Inverse of the right jacobian of the exponential map
   */
  static TangentMap dr_expinv(const Tangent & t)
  {
    using std::sqrt, std::sin, std::cos;
    const Scalar th2 = t.squaredNorm();
    const TangentMap ad = SO3<Scalar>::ad(t);
    if (th2 < eps<Scalar>) {
      // TODO: small angle approximation
      return TangentMap::Identity() + ad / 2;
    }

    const Scalar th = sqrt(th2);

    return TangentMap::Identity()
      + ad / 2
      + ( (Scalar(1) / th2) - (Scalar(1) + cos(th)) / (Scalar(2) * th * sin(th))) * ad * ad;
  }

  // SYNTACTIC SUGAR
  // TODO: inject this with crtp
  /**
   * @brief Access raw data pointer
  */
  Scalar * data() requires StorageLike<Storage, Scalar, 4>
  {
    return s_.data();
  }

  /**
   * @brief Access raw data pointer
   */
  const Scalar * data() const
  {
    return s_.data();
  }

  // TODO: plus/minus/operator* etc

  // SO3-specific stuff
  Eigen::Map<Eigen::Quaternion<Scalar>> unit_quaternion()
  {
    return Eigen::Map<Eigen::Quaternion<Scalar>>(s_.data());
  }

  Eigen::Map<const Eigen::Quaternion<Scalar>> unit_quaternion() const
  {
    return Eigen::Map<const Eigen::Quaternion<Scalar>>(s_.data());
  }
};

// Group typedefs
using SO3d = SO3<double>;

}  // namespace smooth

#endif  // SMOOTH__SO3_HPP_

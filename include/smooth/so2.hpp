#ifndef SMOOTH__SO2_HPP_
#define SMOOTH__SO2_HPP_

#include <Eigen/Geometry>
#include <random>

#include "common.hpp"
#include "concepts.hpp"
#include "lie_group_base.hpp"

namespace smooth
{

/**
 * @brief SO2 Lie Group
 *
 * Memory layout
 * =============
 * Group:    qz qw
 * Tangent:  wz
 *
 * Constraints
 * ===========
 * Group:   qz * qz + qw * qw = 1
 * Tangent: -pi < wz <= pi

 */
template<typename _Scalar, typename _Storage = DefaultStorage<_Scalar, 2>>
requires StorageLike<_Storage, _Scalar, 2>
class SO2 : public LieGroupBase<SO2<_Scalar, _Storage>, 2>
{
private:
  _Storage s_;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template<typename OtherScalar, typename OS>
  requires std::is_same_v<_Scalar, OtherScalar>
  friend class SO2;

  friend class LieGroupBase<SO2<_Scalar, _Storage>, 2>;

  /**
   * @brief Construct from coefficients (does not normalize)
   */
  template<typename Scalar>
  explicit SO2(const Scalar & qz, const Scalar & qw)
  {
    s_[0] = qz; s_[1] = qw;
  }

public:
  // REQUIRED CONSTANTS

  static constexpr uint32_t size = 2;
  static constexpr uint32_t dof = 1;
  static constexpr uint32_t dim = 2;
  static constexpr uint32_t act_dim = 2;

  // REQUIRED TYPES

  using Storage = _Storage;
  using Scalar = _Scalar;

  using Group = SO2<Scalar, DefaultStorage<Scalar, size>>;
  using MatrixGroup = Eigen::Matrix<Scalar, dim, dim>;
  using Tangent = Eigen::Matrix<Scalar, dof, 1>;
  using TangentMap = Eigen::Matrix<Scalar, dof, dof>;
  using Algebra = Eigen::Matrix<Scalar, dim, dim>;
  using Vector = Eigen::Matrix<Scalar, act_dim, 1>;

  // CONSTRUCTOR AND OPERATOR BOILERPLATE

  SO2() = default;
  SO2(const SO2 & o) = default;
  SO2(SO2 && o) = default;
  SO2 & operator=(const SO2 & o) = default;
  SO2 & operator=(SO2 && o) = default;
  ~SO2() = default;

  /**
   * @brief Copy constructor from other storage types
   */
  template<typename OS>
  requires StorageLike<OS, Scalar, size>
  SO2(const SO2<Scalar, OS> & o)
  {
    static_for<size>([&](auto i) {s_[i] = o.coeffs()[i];});
  }

  /**
   * @brief Forwarding constructor to storage for map types
   */
  template<typename T>
  requires std::is_constructible_v<Storage, T *>
  explicit SO2(T * ptr)
  : s_(ptr) {}

  /**
   * @brief Forwarding constructor to storage for const map types
   */
  template<typename T>
  requires std::is_constructible_v<Storage, const T *>
  explicit SO2(const T * ptr)
  : s_(ptr) {}

  /**
   * @brief Copy assignment from other SO2
   */
  template<typename OS>
  requires StorageLike<OS, Scalar, size>
  SO2 & operator=(const SO2<Scalar, OS> & o)
  {
    static_for<size>([&](auto i) {s_[i] = o.s_[i];});
    return *this;
  }

  // SO2-SPECIFIC API

public:
  /**
   * @brief Construct from coefficients
   *
   * @param qz sine of angle
   * @param qw cosine of angle
   */
  SO2(const Scalar & qz, const Scalar & qw)
  {
    s_[0] = qz;
    s_[1] = qw;
    normalize();
  }

  /**
   * @brief Construct from angle
   */
  static SO2 rot(const Scalar & angle)
  {
    return exp(Tangent(angle));
  }

  /**
   * @brief Rotation angle in interval (-pi, pi]
   */
  Scalar angle() const
  {
    return log()(0);
  }

private:
  /**
   * @brief Normalize parameters
   */
  void normalize()
  {
    const Scalar mul = Scalar(1) / (s_[0] * s_[0] + s_[1] * s_[1]);
    s_[0] *= mul;
    s_[1] *= mul;
  }

  // REQUIRED GROUP API

public:
  /**
   * @brief Set to identity element
   */
  void setIdentity() requires ModifiableStorageLike<Storage, Scalar, size>
  {
    s_[0] = Scalar(0); s_[1] = Scalar(1);
  }

  /**
   * @brief Set to a random element
   */
  template<typename RNG>
  void setRandom(RNG & rng)
  requires ModifiableStorageLike<Storage, Scalar, size>&& std::is_floating_point_v<Scalar>
  {
    const Scalar u = Scalar(2 * M_PI) * filler<Scalar>(rng, 0);
    s_[0] = sin(u); s_[1] = cos(u);
  }

  /**
   * @brief Matrix lie group element
   */
  MatrixGroup matrix() const
  {
    return (MatrixGroup() << s_[1], -s_[0], s_[0], s_[1]).finished();
  }

  /**
   * @brief Group action
   */
  template<typename Derived>
  requires(Derived::SizeAtCompileTime == act_dim)
  Vector operator*(const Eigen::MatrixBase<Derived> & x) const
  {
    return matrix() * x;
  }

  /**
   * @brief Group composition
   */
  template<typename OS>
  Group operator*(const SO2<Scalar, OS> & r) const
  {
    return Group(s_[0] * r.s_[1] + s_[1] * r.s_[0], s_[1] * r.s_[1] - s_[0] * r.s_[0]);
  }

  /**
   * @brief Group inverse
   */
  Group inverse() const
  {
    return Group(-s_[0], s_[1]);
  }

  /**
   * @brief Group logarithm
   */
  Tangent log() const
  {
    using std::atan2;
    return Tangent(atan2(s_[0], s_[1]));
  }

  /**
   * @brief Group adjoint
   */
  TangentMap Ad() const
  {
    return TangentMap::Identity();
  }

  // REQUIRED TANGENT API

  /**
   * @brief Group exponential
   */
  template<typename TangentDerived>
  static Group exp(const Eigen::MatrixBase<TangentDerived> & t)
  {
    using std::cos, std::sin;
    return Group(sin(t.x()), cos(t.x()));
  }

  /**
   * @brief Algebra adjoint
   */
  template<typename TangentDerived>
  static TangentMap ad(const Eigen::MatrixBase<TangentDerived> &)
  {
    return TangentMap::Zero();
  }

  /**
   * @brief Algebra hat
   */
  template<typename TangentDerived>
  static MatrixGroup hat(const Eigen::MatrixBase<TangentDerived> & t)
  {
    return (Algebra() <<
           Scalar(0), -t.x(),
           t.x(), Scalar(0)
    ).finished();
  }

  /**
   * @brief Algebra vee
   */
  template<typename AlgebraDerived>
  static Tangent vee(const Eigen::MatrixBase<AlgebraDerived> & a)
  {
    return Tangent(a(1, 0) - a(0, 1)) / Scalar(2);
  }

  /**
   * @brief Right jacobian of the exponential map
   */
  template<typename TangentDerived>
  static TangentMap dr_exp(const Eigen::MatrixBase<TangentDerived> &)
  {
    return TangentMap::Identity();
  }

  /**
   * @brief Inverse of the right jacobian of the exponential map
   */
  template<typename TangentDerived>
  static TangentMap dr_expinv(const Eigen::MatrixBase<TangentDerived> &)
  {
    return TangentMap::Identity();
  }
};

using SO2f = SO2<float>;
using SO2d = SO2<double>;

}  // namespace smooth

#endif  // SMOOTH__SO2_HPP_

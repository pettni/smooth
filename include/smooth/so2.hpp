#ifndef SMOOTH__SO2_HPP_
#define SMOOTH__SO2_HPP_

#include <random>

#include "common.hpp"
#include "concepts.hpp"
#include "lie_group_base.hpp"
#include "macro.hpp"


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
template<typename _Scalar, StorageLike _Storage = DefaultStorage<_Scalar, 2>>
requires(_Storage::SizeAtCompileTime == 2 && std::is_same_v<typename _Storage::Scalar, _Scalar>)
class SO2 : public LieGroupBase<SO2<_Scalar, _Storage>, 2>
{
private:
  _Storage s_;

public:
  // REQUIRED CONSTANTS

  static constexpr uint32_t lie_size = 2;
  static constexpr uint32_t lie_dof = 1;
  static constexpr uint32_t lie_dim = 2;
  static constexpr uint32_t lie_actdim = 2;

  // CONSTRUCTORS AND OPERATORS

  SMOOTH_GROUP_BOILERPLATE(SO2)

  // SO2-SPECIFIC API

  /**
   * @brief Construct from coefficients
   *
   * @param qz sine of angle
   * @param qw cosine of angle
   */
  SO2(const Scalar & qz, const Scalar & qw)
  requires ModifiableStorageLike<Storage>
  {
    using std::sqrt;
    const Scalar n = sqrt(qz * qz + qw * qw);
    s_[0] = qz / n;
    s_[1] = qw / n;
  }

  /**
   * @brief Construct from angle
   */
  static SO2 rot(const Scalar & angle)
  requires ModifiableStorageLike<Storage>
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

  // REQUIRED GROUP API

public:
  /**
   * @brief Set to identity element
   */
  void setIdentity() requires ModifiableStorageLike<Storage>
  {
    s_[0] = Scalar(0); s_[1] = Scalar(1);
  }

  /**
   * @brief Set to a random element
   */
  template<typename RNG>
  void setRandom(RNG & rng)
  requires ModifiableStorageLike<Storage>&& std::is_floating_point_v<Scalar>
  {
    const Scalar u = Scalar(2 * M_PI) * filler<Scalar>(rng, 0);
    s_[0] = sin(u); s_[1] = cos(u);
  }

  /**
   * @brief Matrix lie group element
   */
  MatrixGroup matrix_group() const
  {
    return (MatrixGroup() << s_[1], -s_[0], s_[0], s_[1]).finished();
  }

  /**
   * @brief Group action
   */
  template<typename Derived>
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_actdim)
  Vector operator*(const Eigen::MatrixBase<Derived> & x) const
  {
    return matrix_group() * x;
  }

  /**
   * @brief Group composition
   */
  template<StorageLike OS>
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
  template<typename Derived>
  static Group exp(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    using std::cos, std::sin;
    return Group(sin(a.x()), cos(a.x()));
  }

  /**
   * @brief Algebra adjoint
   */
  template<typename Derived>
  static TangentMap ad(const Eigen::MatrixBase<Derived> &)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    return TangentMap::Zero();
  }

  /**
   * @brief Algebra hat
   */
  template<typename Derived>
  static MatrixGroup hat(const Eigen::MatrixBase<Derived> & a)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    return (MatrixGroup() <<
           Scalar(0), -a.x(),
           a.x(), Scalar(0)
    ).finished();
  }

  /**
   * @brief Algebra vee
   */
  template<typename Derived>
  static Tangent vee(const Eigen::MatrixBase<Derived> & A)
  requires(Derived::RowsAtCompileTime == lie_dim && Derived::ColsAtCompileTime == lie_dim)
  {
    return Tangent(A(1, 0) - A(0, 1)) / Scalar(2);
  }

  /**
   * @brief Right jacobian of the exponential map
   */
  template<typename Derived>
  static TangentMap dr_exp(const Eigen::MatrixBase<Derived> &)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    return TangentMap::Identity();
  }

  /**
   * @brief Inverse of the right jacobian of the exponential map
   */
  template<typename Derived>
  static TangentMap dr_expinv(const Eigen::MatrixBase<Derived> &)
  requires(Derived::IsVectorAtCompileTime == 1 && Derived::SizeAtCompileTime == lie_dof)
  {
    return TangentMap::Identity();
  }
};

using SO2f = SO2<float>;
using SO2d = SO2<double>;

}  // namespace smooth

#endif  // SMOOTH__SO2_HPP_

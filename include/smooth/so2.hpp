#ifndef SMOOTH__SO2_HPP_
#define SMOOTH__SO2_HPP_

#include <Eigen/Core>

#include <complex>

#include "impl/so2.hpp"
#include "lie_group_base.hpp"
#include "macro.hpp"

namespace smooth {

template<typename Scalar>
class SO3;

// CRTP BASE

/**
 * @brief SO2 Lie Group represented as U(1)
 *
 * Memory layout
 * =============
 * Group:    qz qw
 * Tangent:  Ωz
 *
 * Lie group Matrix form
 * =====================
 *
 * [ qw -qz ]
 * [ qz  qw ]
 *
 * Lie algebra Matrix form
 * =======================
 *
 * [  0 -Ωz ]
 * [ Ωz   0 ]
 *
 * Constraints
 * ===========
 * Group:   qz * qz + qw * qw = 1
 * Tangent: -pi < Ωz <= pi
 */
template<typename _Derived>
class SO2Base : public LieGroupBase<_Derived>
{
protected:
  using Base = LieGroupBase<_Derived>;
  SO2Base()  = default;

public:
  SMOOTH_INHERIT_TYPEDEFS

  /**
   * @brief Angle represetation.
   */
  Scalar angle() const { return Base::log().x(); }

  /**
   * @brief Complex number (U(1)) representation.
   */
  std::complex<Scalar> u1() const
  {
    return std::complex<Scalar>(Base::coeffs().y(), Base::coeffs().x());
  }

  /**
   * @brief Rotation action on 2D vector.
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 2, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return Base::matrix() * v;
  }

  /**
   * @brief Lift to SO3.
   *
   * Rotation of SO2 is embedded in SO3 as a rotation around the z axis.
   */
  SO3<Scalar> lift_so3() const
  {
    using std::cos, std::sin;

    const Scalar yaw = Base::log().x();
    return SO3<Scalar>(Eigen::Quaternion<Scalar>(cos(yaw / 2), 0, 0, sin(yaw / 2)));
  }
};

// STORAGE TYPE TRAITS

template<typename _Scalar>
class SO2;

template<typename _Scalar>
struct lie_traits<SO2<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = SO2Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SO2<NewScalar>;
};

// STORAGE TYPE

template<typename _Scalar>
class SO2 : public SO2Base<SO2<_Scalar>>
{
  using Base = SO2Base<SO2<_Scalar>>;
  SMOOTH_GROUP_API(SO2)
public:
  /**
   * @brief Construct from underlying representation.
   *
   * Incoming arguments are normalized to ensure group constraint.
   *
   * @param qz sine of angle
   * @param qw cosine of angle
   */
  SO2(const Scalar & qz, const Scalar & qw)
  {
    const Scalar n = sqrt(qw * qw + qz * qz);
    coeffs_.x() = qz / n;
    coeffs_.y() = qw / n;
  }

  /**
   * @brief Construct from angle.
   */
  explicit SO2(const Scalar & angle)
  {
    using std::cos, std::sin;
    coeffs_.x() = sin(angle);
    coeffs_.y() = cos(angle);
  }

  /**
   * @brief Construct from complex number.
   */
  SO2(const std::complex<Scalar> & c)
  {
    coeffs_.x() = c.imag();
    coeffs_.y() = c.real();
  }
};

using SO2f = SO2<float>;
using SO2d = SO2<double>;

}  // namespace smooth

// MAP TYPE TRAITS

template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<smooth::SO2<_Scalar>>>
    : public lie_traits<smooth::SO2<_Scalar>>
{};

// MAP TYPE

template<typename _Scalar>
class Eigen::Map<smooth::SO2<_Scalar>> : public smooth::SO2Base<Eigen::Map<smooth::SO2<_Scalar>>>
{
  using Base = smooth::SO2Base<Eigen::Map<smooth::SO2<_Scalar>>>;
  SMOOTH_MAP_API(Map)
};

// CONST MAP TYPE TRAITS

template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<const smooth::SO2<_Scalar>>>
    : public lie_traits<smooth::SO2<_Scalar>>
{
  static constexpr bool is_mutable = false;
};

// CONST MAP TYPE

template<typename _Scalar>
class Eigen::Map<const smooth::SO2<_Scalar>>
    : public smooth::SO2Base<Eigen::Map<const smooth::SO2<_Scalar>>>
{
  using Base = smooth::SO2Base<Eigen::Map<const smooth::SO2<_Scalar>>>;
  SMOOTH_CONST_MAP_API(Map)
};

#endif  // SMOOTH__SO2_HPP_

// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <complex>

#include <Eigen/Core>

#include "detail/macro.hpp"
#include "detail/so2.hpp"
#include "lie_group_base.hpp"

namespace smooth {
inline namespace v1_0 {

// \cond
template<typename Scalar>
class SO3;
// \endcond

/**
 * @brief Base class for SO2 Lie group types.
 *
 * Internally represented as \f$\mathbb{U}(1)\f$ (unit complex numbers).
 *
 * Memory layout
 * -------------
 *
 * - Group:    \f$ \mathbf{x} = [q_z, q_w] \f$
 * - Tangent:  \f$ \mathbf{a} = [\omega_z] \f$
 *
 * Constraints
 * -----------
 *
 * - Group:   \f$q_z^2 + q_w^2 = 1 \f$
 * - Tangent: \f$ -\pi < \omega_z \leq \pi \f$
 *
 * Lie group matrix form
 * ---------------------
 *
 * \f[
 * \mathbf{X} =
 * \begin{bmatrix}
 *  q_w & -q_z \\
 *  q_z &  q_w
 * \end{bmatrix} \in \mathbb{R}^{2 \times 2}
 * \f]
 *
 *
 * Lie algebra matrix form
 * -----------------------
 *
 * \f[
 * \mathbf{a}^\wedge =
 * \begin{bmatrix}
 *   0 & -\omega_z \\
 *  \omega_z &   0 \\
 * \end{bmatrix} \in \mathbb{R}^{2 \times 2}
 * \f]
 */
template<typename _Derived>
class SO2Base : public LieGroupBase<_Derived>
{
  using Base = LieGroupBase<_Derived>;

protected:
  SO2Base() = default;

public:
  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * @brief Angle representation, in \f$[-\pi,\pi]\f$ range
   */
  Scalar angle() const { return Base::log().x(); }

  /**
   * @brief Angle representation in clockwise direction, in \f$[-2\pi,0]\f$ range
   */
  Scalar angle_cw() const
  {
    const auto & x = static_cast<const _Derived &>(*this).coeffs().y();
    const auto & y = static_cast<const _Derived &>(*this).coeffs().x();

    using std::atan2;
    if (y <= 0.) {
      return atan2(y, x);
    } else {
      return atan2(-y, -x) - Scalar(M_PI);
    }
  }

  /**
   * @brief Angle representation in clockwise direction, in \f$[0,2\pi]\f$ range
   */
  Scalar angle_ccw() const
  {
    const auto & x = static_cast<const _Derived &>(*this).coeffs().y();
    const auto & y = static_cast<const _Derived &>(*this).coeffs().x();

    using std::atan2;
    if (y >= 0.) {
      return atan2(y, x);
    } else {
      return Scalar(M_PI) + atan2(-y, -x);
    }
  }

  /**
   * @brief Unit complex number (U(1)) representation.
   */
  Eigen::Vector2<Scalar> unit_complex() const
  {
    return Eigen::Vector2<Scalar>(
      static_cast<const _Derived &>(*this).coeffs().y(), static_cast<const _Derived &>(*this).coeffs().x());
  }

  /**
   * @brief Unit complex number (U(1)) representation.
   */
  std::complex<Scalar> u1() const
  {
    return std::complex<Scalar>(
      static_cast<const _Derived &>(*this).coeffs().y(), static_cast<const _Derived &>(*this).coeffs().x());
  }

  /**
   * @brief Rotation action on 2D vector.
   */
  template<typename EigenDerived>
  Eigen::Vector2<Scalar> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return Base::matrix() * v;
  }

  /**
   * @brief Jacobian of rotation action w.r.t. group.
   *
   * \f[
   *   \mathrm{d}^r (X v)_X
   * \f]
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 2, 1> dr_action(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return Base::matrix() * Base::hat(Eigen::Vector<Scalar, 1>::Ones()) * v;
  }

  /**
   * @brief Lift to SO3.
   *
   * Rotation of SO2 is embedded in SO3 as a rotation around the z axis.
   *
   * @note SO3 header must be included.
   */
  SO3<Scalar> lift_so3() const
  {
    using std::cos, std::sin;

    const Scalar yaw = Base::log().x();
    return SO3<Scalar>(Eigen::Quaternion<Scalar>(cos(yaw / 2), 0, 0, sin(yaw / 2)));
  }
};

// \cond
template<typename _Scalar>
class SO2;
// \endcond

// \cond
template<typename _Scalar>
struct liebase_info<SO2<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = SO2Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SO2<NewScalar>;
};
// \endcond

/**
 * @brief Storage implementation of SO2 Lie group.
 *
 * @see SO2Base for memory layout.
 */
template<typename _Scalar>
class SO2 : public SO2Base<SO2<_Scalar>>
{
  using Base = SO2Base<SO2<_Scalar>>;
  SMOOTH_GROUP_API(SO2);

public:
  /**
   * @brief Construct from coefficients.
   *
   * @param qz sine of rotation angle
   * @param qw cosine of rotation angle
   *
   * @note Inputs are are normalized to ensure group constraint.
   */
  SO2(const Scalar & qz, const Scalar & qw)
  {
    using std::sqrt;

    const Scalar n = sqrt(qw * qw + qz * qz);
    m_coeffs.x()   = qz / n;
    m_coeffs.y()   = qw / n;
  }

  /**
   * @brief Construct from angle.
   *
   * @param angle angle of rotation (radians).
   */
  explicit SO2(const Scalar & angle)
  {
    using std::cos, std::sin;

    m_coeffs.x() = sin(angle);
    m_coeffs.y() = cos(angle);
  }

  /**
   * @brief Construct from unit complex number.
   *
   * @param c complex number.
   *
   * @note Input is normalized to ensure group constraint.
   */
  SO2(const std::complex<Scalar> & c)
  {
    using std::sqrt;

    const Scalar n = sqrt(c.imag() * c.imag() + c.real() * c.real());
    m_coeffs.x()   = c.imag() / n;
    m_coeffs.y()   = c.real() / n;
  }
};

// \cond
template<typename _Scalar>
struct liebase_info<Map<SO2<_Scalar>>> : public liebase_info<SO2<_Scalar>>
{};
// \endcond

/**
 * @brief Memory mapping of SO2 Lie group.
 *
 * @see SO2Base for memory layout.
 */
template<typename _Scalar>
class Map<SO2<_Scalar>> : public SO2Base<Map<SO2<_Scalar>>>
{
  using Base = SO2Base<Map<SO2<_Scalar>>>;

  SMOOTH_MAP_API();
};

// \cond
template<typename _Scalar>
struct liebase_info<Map<const SO2<_Scalar>>> : public liebase_info<SO2<_Scalar>>
{
  static constexpr bool is_mutable = false;
};
// \endcond

/**
 * @brief Const memory mapping of SO2 Lie group.
 *
 * @see SO2Base for memory layout.
 */
template<typename _Scalar>
class Map<const SO2<_Scalar>> : public SO2Base<Map<const SO2<_Scalar>>>
{
  using Base = SO2Base<Map<const SO2<_Scalar>>>;

  SMOOTH_CONST_MAP_API();
};

using SO2f = SO2<float>;   ///< SO2 with float scalar representation
using SO2d = SO2<double>;  ///< SO2 with double scalar representation

}  // namespace v1_0
}  // namespace smooth

// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SMOOTH__SO2_HPP_
#define SMOOTH__SO2_HPP_

#include <Eigen/Core>

#include <complex>

#include "internal/lie_group_base.hpp"
#include "internal/macro.hpp"
#include "internal/so2.hpp"
#include "lie_group.hpp"
#include "map.hpp"

namespace smooth {

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
   * @brief Angle represetation.
   */
  Scalar angle() const { return Base::log().x(); }

  /**
   * @brief Unit complex number (U(1)) representation.
   */
  std::complex<Scalar> u1() const
  {
    return std::complex<Scalar>(static_cast<const _Derived &>(*this).coeffs().y(),
      static_cast<const _Derived &>(*this).coeffs().x());
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
struct lie_traits<SO2<_Scalar>>
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
    coeffs_.x()    = qz / n;
    coeffs_.y()    = qw / n;
  }

  /**
   * @brief Construct from angle.
   *
   * @param angle angle of rotation (radians).
   */
  explicit SO2(const Scalar & angle)
  {
    using std::cos, std::sin;

    coeffs_.x() = sin(angle);
    coeffs_.y() = cos(angle);
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
    coeffs_.x()    = c.imag() / n;
    coeffs_.y()    = c.real() / n;
  }
};

// \cond
template<typename _Scalar>
struct lie_traits<Map<SO2<_Scalar>>> : public lie_traits<SO2<_Scalar>>
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

  SMOOTH_MAP_API(SO2);
};

// \cond
template<typename _Scalar>
struct lie_traits<Map<const SO2<_Scalar>>> : public lie_traits<SO2<_Scalar>>
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

}  // namespace smooth

#endif  // SMOOTH__SO2_HPP_

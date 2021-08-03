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

#ifndef SMOOTH__C1_HPP_
#define SMOOTH__C1_HPP_

#include <Eigen/Core>

#include <complex>

#include "internal/c1.hpp"
#include "internal/lie_group_base.hpp"
#include "internal/macro.hpp"
#include "so2.hpp"

namespace smooth {

// \cond
template<typename Scalar>
class SO2;
// \endcond

/**
 * @brief Base class for C1 Lie group types.
 *
 * Memory layout
 * -------------
 *
 * - Group:    \f$ \mathbf{x} = [a, b] \f$
 * - Tangent:  \f$ \mathbf{a} = [s, \omega_z] \f$
 *
 * Constraints
 * -----------
 *
 * - Group:   \f$ a^2 + b^2 > 0 \f$
 * - Tangent: \f$ -\pi < \omega_z \leq \pi \f$
 *
 * Lie group matrix form
 * ---------------------
 *
 * \f[
 * \mathbf{X} =
 * \begin{bmatrix}
 *  b & -a \\
 *  a &  b
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
 *   s & -\omega_z \\
 *  \omega_z &   s \\
 * \end{bmatrix} \in \mathbb{R}^{2 \times 2}
 * \f]
 */
template<typename _Derived>
class C1Base : public LieGroupBase<_Derived>
{
  using Base = LieGroupBase<_Derived>;

protected:
  C1Base() = default;

public:
  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * @brief Rotation angle.
   */
  Scalar angle() const
  {
    using std::atan2;

    return atan2(static_cast<const _Derived &>(*this).coeffs().x(),
      static_cast<const _Derived &>(*this).coeffs().y());
  }

  /**
   * @brief Scaling.
   */
  Scalar scaling() const
  {
    using std::sqrt;

    return sqrt(static_cast<const _Derived &>(*this).coeffs().x()
                  * static_cast<const _Derived &>(*this).coeffs().x()
                + static_cast<const _Derived &>(*this).coeffs().y()
                    * static_cast<const _Derived &>(*this).coeffs().y());
  }

  /**
   * @brief Rotation.
   */
  SO2<Scalar> so2() const
  {
    return SO2<Scalar>(c1());  // it's normalized inside SO2
  }

  /**
   * @brief Complex number representation.
   */
  std::complex<Scalar> c1() const
  {
    return std::complex<Scalar>(static_cast<const _Derived &>(*this).coeffs().y(),
      static_cast<const _Derived &>(*this).coeffs().x());
  }

  /**
   * @brief Rotation and scaling action on 2D vector.
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 2, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return Base::matrix() * v;
  }
};

// \cond
template<typename _Scalar>
class C1;
// \endcond

// \cond
template<typename _Scalar>
struct lie_traits<C1<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = C1Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = C1<NewScalar>;
};
// \endcond

/**
 * @brief Storage implementation of C1 Lie group.
 *
 * @see C1Base for memory layout.
 */
template<typename _Scalar>
class C1 : public C1Base<C1<_Scalar>>
{
  using Base = C1Base<C1<_Scalar>>;
  SMOOTH_GROUP_API(C1);

public:
  /**
   * @brief Construct from scaling and angle.
   *
   * @param scaling strictly greater than zero.
   * @param angle angle of rotation (radians).
   */
  C1(const Scalar & scaling, const Scalar & angle)
  {
    using std::cos, std::sin;

    coeffs_.x() = scaling * sin(angle);
    coeffs_.y() = scaling * cos(angle);
  }

  /**
   * @brief Construct from complex number.
   *
   * @param c complex number.
   */
  C1(const std::complex<Scalar> & c)
  {
    coeffs_.x() = c.imag();
    coeffs_.y() = c.real();
  }
};

using C1f = C1<float>;   ///< C1 with float scalar representation
using C1d = C1<double>;  ///< C1 with double scalar representation

}  // namespace smooth

// \cond
template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<smooth::C1<_Scalar>>> : public lie_traits<smooth::C1<_Scalar>>
{};
// \endcond

/**
 * @brief Memory mapping of C1 Lie group.
 *
 * @see C1Base for memory layout.
 */
template<typename _Scalar>
class Eigen::Map<smooth::C1<_Scalar>> : public smooth::C1Base<Eigen::Map<smooth::C1<_Scalar>>>
{
  using Base = smooth::C1Base<Eigen::Map<smooth::C1<_Scalar>>>;

  SMOOTH_MAP_API(Map);
};

// \cond
template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<const smooth::C1<_Scalar>>>
    : public lie_traits<smooth::C1<_Scalar>>
{
  static constexpr bool is_mutable = false;
};
// \endcond

/**
 * @brief Const memory mapping of C1 Lie group.
 *
 * @see C1Base for memory layout.
 */
template<typename _Scalar>
class Eigen::Map<const smooth::C1<_Scalar>>
    : public smooth::C1Base<Eigen::Map<const smooth::C1<_Scalar>>>
{
  using Base = smooth::C1Base<Eigen::Map<const smooth::C1<_Scalar>>>;

  SMOOTH_CONST_MAP_API(Map);
};

#endif  // SMOOTH__C1_HPP_

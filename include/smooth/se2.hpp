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

#ifndef SMOOTH__SE2_HPP_
#define SMOOTH__SE2_HPP_

#include <Eigen/Core>

#include <complex>

#include "internal/lie_group_base.hpp"
#include "internal/macro.hpp"
#include "internal/se2.hpp"
#include "lie_group.hpp"
#include "so2.hpp"

namespace smooth {

// \cond
template<typename Scalar>
class SE3;
// \endcond

/**
 * @brief Base class for SE2 Lie group types.
 *
 * Internally represented as \f$\mathbb{U}(1) \times \mathbb{R}^2\f$.
 *
 * Memory layout
 * -------------
 *
 * - Group:    \f$ \mathbf{x} = [x, y, q_z, q_w] \f$
 * - Tangent:  \f$ \mathbf{a} = [v_x, v_y, \omega_z] \f$
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
 *  q_w & -q_z & x \\
 *  q_z &  q_w & y \\
 *  0   &    0 & 1
 * \end{bmatrix} \in \mathbb{R}^{3 \times 3}
 * \f]
 *
 *
 * Lie algebra matrix form
 * -----------------------
 *
 * \f[
 * \mathbf{a}^\wedge =
 * \begin{bmatrix}
 *   0 & -\omega_z & v_x \\
 *  \omega_z &   0 & v_y \\
 *  0 & 0 & 0
 * \end{bmatrix} \in \mathbb{R}^{3 \times 3}
 * \f]
 */
template<typename _Derived>
class SE2Base : public LieGroupBase<_Derived>
{
  using Base = LieGroupBase<_Derived>;

protected:
  SE2Base() = default;

public:
  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * @brief Access SO(2) part.
   */
  Map<SO2<Scalar>> so2() requires is_mutable
  {
    return Map<SO2<Scalar>>(static_cast<_Derived &>(*this).data() + 2);
  }

  /**
   * @brief Const access SO(2) part.
   */
  Map<const SO2<Scalar>> so2() const
  {
    return Map<const SO2<Scalar>>(static_cast<const _Derived &>(*this).data() + 2);
  }

  /**
   * @brief Access R2 part.
   */
  Eigen::Map<Eigen::Matrix<Scalar, 2, 1>> r2() requires is_mutable
  {
    return Eigen::Map<Eigen::Matrix<Scalar, 2, 1>>(static_cast<_Derived &>(*this).data());
  }

  /**
   * @brief Const access R2 part.
   */
  Eigen::Map<const Eigen::Matrix<Scalar, 2, 1>> r2() const
  {
    return Eigen::Map<const Eigen::Matrix<Scalar, 2, 1>>(
      static_cast<const _Derived &>(*this).data());
  }

  /**
   * @brief Tranformation action on 2D vector.
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 2, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return so2() * v + r2();
  }

  /**
   * @brief Lift to SE3.
   *
   * @note SE3 header must be included.
   */
  SE3<Scalar> lift_se3() const
  {
    return SE3<Scalar>(
      so2().lift_so3(), Eigen::Matrix<Scalar, 3, 1>(r2().x(), r2().y(), Scalar(0)));
  }
};

// \cond
template<typename _Scalar>
class SE2;
// \endcond

// \cond
template<typename _Scalar>
struct lie_traits<SE2<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = SE2Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SE2<NewScalar>;
};
// \endcond

/**
 * @brief Storage implementation of SE2 Lie group.
 *
 * @see SE2Base for memory layout.
 */
template<typename _Scalar>
class SE2 : public SE2Base<SE2<_Scalar>>
{
  using Base = SE2Base<SE2<_Scalar>>;

  SMOOTH_GROUP_API(SE2);

public:
  /**
   * @brief Construct from SO2 and R2.
   *
   * @param so2 orientation component.
   * @param r2 translation component.
   */
  template<typename SO2Derived, typename T2Derived>
  SE2(const SO2Base<SO2Derived> & so2, const Eigen::MatrixBase<T2Derived> & r2)
  {
    Base::so2() = static_cast<const SO2Derived &>(so2);
    Base::r2()  = static_cast<const T2Derived &>(r2);
  }
};

// MAP TYPE TRAITS

// \cond
template<typename _Scalar>
struct lie_traits<Map<SE2<_Scalar>>> : public lie_traits<SE2<_Scalar>>
{};
// \endcond

/**
 * @brief Memory mapping of SE2 Lie group.
 *
 * @see SE2Base for memory layout.
 */
template<typename _Scalar>
class Map<SE2<_Scalar>> : public SE2Base<Map<SE2<_Scalar>>>
{
  using Base = SE2Base<Map<SE2<_Scalar>>>;

  SMOOTH_MAP_API(SE2);
};

// \cond
template<typename _Scalar>
struct lie_traits<Map<const SE2<_Scalar>>> : public lie_traits<SE2<_Scalar>>
{
  static constexpr bool is_mutable = false;
};
// \endcond

/**
 * @brief Const memory mapping of SE2 Lie group.
 *
 * @see SE2Base for memory layout.
 */
template<typename _Scalar>
class Map<const SE2<_Scalar>> : public SE2Base<Map<const SE2<_Scalar>>>
{
  using Base = SE2Base<Map<const SE2<_Scalar>>>;

  SMOOTH_CONST_MAP_API();
};

using SE2f = SE2<float>;   ///< SE2 with float
using SE2d = SE2<double>;  ///< SE2 with double

}  // namespace smooth

#endif  // SMOOTH__SE2_HPP_

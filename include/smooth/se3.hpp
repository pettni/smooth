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

#ifndef SMOOTH__SE3_HPP_
#define SMOOTH__SE3_HPP_

#include <Eigen/Core>

#include <complex>

#include "internal/lie_group_base.hpp"
#include "internal/macro.hpp"
#include "internal/se3.hpp"
#include "so3.hpp"

namespace smooth {

template<typename Scalar>
class SE2;

/**
 * @brief Base class for SE3 Lie group types.
 *
 * Internally represented as \f$\mathbb{S}^3 \times \mathbb{R}^3\f$.
 *
 * Memory layout
 * -------------
 *
 * - Group:    \f$ \mathbf{x} = [x, y, z, q_x, q_y, q_z, q_w] \f$
 * - Tangent:  \f$ \mathbf{a} = [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z] \f$
 *
 * Constraints
 * -----------
 *
 * - Group:   \f$q_x^2 + q_y^2 + q_z^2 + q_w^2 = 1 \f$
 * - Tangent: \f$ -\pi < \omega_x, \omega_y, \omega_z \leq \pi \f$
 *
 * Lie group matrix form
 * ---------------------
 *
 * \f[
 * \mathbf{X} =
 * \begin{bmatrix}
 *   R & T \\
 *   0 & 1
 * \end{bmatrix}
 * \f]
 *
 * where \f$R\f$ is a 3x3 rotation matrix and \f$ T = [x, y, z]^T \f$.
 *
 *
 * Lie algebra matrix form
 * -----------------------
 *
 * \f[
 * \mathbf{a}^\wedge =
 * \begin{bmatrix}
 *   0 & -\omega_z & \omega_y & v_x \\
 *  \omega_z &   0 & -\omega_x & v_y \\
 *  -\omega_y &  \omega_x & 0 v_y \\
 *  0 & 0 & 0 & 0
 * \end{bmatrix}
 * \f]
 */
template<typename _Derived>
class SE3Base : public LieGroupBase<_Derived>
{
  using Base = LieGroupBase<_Derived>;

protected:
  SE3Base() = default;

public:
  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * @brief Access SO(3) part.
   */
  Eigen::Map<SO3<Scalar>> so3() requires is_mutable
  {
    return Eigen::Map<SO3<Scalar>>(static_cast<_Derived &>(*this).data() + 3);
  }

  /**
   * @brief Const access SO(3) part.
   */
  Eigen::Map<const SO3<Scalar>> so3() const
  {
    return Eigen::Map<const SO3<Scalar>>(static_cast<const _Derived &>(*this).data() + 3);
  }

  /**
   * @brief Access T(3) part.
   */
  Eigen::Map<Eigen::Matrix<Scalar, 3, 1>> t3() requires is_mutable
  {
    return Eigen::Map<Eigen::Matrix<Scalar, 3, 1>>(static_cast<_Derived &>(*this).data());
  }

  /**
   * @brief Const access T(3) part.
   */
  Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> t3() const
  {
    return Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>>(
      static_cast<const _Derived &>(*this).data());
  }

  /**
   * @brief Tranformation action on 3D vector.
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, 3, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v)
  {
    return so3() * v + t3();
  }

  /**
   * @brief Project to SE2.
   *
   * @note SE2 header must be included.
   */
  SE2<Scalar> project_se2() const
  {
    return SE2<Scalar>(so3().project_so2(), t3().template head<2>());
  }
};

// \cond
template<typename _Scalar>
class SE3;
// \endcond

// \cond
template<typename _Scalar>
struct lie_traits<SE3<_Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = SE3Impl<_Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = SE3<NewScalar>;
};
// \endcond

/**
 * @brief Storage implementation of SE3 Lie group.
 *
 * @see SE3Base for memory layout.
 */
template<typename _Scalar>
class SE3 : public SE3Base<SE3<_Scalar>>
{
  using Base = SE3Base<SE3<_Scalar>>;

  SMOOTH_GROUP_API(SE3);

public:
  /**
   * @brief Construct from SO3 and translation.
   *
   * @param so3 orientation component.
   * @param t3 translation component.
   */
  template<typename SO3Derived, typename T3Derived>
  SE3(const SO3Base<SO3Derived> & so3, const Eigen::MatrixBase<T3Derived> & t3)
  {
    Base::so3() = so3;
    Base::t3()  = t3;
  }
};

using SE3f = SE3<float>;   ///< SE3 with float
using SE3d = SE3<double>;  ///< SE3 with double

}  // namespace smooth

// \cond
template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<smooth::SE3<_Scalar>>>
    : public lie_traits<smooth::SE3<_Scalar>>
{};
// \endcond

/**
 * @brief Memory mapping of SE3 Lie group.
 *
 * @see SE3Base for memory layout.
 */
template<typename _Scalar>
class Eigen::Map<smooth::SE3<_Scalar>> : public smooth::SE3Base<Eigen::Map<smooth::SE3<_Scalar>>>
{
  using Base = smooth::SE3Base<Eigen::Map<smooth::SE3<_Scalar>>>;

  SMOOTH_MAP_API(Map);
};

// \cond
template<typename _Scalar>
struct smooth::lie_traits<Eigen::Map<const smooth::SE3<_Scalar>>>
    : public lie_traits<smooth::SE3<_Scalar>>
{
  static constexpr bool is_mutable = false;
};
// \endcond

/**
 * @brief Const memory mapping of SE3 Lie group.
 *
 * @see SE3Base for memory layout.
 */
template<typename _Scalar>
class Eigen::Map<const smooth::SE3<_Scalar>>
    : public smooth::SE3Base<Eigen::Map<const smooth::SE3<_Scalar>>>
{
  using Base = smooth::SE3Base<Eigen::Map<const smooth::SE3<_Scalar>>>;

  SMOOTH_CONST_MAP_API(Map);
};

#endif  // SMOOTH__SE3_HPP_

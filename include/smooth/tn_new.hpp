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

#ifndef SMOOTH__TN_HPP_
#define SMOOTH__TN_HPP_

/**
 * @file
 * @brief Lie group of translations.
 */

#include <Eigen/Core>

#include "internal/lie_group_base.hpp"
#include "internal/macro.hpp"
#include "internal/tn.hpp"

namespace smooth {

/// Typedef Rn as Eigen vector
template<int _N, typename _Scalar>
requires(_N > 0) using Rn = Eigen::Matrix<_Scalar, _N, 1>;

// \cond
/// Specializing lie_traits for Eigen vectors makes it possible to use
// Eigen vectors in the bundle
template<int _N, typename _Scalar>
struct lie_traits<Rn<_N, _Scalar>>
{
  //! Lie group operations
  using Impl = TnImpl<_N, _Scalar>;
  //! Scalar type
  using Scalar = _Scalar;

  //! Plain return type.
  template<typename NewScalar>
  using PlainObject = Eigen::Matrix<NewScalar, _N, 1>;
};
// \endcond

/**
 * @brief Eigen type as \f$T(n)\f$ Lie Group.
 *
 * @note Tn is a typedef for an Eigen column vector and therefore does not
 * have the Lie group API from LieGroupBase.
 * lie_traits<Tn<N, Scalar>> defines the operations corresponding to the Lie group \f$T(n)\f$.
 *
 * Memory layout
 * -------------
 * - Group:    \f$ \mathbf{t} = [x_1, \ldots, x_n] \f$
 * - Tangent:  \f$ \mathbf{v} = [v_1, \ldots, v_n] \f$
 *
 * Lie group Matrix form
 * ---------------------
 *
 * \f[
 * \mathbf{X} =
 * \begin{bmatrix}
 *  I & \mathbf{t} \\
 *  0 & 1
 * \end{bmatrix}
 * \f]
 *
 * Lie algebra Matrix form
 * -----------------------
 *
 * \f[
 * \mathbf{v}^\wedge =
 * \begin{bmatrix}
 *  0 & \mathbf{v} \\
 *  0 & 0
 * \end{bmatrix}
 * \f]
 */
template<typename _Derived>
class TnBase : public LieGroupBase<_Derived>
{
  using Base = LieGroupBase<_Derived>;

protected:
  TnBase() = default;

public:
  SMOOTH_INHERIT_TYPEDEFS;

  /**
   * @brief Complex number (U(1)) representation.
   */
  Eigen::Map<Rn<Dof, Scalar>> rn() requires is_mutable
  {
    return Eigen::Map<Rn<Dof, Scalar>>(static_cast<_Derived &>(*this).data());
  }

  /**
   * @brief Complex number (U(1)) representation.
   */
  Eigen::Map<const Rn<Dof, Scalar>> rn() const
  {
    return Eigen::Map<const Rn<Dof, Scalar>>(static_cast<const _Derived &>(*this).data());
  }

  /**
   * @brief Translation action on ND vector.
   */
  template<typename EigenDerived>
  Eigen::Matrix<Scalar, Dof, 1> operator*(const Eigen::MatrixBase<EigenDerived> & v) const
  {
    return rn() + v;
  }
};

// \cond
template<int N, typename _Scalar>
requires(N > 0) class Tn;
// \endcond

// \cond
template<int N, typename _Scalar>
struct lie_traits<Tn<N, _Scalar>>
{
  static constexpr bool is_mutable = true;

  using Impl   = TnImpl<N, _Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = Tn<N, NewScalar>;
};
// \endcond

/**
 * @brief Storage implementation of Tn Lie group.
 *
 * @see TnBase for memory layout.
 */
template<int N, typename _Scalar>
requires(N > 0) class Tn : public TnBase<Tn<N, _Scalar>>
{
  using Base = TnBase<Tn<N, _Scalar>>;
  SMOOTH_GROUP_API(Tn);

public:
  /**
   * @brief Construct from Eigen vector.
   *
   * @param rn Eigen vector.
   */
  template<typename Derived>
  Tn(const Eigen::MatrixBase<Derived> & rn) : coeffs_(rn)
  {}
};

}  // namespace smooth

// \cond
template<int N, typename _Scalar>
struct smooth::lie_traits<Eigen::Map<smooth::Tn<N, _Scalar>>>
    : public lie_traits<smooth::Tn<N, _Scalar>>
{};
// \endcond

/**
 * @brief Memory mapping of Tn Lie group.
 *
 * @see TnBase for memory layout.
 */
template<int N, typename _Scalar>
class Eigen::Map<smooth::Tn<N, _Scalar>> : public smooth::TnBase<Eigen::Map<smooth::Tn<N, _Scalar>>>
{
  using Base = smooth::TnBase<Eigen::Map<smooth::Tn<N, _Scalar>>>;

  SMOOTH_MAP_API(Map);
};

// \cond
template<int N, typename _Scalar>
struct smooth::lie_traits<Eigen::Map<const smooth::Tn<N, _Scalar>>>
    : public lie_traits<smooth::Tn<N, _Scalar>>
{
  static constexpr bool is_mutable = false;
};
// \endcond

/**
 * @brief Const memory mapping of Tn Lie group.
 *
 * @see TnBase for memory layout.
 */
template<int N, typename _Scalar>
class Eigen::Map<const smooth::Tn<N, _Scalar>>
    : public smooth::TnBase<Eigen::Map<const smooth::Tn<N, _Scalar>>>
{
  using Base = smooth::TnBase<Eigen::Map<const smooth::Tn<N, _Scalar>>>;

  SMOOTH_CONST_MAP_API(Map);
};

#endif  // SMOOTH__TN_HPP_

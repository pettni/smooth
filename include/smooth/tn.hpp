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

#include <Eigen/Core>

#include "internal/lie_group_base.hpp"
#include "internal/macro.hpp"
#include "internal/tn.hpp"

namespace smooth {

/**
 * @brief Lie group \f$\mathbb{T}(n)\f$ of \f$n\f$-dimensional translations.
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
 * \end{bmatrix} \in \mathbb{R}^{n+1 \times n+1}
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
 * \end{bmatrix} \in \mathbb{R}^{n+1 \times n+1}
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
   * @brief Euclidean vector (Rn) representation.
   */
  Eigen::Map<Eigen::Matrix<Scalar, Dof, 1>> rn() requires is_mutable
  {
    return Eigen::Map<Eigen::Matrix<Scalar, Dof, 1>>(static_cast<_Derived &>(*this).data());
  }

  /**
   * @brief Euclidean vector (Rn) representation.
   */
  Eigen::Map<const Eigen::Matrix<Scalar, Dof, 1>> rn() const
  {
    return Eigen::Map<const Eigen::Matrix<Scalar, Dof, 1>>(
      static_cast<const _Derived &>(*this).data());
  }

  /**
   * @brief Translation action on Rn vector.
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
// \cond
requires(N > 0)
  // \endcond
  class Tn : public TnBase<Tn<N, _Scalar>>
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

namespace smooth {
// \cond
template<typename Scalar>
using T1 = Tn<1, Scalar>;
template<typename Scalar>
using T2 = Tn<2, Scalar>;
template<typename Scalar>
using T3 = Tn<3, Scalar>;
template<typename Scalar>
using T4 = Tn<4, Scalar>;
template<typename Scalar>
using T5 = Tn<5, Scalar>;
template<typename Scalar>
using T6 = Tn<6, Scalar>;
template<typename Scalar>
using T7 = Tn<7, Scalar>;
template<typename Scalar>
using T8 = Tn<8, Scalar>;
template<typename Scalar>
using T9 = Tn<9, Scalar>;
template<typename Scalar>
using T10 = Tn<10, Scalar>;

using T1f  = T1<float>;
using T2f  = T2<float>;
using T3f  = T3<float>;
using T4f  = T4<float>;
using T5f  = T5<float>;
using T6f  = T6<float>;
using T7f  = T7<float>;
using T8f  = T8<float>;
using T9f  = T9<float>;
using T10f = T10<float>;

using T1d  = T1<double>;
using T2d  = T2<double>;
using T3d  = T3<double>;
using T4d  = T4<double>;
using T5d  = T5<double>;
using T6d  = T6<double>;
using T7d  = T7<double>;
using T8d  = T8<double>;
using T9d  = T9<double>;
using T10d = T10<double>;
// \endcond
}  // namespace smooth

#endif  // SMOOTH__TN_HPP_

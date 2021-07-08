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

#include "internal/lie_group_base.hpp"
#include "internal/tn.hpp"

namespace smooth {

/**
 * @brief Eigen type as \f$T(n)\f$ Lie Group.
 *
 * \note Tn is a typedef for an Eigen column vector and therefore does not
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
template<int _N, typename _Scalar>
using Tn = Eigen::Matrix<_Scalar, _N, 1>;

// \cond
template<int _N, typename _Scalar>
struct lie_traits<Tn<_N, _Scalar>>
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

//! One-dimensional vector.
template<typename _Scalar>
using T1 = Tn<1, _Scalar>;

//! Two-dimensional vector.
template<typename _Scalar>
using T2 = Tn<2, _Scalar>;

//! Three-dimensional vector.
template<typename _Scalar>
using T3 = Tn<3, _Scalar>;

//! Four-dimensional vector.
template<typename _Scalar>
using T4 = Tn<4, _Scalar>;

//! Five-dimensional vector.
template<typename _Scalar>
using T5 = Tn<5, _Scalar>;

//! Six-dimensional vector.
template<typename _Scalar>
using T6 = Tn<6, _Scalar>;

//! Seven-dimensional vector.
template<typename _Scalar>
using T7 = Tn<7, _Scalar>;

//! Eight-dimensional vector.
template<typename _Scalar>
using T8 = Tn<8, _Scalar>;

//! Nine-dimensional vector.
template<typename _Scalar>
using T9 = Tn<9, _Scalar>;

//! Ten-dimensional vector.
template<typename _Scalar>
using T10 = Tn<10, _Scalar>;

//! One-dimensional float vector.
using T1f  = T1<float>;
//! Two-dimensional float vector.
using T2f  = T2<float>;
//! Three-dimensional float vector.
using T3f  = T3<float>;
//! Four-dimensional float vector.
using T4f  = T4<float>;
//! Five-dimensional float vector.
using T5f  = T5<float>;
//! Six-dimensional float vector.
using T6f  = T6<float>;
//! Seven-dimensional float vector.
using T7f  = T7<float>;
//! Eight-dimensional float vector.
using T8f  = T8<float>;
//! Nine-dimensional float vector.
using T9f  = T9<float>;
//! Ten-dimensional float vector.
using T10f = T10<float>;

//! One-dimensional double vector.
using T1d  = T1<double>;
//! Two-dimensional double vector.
using T2d  = T2<double>;
//! Three-dimensional double vector.
using T3d  = T3<double>;
//! Four-dimensional double vector.
using T4d  = T4<double>;
//! Five-dimensional double vector.
using T5d  = T5<double>;
//! Six-dimensional double vector.
using T6d  = T6<double>;
//! Seven-dimensional double vector.
using T7d  = T7<double>;
//! Eight-dimensional double vector.
using T8d  = T8<double>;
//! Nine-dimensional double vector.
using T9d  = T9<double>;
//! Ten-dimensional double vector.
using T10d = T10<double>;

}  // namespace smooth

#endif  // SMOOTH__TN_HPP_

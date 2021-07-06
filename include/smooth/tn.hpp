#ifndef SMOOTH__TN_HPP_
#define SMOOTH__TN_HPP_

/**
 * @file
 * @brief Lie group of translations.
 */

#include "impl/tn.hpp"
#include "lie_group_base.hpp"

namespace smooth {

/**
 * @brief Eigen type as \f$T(n)\f$ Lie Group.
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

template<typename _Scalar>
using T1 = Tn<1, _Scalar>;

template<typename _Scalar>
using T2 = Tn<2, _Scalar>;

template<typename _Scalar>
using T3 = Tn<3, _Scalar>;

template<typename _Scalar>
using T4 = Tn<4, _Scalar>;

template<typename _Scalar>
using T5 = Tn<5, _Scalar>;

template<typename _Scalar>
using T6 = Tn<6, _Scalar>;

template<typename _Scalar>
using T7 = Tn<7, _Scalar>;

template<typename _Scalar>
using T8 = Tn<8, _Scalar>;

template<typename _Scalar>
using T9 = Tn<9, _Scalar>;

template<typename _Scalar>
using T10 = Tn<10, _Scalar>;

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

}  // namespace smooth

#endif  // SMOOTH__TN_HPP_

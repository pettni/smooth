#ifndef TN_HPP_
#define TN_HPP_

#include "impl/tn.hpp"
#include "lie_group.hpp"

namespace smooth {

template<int _N, typename _Scalar>
using Tn = Eigen::Matrix<_Scalar, _N, 1>;

template<int _N, typename _Scalar>
struct lie_traits<Tn<_N, _Scalar>>
{
  using Impl   = TnImpl<_N, _Scalar>;
  using Scalar = _Scalar;

  template<typename NewScalar>
  using PlainObject = Eigen::Matrix<NewScalar, _N, 1>;
};

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

}  // namespace smooth

#endif  // TN_HPP_

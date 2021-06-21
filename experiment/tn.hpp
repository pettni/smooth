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

}  // namespace smooth

#endif  // TN_HPP_

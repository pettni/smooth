#ifndef SMOOTH__COMMON_HPP_
#define SMOOTH__COMMON_HPP_

namespace smooth
{

template<typename Scalar, int Size>
using DefaultStorage = Eigen::Matrix<Scalar, Size, 1>;

}  // namespace smooth

#endif  // SMOOTH__COMMON_HPP_

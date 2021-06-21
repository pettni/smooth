#ifndef IMPL__COMMON_IMPL_HPP_
#define IMPL__COMMON_IMPL_HPP_

namespace smooth {

static constexpr double eps2 = 1e-8;

#define DEFINE_REFS                                                             \
  using GRefIn  = const Eigen::Ref<const Eigen::Matrix<Scalar, RepSize, 1>> &;  \
  using GRefOut = Eigen::Ref<Eigen::Matrix<Scalar, RepSize, 1>>;                \
                                                                                \
  using TRefIn  = const Eigen::Ref<const Eigen::Matrix<Scalar, Dof, 1>> &;      \
  using TRefOut = Eigen::Ref<Eigen::Matrix<Scalar, Dof, 1>>;                    \
                                                                                \
  using TMapRefIn  = const Eigen::Ref<const Eigen::Matrix<Scalar, Dof, Dof>> &; \
  using TMapRefOut = Eigen::Ref<Eigen::Matrix<Scalar, Dof, Dof>>;

}  // namespace smooth

#endif  // IMPL__COMMON_IMPL_HPP_

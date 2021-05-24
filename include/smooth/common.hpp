#ifndef SMOOTH__COMMON_HPP_
#define SMOOTH__COMMON_HPP_

#include "concepts.hpp"

namespace smooth {

// cutoff points for applying small-angle approximations
static constexpr double eps2 = 1e-8;

// The bundle supports Eigen vector types to represent En, these typedefs
template<typename Scalar>
using R1 = Eigen::Matrix<Scalar, 1, 1>;
template<typename Scalar>
using R2 = Eigen::Matrix<Scalar, 2, 1>;
template<typename Scalar>
using R3 = Eigen::Matrix<Scalar, 3, 1>;
template<typename Scalar>
using R4 = Eigen::Matrix<Scalar, 4, 1>;
template<typename Scalar>
using R5 = Eigen::Matrix<Scalar, 5, 1>;
template<typename Scalar>
using R6 = Eigen::Matrix<Scalar, 6, 1>;
template<typename Scalar>
using R7 = Eigen::Matrix<Scalar, 7, 1>;
template<typename Scalar>
using R8 = Eigen::Matrix<Scalar, 8, 1>;
template<typename Scalar>
using R9 = Eigen::Matrix<Scalar, 9, 1>;
template<typename Scalar>
using R10 = Eigen::Matrix<Scalar, 10, 1>;

using R1f = R1<float>;
using R2f = R2<float>;
using R3f = R3<float>;
using R4f = R4<float>;
using R5f = R5<float>;
using R6f = R6<float>;
using R7f = R7<float>;
using R8f = R8<float>;
using R9f = R9<float>;
using R10f = R10<float>;

using R1d = R1<double>;
using R2d = R2<double>;
using R3d = R3<double>;
using R4d = R4<double>;
using R5d = R5<double>;
using R6d = R6<double>;
using R7d = R7<double>;
using R8d = R8<double>;
using R9d = R9<double>;
using R10d = R10<double>;

// Helper trait to extract relevant properties for lie and en types
namespace detail {

template<typename T>
struct lie_info;

template<LieGroupLike G>
struct lie_info<G>
{
  static constexpr int lie_size   = G::lie_size;
  static constexpr int lie_dof    = G::lie_dof;
  static constexpr int lie_dim    = G::lie_dim;
  static constexpr int lie_actdim = G::lie_actdim;

  static int lie_size_dynamic(const G &) { return lie_size; }
  static int lie_dof_dynamic(const G &) { return lie_dof; }
  static int lie_dim_dynamic(const G &) { return lie_dim; }
  static int lie_actdim_dynamic(const G &) { return lie_actdim; }
};

template<RnLike G>
struct lie_info<G>
{
  static constexpr int lie_size   = G::SizeAtCompileTime;
  static constexpr int lie_dof    = G::SizeAtCompileTime;
  static constexpr int lie_dim    = lie_size == -1 ? -1 : G::SizeAtCompileTime + 1;
  static constexpr int lie_actdim = G::SizeAtCompileTime;

  static int lie_size_dynamic(const G & g) { return g.size(); }
  static int lie_dof_dynamic(const G & g) { return g.size(); }
  static int lie_dim_dynamic(const G & g) { return g.size() + 1; }
  static int lie_actdim_dynamic(const G & g) { return g.size(); }
};

}  // namespace detail

}  // namespace smooth

#endif  // SMOOTH__COMMON_HPP_

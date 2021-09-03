#ifndef SMOOTH__MANIFOLD_HPP_
#define SMOOTH__MANIFOLD_HPP_

#include <Eigen/Core>
#include <concepts>

namespace smooth {

/**
 * @brief Trait class for making a class n AdaptedManifold
 */
template<typename T>
struct man;

// clang-format off

/**
 * @brief Class-external Lie group interface defined via the man trait.
 */
template<typename M>
concept AdaptedManifold =
std::is_default_constructible_v<M> &&
std::is_copy_constructible_v<M> &&
std::is_copy_assignable_v<M> &&
requires {
  typename man<M>::Scalar;
  {man<M>::Dof}->std::convertible_to<Eigen::Index>;
} &&
requires(const M & m1, const M & m2) {
  {man<M>::dof(m1)}->std::convertible_to<Eigen::Index>;
  {man<M>::template cast<double>(m1)};
  {man<M>::template cast<float>(m1)};
  {man<M>::rsub(m1, m2)}->std::convertible_to<Eigen::Matrix<typename man<M>::Scalar, man<M>::Dof, 1>>;
} &&
requires(const M & m, const Eigen::Matrix<typename man<M>::Scalar, man<M>::Dof, 1> & a) {
  {man<M>::rplus(m, a)}->std::convertible_to<M>;
};

// clang-format on

}  // namespace smooth

#endif

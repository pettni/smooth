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
  {man<M>::Dof}->std::convertible_to<Eigen::Index>;
  typename man<M>::Scalar;
} &&
requires(const M & m1, const M & m2) {
  {man<M>::dof(m1)}->std::convertible_to<Eigen::Index>;
  {man<M>::template cast<double>(m1)};
  {man<M>::template cast<float>(m1)};
  {man<M>::rminus(m1, m2)}->std::convertible_to<Eigen::Matrix<typename man<M>::Scalar, man<M>::Dof, 1>>;
} &&
requires(const M & m, const Eigen::Matrix<typename man<M>::Scalar, man<M>::Dof, 1> & a) {
  {man<M>::rplus(m, a)}->std::convertible_to<M>;
};


template<AdaptedManifold M>
static inline constexpr Eigen::Index Dof = man<M>::Dof;

/**
 * @brief Group scalar type
 */
template<AdaptedManifold M>
using Scalar = typename man<M>::Scalar;

/**
 * @brief Cast'ed type
 */
template<AdaptedManifold M, typename NewScalar>
using CastT = decltype(man<M>::template cast<NewScalar>(std::declval<M>()));

/**
 * @brief Degrees of freedom of Lie group
 */
template<AdaptedManifold M>
inline auto dof(const M & m)
{
  return man<M>::dof(m);
}

/**
 * @brief Vector of size Dof
 */
template<AdaptedManifold M>
using Tangent = Eigen::Matrix<typename man<M>::Scalar, man<M>::Dof, 1>;

/**
 * @brief Cast to different scalar type
 */
template<typename NewScalar, AdaptedManifold M>
inline CastT<M, NewScalar> cast(const M & m)
{
  return man<M>::template cast<NewScalar>(m);
}


/**
 * @brief Right-plus
 */
template<AdaptedManifold M, typename Derived>
inline M rplus(const M & m, const Eigen::MatrixBase<Derived> & a)
{
  return man<M>::rplus(m, a);
}

/**
 * @brief Right-minus
 */
template<AdaptedManifold M>
inline Eigen::Matrix<typename man<M>::Scalar, man<M>::Dof, 1> rminus(const M & g1, const M & g2)
{
  return man<M>::rminus(g1, g2);
}

// clang-format on

}  // namespace smooth

#endif

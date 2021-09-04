#ifndef SMOOTH__MANIFOLD_HPP_
#define SMOOTH__MANIFOLD_HPP_

#include <Eigen/Core>
#include <concepts>

#include "lie_group.hpp"

/**
 * @file manifold.hpp Manifold interface
 */

namespace smooth {

/**
 * @brief Trait class for making a class n Manifold
 */
template<typename T>
struct man;

// clang-format off

/**
 * @brief Class-external Lie group interface defined via the man trait.
 */
template<typename M>
concept Manifold =
requires {
  {man<M>::Dof}->std::convertible_to<Eigen::Index>;
  typename man<M>::Scalar;
  typename man<M>::PlainObject;
} &&
requires(const M & m1, const M & m2) {
  {man<M>::dof(m1)}->std::convertible_to<Eigen::Index>;
  {man<M>::rminus(m1, m2)}->std::convertible_to<Eigen::Matrix<typename man<M>::Scalar, man<M>::Dof, 1>>;
} &&
requires(const M & m, const Eigen::Matrix<typename man<M>::Scalar, man<M>::Dof, 1> & a) {
  {man<M>::rplus(m, a)}->std::convertible_to<typename man<M>::PlainObject>;
} && (
  !std::is_convertible_v<typename man<M>::Scalar, double> ||
  requires (const M & m) {
    {man<M>::template cast<double>(m)}->std::convertible_to<typename man<M>::template CastT<double>>;
  }
) && (
  !std::is_convertible_v<typename man<M>::Scalar, float> ||
  requires (const M & m) {
    {man<M>::template cast<float>(m)}->std::convertible_to<typename man<M>::template CastT<float>>;
  }
) &&
std::is_default_constructible_v<typename man<M>::PlainObject> &&
std::is_copy_constructible_v<typename man<M>::PlainObject> &&
std::is_assignable_v<M &, typename man<M>::PlainObject>;

// clang-format on

////////////////////////////////////////////////
//// Free functions that dispatch to man<G> ////
////////////////////////////////////////////////

// Static constants

/**
 * @brief Manifold degrees of freedom (tangent space dimension)
 *
 * @note Equal to -1 for a dynamically sized Manifold
 */
template<Manifold M>
static inline constexpr Eigen::Index Dof = man<M>::Dof;

// Types

/**
 * @brief Manifold scalar type
 */
template<Manifold M>
using Scalar = typename man<M>::Scalar;

/**
 * @brief Manifold default type
 */
template<Manifold M>
using PlainObject = typename man<M>::PlainObject;

/**
 * @brief Cast'ed type
 */
template<typename NewScalar, Manifold M>
using CastT = typename man<M>::template CastT<NewScalar>;

// Functions

/**
 * @brief Manifold degrees of freedom (tangent space dimension)
 */
template<Manifold M>
inline auto dof(const M & m)
{
  return man<M>::dof(m);
}

/**
 * @brief Tangent as a Dof-lenth Eigen vector
 */
template<Manifold M>
using Tangent = Eigen::Matrix<typename man<M>::Scalar, man<M>::Dof, 1>;

/**
 * @brief Cast to different scalar type
 */
template<typename NewScalar, Manifold M>
inline CastT<NewScalar, M> cast(const M & m)
{
  return man<M>::template cast<NewScalar>(m);
}

/**
 * @brief Manifold right-plus
 */
template<Manifold M, typename Derived>
inline M rplus(const M & m, const Eigen::MatrixBase<Derived> & a)
{
  return man<M>::rplus(m, a);
}

/**
 * @brief Manifold right-minus
 */
template<Manifold M>
inline Eigen::Matrix<typename man<M>::Scalar, man<M>::Dof, 1> rminus(const M & g1, const M & g2)
{
  return man<M>::rminus(g1, g2);
}

/**
 * @brief Manifold interface for LieGroup
 */
template<LieGroup G>
struct man<G>
{
  // \cond
  using Scalar      = typename lie<G>::Scalar;
  using PlainObject = typename lie<G>::PlainObject;
  template<typename NewScalar>
  using CastT = typename lie<G>::template CastT<NewScalar>;

  static constexpr Eigen::Index Dof = lie<G>::Dof;

  static inline Eigen::Index dof(const G & g) { return lie<G>::dof(g); }

  template<typename NewScalar>
  static inline auto cast(const G & g)
  {
    return lie<G>::template cast<NewScalar>(g);
  }

  template<typename Derived>
  static inline G rplus(const G & g, const Eigen::MatrixBase<Derived> & a)
  {
    return lie<G>::composition(g, lie<G>::exp(a));
  }

  static inline Eigen::Matrix<Scalar, Dof, 1> rminus(const G & g1, const G & g2)
  {
    return lie<G>::log(lie<G>::composition(lie<G>::inverse(g2), g1));
  }
  // \endcond
};

}  // namespace smooth

#endif

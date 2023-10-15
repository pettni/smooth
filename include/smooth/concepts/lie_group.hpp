// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <concepts>

#include <Eigen/Core>

#include "manifold.hpp"

/**
 * @file lie_group.hpp Internal and external LieGroup interfaces and free LieGroup functions.
 */

SMOOTH_BEGIN_NAMESPACE

namespace traits {

/**
 * @brief Trait class for making a class a LieGroup instance via specialization.
 */
template<typename T>
struct lie;

}  // namespace traits

// clang-format off

/**
 * @brief Class-external Lie group interface defined through the traits::lie trait class.
 */
template<typename G>
concept LieGroup =
requires (Eigen::Index dof) {
  // Underlying scalar type
  typename traits::lie<G>::Scalar;
  // Default representation
  typename traits::lie<G>::PlainObject;
  // Compile-time degrees of freedom (tangent space dimension). Can be dynamic (equal to -1)
  {traits::lie<G>::Dof}->std::convertible_to<int>;
  // Commutativity
  {traits::lie<G>::IsCommutative}->std::convertible_to<bool>;
  // Return the identity element (dof = Dof for static size)
  {traits::lie<G>::Identity(dof)}->std::convertible_to<typename traits::lie<G>::PlainObject>;
  // Return a random element (dof = Dof for static size)
  {traits::lie<G>::Random(dof)}->std::convertible_to<typename traits::lie<G>::PlainObject>;
} &&
// GROUP INTERFACE
requires(const G & g1, const G & g2, typename traits::lie<G>::Scalar eps) {
  // Group adjoint
  {traits::lie<G>::Ad(g1)}->std::convertible_to<Eigen::Matrix<typename traits::lie<G>::Scalar, traits::lie<G>::Dof, traits::lie<G>::Dof>>;
  // Group composition
  {traits::lie<G>::composition(g1, g2)}->std::convertible_to<typename traits::lie<G>::PlainObject>;
  // Run-time degrees of freedom (tangent space dimension).
  {traits::lie<G>::dof(g1)}->std::convertible_to<Eigen::Index>;
  // Group inverse
  {traits::lie<G>::inverse(g1)}->std::convertible_to<typename traits::lie<G>::PlainObject>;
  // Check if two elements are (approximately) equal
  {traits::lie<G>::isApprox(g1, g2, eps)}->std::convertible_to<bool>;
  // Group logarithm (maps from group to algebra)
  {traits::lie<G>::log(g1)}->std::convertible_to<Eigen::Vector<typename traits::lie<G>::Scalar, traits::lie<G>::Dof>>;
} &&
// TANGENT INTERFACE
requires(const Eigen::Vector<typename traits::lie<G>::Scalar, traits::lie<G>::Dof> & a) {
  // Algebra adjoint
  {traits::lie<G>::ad(a)}->std::convertible_to<Eigen::Matrix<typename traits::lie<G>::Scalar, traits::lie<G>::Dof, traits::lie<G>::Dof>>;
  // Algebra exponential (maps from algebra to group)
  {traits::lie<G>::exp(a)}->std::convertible_to<typename traits::lie<G>::PlainObject>;
  // Right derivative of the exponential map
  {traits::lie<G>::dr_exp(a)}->std::convertible_to<Eigen::Matrix<typename traits::lie<G>::Scalar, traits::lie<G>::Dof, traits::lie<G>::Dof>>;
  // Right derivative of the exponential map inverse
  {traits::lie<G>::dr_expinv(a)}->std::convertible_to<Eigen::Matrix<typename traits::lie<G>::Scalar, traits::lie<G>::Dof, traits::lie<G>::Dof>>;
  // Second right derivative of the exponential map
  {traits::lie<G>::d2r_exp(a)}->std::convertible_to<Eigen::Matrix<typename traits::lie<G>::Scalar, traits::lie<G>::Dof, (traits::lie<G>::Dof > 0 ? traits::lie<G>::Dof * traits::lie<G>::Dof : -1)>>;
  // Second right derivative of the exponential map inverse
  {traits::lie<G>::d2r_expinv(a)}->std::convertible_to<Eigen::Matrix<typename traits::lie<G>::Scalar, traits::lie<G>::Dof, (traits::lie<G>::Dof > 0 ? traits::lie<G>::Dof * traits::lie<G>::Dof : -1)>>;
} && (
  // Cast to different scalar type
  !std::is_convertible_v<typename traits::lie<G>::Scalar, double> ||
  requires (const G & g) {
    {traits::lie<G>::template cast<double>(g)};
  }
) && (
  !std::is_convertible_v<typename traits::lie<G>::Scalar, float> ||
  requires (const G & g) {
    {traits::lie<G>::template cast<double>(g)};
  }
) &&
// PlainObject must be default-constructible
std::is_default_constructible_v<typename traits::lie<G>::PlainObject> &&
std::is_copy_constructible_v<typename traits::lie<G>::PlainObject> &&
// PlainObject must be assignable from G
std::is_assignable_v<typename traits::lie<G>::PlainObject &, G>;

namespace traits {

/**
 * @brief Manifold interface for LieGroup types (all LieGroups are also Manifolds).
 */
template<LieGroup G>
struct man<G>
{
  // \cond
  using Scalar      = typename traits::lie<G>::Scalar;
  using PlainObject = typename traits::lie<G>::PlainObject;
  template<typename NewScalar>
  using CastT = typename traits::lie<G>::template CastT<NewScalar>;

  static constexpr int Dof = traits::lie<G>::Dof;

  static inline PlainObject Default(Eigen::Index dof) { return traits::lie<G>::Identity(dof); }

  static inline Eigen::Index dof(const G & g) { return traits::lie<G>::dof(g); }

  template<typename NewScalar>
  static inline CastT<NewScalar> cast(const G & g)
  {
    return traits::lie<G>::template cast<NewScalar>(g);
  }

  template<typename Derived>
  static inline PlainObject rplus(const G & g, const Eigen::MatrixBase<Derived> & a)
  {
    return traits::lie<G>::composition(g, traits::lie<G>::exp(a));
  }

  template<LieGroup Go = G>
  static inline Eigen::Matrix<Scalar, Dof, 1> rminus(const G & g1, const Go & g2)
  {
    return traits::lie<G>::log(traits::lie<Go>::composition(traits::lie<Go>::inverse(g2), g1));
  }
  // \endcond
};

}  // namespace

////////////////////////////////////////////////////////
//// Free functions that dispatch to traits::lie<G> ////
////////////////////////////////////////////////////////

// Group interface

template<LieGroup G>
static constexpr bool IsCommutative = traits::lie<G>::IsCommutative;

/**
 * @brief Identity in Lie group
 *
 * @param dof degrees of freedom
 */
template<LieGroup G>
inline PlainObject<G> Identity(Eigen::Index dof)
{
  return traits::lie<G>::Identity(dof);
}

/**
 * @brief Identity in Lie group with static Dof
 */
template<LieGroup G>
  requires(Dof<G> > 0)
inline PlainObject<G> Identity() { return traits::lie<G>::Identity(Dof<G>); }

/**
 * @brief Random element in Lie group
 *
 * @param dof degrees of freedom
 */
template<LieGroup G>
inline PlainObject<G> Random(Eigen::Index dof)
{
  return traits::lie<G>::Random(dof);
}

/**
 * @brief Random element in Lie group with static Dof
 */
template<LieGroup G>
  requires(Dof<G> > 0)
inline PlainObject<G> Random() { return traits::lie<G>::Random(Dof<G>); }

/**
 * @brief Group adjoint \f$ Ad_g a \coloneq (G * \hat(a) * G^{-1})^{\wedge} \f$
 */
template<LieGroup G>
inline TangentMap<G> Ad(const G & g)
{
  return traits::lie<G>::Ad(g);
}

/**
 * @brief Group binary composition
 */
template<LieGroup G, typename Arg>
inline PlainObject<G> composition(const G & g, Arg && a)
{
  return traits::lie<G>::composition(g, std::forward<Arg>(a));
}

/**
 * @brief Group multinary composition
 */
template<LieGroup G, typename Arg, typename... Args>
inline PlainObject<G> composition(const G & g, Arg && a, Args &&... as)
{
  return composition(composition(g, std::forward<Arg>(a)), std::forward<Args>(as)...);
}

/**
 * @brief Group inverse
 */
template<LieGroup G>
inline PlainObject<G> inverse(const G & g)
{
  return traits::lie<G>::inverse(g);
}

/**
 * @brief Check if two group elements are approximately equal
 */
template<LieGroup G, typename Arg>
inline bool isApprox(
  const G & g,
  Arg && a,
  typename traits::lie<G>::Scalar eps = Eigen::NumTraits<typename traits::lie<G>::Scalar>::dummy_precision())
{
  return traits::lie<G>::isApprox(g, std::forward<Arg>(a), eps);
}

/**
 * @brief Group logarithm
 *
 * @see exp()
 */
template<LieGroup G>
inline Tangent<G> log(const G & g)
{
  return traits::lie<G>::log(g);
}

// Tangent interface

/**
 * @brief Lie algebra adjoint \f$ ad_a b = [a, b] \f$
 */
template<LieGroup G, typename Arg>
inline TangentMap<G> ad(Arg && a)
{
  return traits::lie<G>::ad(std::forward<Arg>(a));
}

/**
 * @brief Lie algebra exponential
 *
 * @see log()
 */
template<LieGroup G, typename Arg>
inline PlainObject<G> exp(Arg && a)
{
  return traits::lie<G>::exp(std::forward<Arg>(a));
}

/**
 * @brief Right Jacobian of exponential map
 */
template<LieGroup G, typename Arg>
inline TangentMap<G> dr_exp(Arg && a)
{
  return traits::lie<G>::dr_exp(std::forward<Arg>(a));
}

/**
 * @brief Right Jacobian of exponential map inverse
 */
template<LieGroup G, typename Arg>
inline TangentMap<G> dr_expinv(Arg && a)
{
  return traits::lie<G>::dr_expinv(std::forward<Arg>(a));
}

/**
 * @brief Right Hessian of exponential map
 */
template<LieGroup G, typename Arg>
inline Hessian<G> d2r_exp(Arg && a)
{
  return traits::lie<G>::d2r_exp(std::forward<Arg>(a));
}

/**
 * @brief Right Hessian of exponential map inverse
 */
template<LieGroup G, typename Arg>
inline Hessian<G> d2r_expinv(Arg && a)
{
  return traits::lie<G>::d2r_expinv(std::forward<Arg>(a));
}

// Convenience methods

/**
 * @brief Left-plus
 */
template<LieGroup G, typename Derived>
inline PlainObject<G> lplus(const G & g, const Eigen::MatrixBase<Derived> & a)
{
  return composition(::smooth::exp<G>(a), g);
}

/**
 * @brief Left-minus
 */
template<LieGroup G, LieGroup Go>
inline Tangent<G> lminus(const G & g1, const Go & g2)
{
  return log(composition(g1, inverse(g2)));
}

/**
 * @brief Left Jacobian of exponential map
 */
template<LieGroup G, typename Derived>
inline TangentMap<G> dl_exp(const Eigen::MatrixBase<Derived> & a)
{
  return dr_exp<G>(-a);
}

/**
 * @brief Left Jacobian of exponential map inverse
 */
template<LieGroup G, typename Derived>
inline TangentMap<G> dl_expinv(const Eigen::MatrixBase<Derived> & a)
{
  return dr_expinv<G>(-a);
}

/**
 * @brief Left Hessian of exponential map
 */
template<LieGroup G, typename Derived>
inline Hessian<G> d2l_exp(const Eigen::MatrixBase<Derived> & a)
{
  return -d2r_exp<G>(-a);
}

/**
 * @brief Left Hessian of exponential map inverse
 */
template<LieGroup G, typename Derived>
inline Hessian<G> d2l_expinv(const Eigen::MatrixBase<Derived> & a)
{
  return -d2r_expinv<G>(-a);
}

SMOOTH_END_NAMESPACE

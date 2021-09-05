// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SMOOTH__MANIFOLD_HPP_
#define SMOOTH__MANIFOLD_HPP_

#include <Eigen/Core>
#include <concepts>

#include "lie_group.hpp"

/**
 * @file manifold.hpp Manifold interface and free Manifold functions.
 */

namespace smooth {

/**
 * @brief Trait class for making a class a Manifold instance via specialization.
 */
template<typename T>
struct man;

// clang-format off

/**
 * @brief Class-external Manifold interface defined through the man trait class.
 */
template<typename M>
concept Manifold =
requires {
  // Underlying scalar type
  typename man<M>::Scalar;
  // Default representation
  typename man<M>::PlainObject;
  // Compile-time degrees of freedom (tangent space dimension). Can be dynamic (equal to -1)
  {man<M>::Dof}->std::convertible_to<Eigen::Index>;
} &&
requires(const M & m1, const M & m2, const Eigen::Matrix<typename man<M>::Scalar, man<M>::Dof, 1> & a) {
  // Run-time degrees of freedom (tangent space dimension)
  {man<M>::dof(m1)}->std::convertible_to<Eigen::Index>;
  // Right-plus: add a tangent vector to a Manifold element to obtain a new Manifold element
  {man<M>::rplus(m1, a)}->std::convertible_to<typename man<M>::PlainObject>;
  // Right-minus: subtract a Manifold element from another to obtain the difference tangent vector
  {man<M>::rminus(m1, m2)}->std::convertible_to<Eigen::Matrix<typename man<M>::Scalar, man<M>::Dof, 1>>;
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
// PlainObject must be default-constructible
std::is_default_constructible_v<typename man<M>::PlainObject> &&
std::is_copy_constructible_v<typename man<M>::PlainObject> &&
// PlainObject must be assignable from M
std::is_assignable_v<typename man<M>::PlainObject &, M>;

// clang-format on

////////////////////////////////////////////////
//// Free functions that dispatch to man<M> ////
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

/**
 * @brief Tangent as a Dof-lenth Eigen vector
 */
template<Manifold M>
using Tangent = Eigen::Matrix<typename man<M>::Scalar, man<M>::Dof, 1>;

// Functions

/**
 * @brief Manifold degrees of freedom (tangent space dimension)
 */
template<Manifold M>
inline Eigen::Index dof(const M & m)
{
  return man<M>::dof(m);
}

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
inline PlainObject<M> rplus(const M & m, const Eigen::MatrixBase<Derived> & a)
{
  return man<M>::rplus(m, a);
}

/**
 * @brief Manifold right-minus
 */
template<Manifold M, Manifold Mo>
inline Tangent<M> rminus(const M & g1, const Mo & g2)
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
  static inline CastT<NewScalar> cast(const G & g)
  {
    return lie<G>::template cast<NewScalar>(g);
  }

  template<typename Derived>
  static inline PlainObject rplus(const G & g, const Eigen::MatrixBase<Derived> & a)
  {
    return lie<G>::composition(g, lie<G>::exp(a));
  }

  template<LieGroup Go = G>
  static inline Eigen::Matrix<Scalar, Dof, 1> rminus(const G & g1, const Go & g2)
  {
    return lie<G>::log(lie<Go>::composition(lie<Go>::inverse(g2), g1));
  }
  // \endcond
};

}  // namespace smooth

#endif

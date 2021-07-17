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

#ifndef SMOOTH__CONCEPTS_HPP_
#define SMOOTH__CONCEPTS_HPP_

/**
 * @file concepts.hpp Library concept definitions.
 */

#include <concepts>

#include <Eigen/Core>


namespace smooth
{

/**
 * @brief A concept defining a (smooth) manifold.
 *
 * - `M::Scalar` scalar type
 * - `M::SizeAtCompileTime` tangent space dimension (compile time, -1 if dynamic)
 * - `M.size()` : tangent space dimension (runtime)
 * - `M + T -> M` : geodesic addition
 * - `M - M -> T` : inverse of geodesic addition (in practice only used for infinitesimal values)
 *
 * Where `T = Eigen::Matrix<Scalar, SizeAtCompileTime, 1>` is the tangent type
 */
template<typename M>
concept Manifold =
requires
{
  typename M::Scalar;
  typename M::PlainObject;
  {M::SizeAtCompileTime}->std::convertible_to<Eigen::Index>;  // degrees of freedom at compile time
} &&
requires(const M & m1, const M & m2, const Eigen::Matrix<typename M::Scalar, M::SizeAtCompileTime, 1> & a)
{
  {m1.size()}->std::convertible_to<Eigen::Index>;             // degrees of freedom at runtime
  {m1 + a}->std::convertible_to<typename M::PlainObject>;
  {m1 - m2}->std::convertible_to<Eigen::Matrix<typename M::Scalar, M::SizeAtCompileTime, 1>>;
  {m1.template cast<double>()};
};

/**
 * @brief Lie group concept.
 *
 * Requires the exp and log maps, and the upper and lowercase adjoints.
 */
template<typename G>
concept LieGroup = Manifold<G> &&
// static constants
requires {
  {G::Dof}->std::convertible_to<Eigen::Index>;      // degrees of freedom
} &&
(G::Dof >= 1) &&
(G::SizeAtCompileTime == G::Dof) &&
// member methods
requires(const G & g1, const G & g2)
{
  {g1.inverse()}->std::convertible_to<typename G::PlainObject>;
  {g1 * g2}->std::convertible_to<typename G::PlainObject>;
  {g1.log()}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, 1>>;
  {g1.Ad()}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
} &&
// static methods
requires(const Eigen::Matrix<typename G::Scalar, G::Dof, 1> & a)
{
  {G::Identity()}->std::convertible_to<typename G::PlainObject>;
  {G::exp(a)}->std::convertible_to<typename G::PlainObject>;
  {G::ad(a)}->std::convertible_to<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof>>;
};

/**
 * @brief Type trait that maps a type to Lie group operations.
 *
 * Required members:
 *  - `typename Impl`: Lie group operations
 *  - `typename Scalar`: Scalar type (\p float, \p double, ...)
 *  - `typename template<NewScalar> PlainObject`: Default return type
 */
template<typename T>
struct lie_traits;

/**
 * @brief Type for which \p lie_traits is properly specialized.
 */

} // namespace smooth

#endif  // SMOOTH__CONCEPTS_HPP_

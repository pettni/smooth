// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Differentiation on Manifolds.
 */

#include <utility>

#include "manifolds.hpp"
#include "wrt.hpp"

namespace smooth {
inline namespace v1_0 {
namespace diff {
/**
 * @brief Available differentiation methods
 */
enum class Type {
  Numerical,  ///< Numerical (forward) derivatives
  Autodiff,   ///< Uses the autodiff (https://autodiff.github.io) library; requires  \p
              ///< compat/autodiff.hpp
  Ceres,      ///< Uses the Ceres (http://ceres-solver.org) built-in autodiff; requires \p
              ///< compat/ceres.hpp
  Analytic,   ///< Hand-coded derivative. Type must have a function named 'jacobian' : x -> Mat
              ///< (order 1) and 'hesssian': x -> Mat (order 2) that compute the derivatives.
  Default     ///< Select based on availability (Analytic > Autodiff > Ceres > Numerical)
};

/**
 * @brief Differentiation in tangent space
 *
 * @tparam K differentiation order (0, 1 or 2)
 * @tparam D differentiation method to use
 *
 * First derivatives are returned as a matrix df s.t.
 * df(i, j) = dfi / dxj, where fi is the i:th degree of freedom of f, and xj the j:th degree of
 * freedom of x.
 *
 * Second derivatives are stored as a horizontally stacked block matrix
 * d2f = [ d2f0 d2f1 ... d2fN ],
 * where d2fi(j, k) = d2fi / dxjxk is the Hessian of the i:th degree of freedom of f.
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return {f(x)} for K = 0, {f(x), dr f(x)} for K = 1, {f(x), dr f(x), d2r f(x)} for K = 2
 *
 * @note Only scalar functions suppored for K = 2
 *
 * @note All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the Manifold concept.
 */
template<std::size_t K, Type D>
auto dr(auto && f, auto && x);

/**
 * @brief Differentiation in tangent space.
 *
 * Like above, but calculate a subset idx of derivatives.
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @param idx indices defining subset of x
 */
template<std::size_t K, Type D, std::size_t... Idx>
auto dr(auto && f, auto && x, std::index_sequence<Idx...> idx);

/**
 * @brief Differentiation in tangent space using default method
 *
 * @tparam K differentiation order
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 *
 * @note All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the Manifold concept.
 */
template<std::size_t K>
auto dr(auto && f, auto && x);

/**
 * @brief Differentiation in tangent space using default method.
 *
 * Like above, but calculate a subset idx of derivatives.
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @param idx indices defining subset of x
 */
template<std::size_t K, std::size_t... Idx>
auto dr(auto && f, auto && x, std::index_sequence<Idx...> idx);

}  // namespace diff
}  // namespace v1_0
}  // namespace smooth

#include "detail/diff_impl.hpp"

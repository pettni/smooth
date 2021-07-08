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

#ifndef SMOOTH__DIFF_HPP_
#define SMOOTH__DIFF_HPP_

/**
 * @file
 * @brief Differentiation on Lie groups.
 */

#include <Eigen/Core>
#include <type_traits>

#include "concepts.hpp"
#include "internal/utils.hpp"

namespace smooth {

/**
 * @brief Grouping of function arguments.
 *
 * A tuple of references is created from the input arguments,
 * which is the expected format in e.g. dr() and minimize().
 */
template<typename... _Args>
auto wrt(_Args &&... args)
{
  return std::forward_as_tuple(std::forward<_Args>(args)...);
}

namespace diff {
namespace detail {

/**
 * @brief Numerical differentiation in tangent space.
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 *
 * All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the Manifold concept.
 */
template<typename _F, typename _Wrt>
auto dr_numerical(_F && f, _Wrt && x)
{
  using Result = decltype(std::apply(f, x));
  using Scalar = typename Result::Scalar;

  // static sizes
  static constexpr Eigen::Index Nx = utils::tuple_dof<std::decay_t<_Wrt>>::value;
  static constexpr Eigen::Index Ny = Result::SizeAtCompileTime;

  const Scalar eps = std::sqrt(Eigen::NumTraits<Scalar>::epsilon());

  auto val = std::apply(f, x);

  // dynamic sizes
  Eigen::Index nx = std::apply([](auto &&... args) { return (args.size() + ...); }, x);
  Eigen::Index ny = val.size();

  // output variable
  Eigen::Matrix<Scalar, Ny, Nx> jac(ny, nx);

  Eigen::Index index_pos = 0;

  utils::static_for<std::tuple_size_v<std::decay_t<_Wrt>>>([&](auto i) {
    static constexpr Eigen::Index Nx_j =
      std::decay_t<std::tuple_element_t<i, std::decay_t<_Wrt>>>::SizeAtCompileTime;
    auto & w       = std::get<i>(x);
    const int nx_j = w.size();
    for (auto j = 0; j != nx_j; ++j) {
      Scalar eps_j = eps;
      if constexpr (RnLike<std::decay_t<decltype(w)>>) {
        // scale step size if we are in Rn
        eps_j *= abs(w[j]);
        if (eps_j == 0.) { eps_j = eps; }
      }
      w += (eps_j * Eigen::Matrix<Scalar, Nx_j, 1>::Unit(nx_j, j));
      jac.col(index_pos + j) = (std::apply(f, x) - val) / eps_j;
      w += (-eps_j * Eigen::Matrix<Scalar, Nx_j, 1>::Unit(nx_j, j));
    }
    index_pos += nx_j;
  });

  return std::make_pair(val, jac);
}

}  // namespace detail

/**
 * @brief Differentiation methods
 *
 * - \p NUMERICAL calculates numerical (forward) derivatives
 * - \p AUTODIFF uses the autodiff (https://autodiff.github.io) library and requires the
 * autodiff compatability header
 * - \p CERES uses the Ceres (http://ceres-solver.org) built-in autodiff and requires the
 * Ceres compatability header
 * - \p ANALYTIC is meant for hand-coded derivatives, and requires that
 * the function returns a pair \f$( f(x), \mathrm{d}^r f_x )\f$
 * - \p DEFAULT selects from a priority list based on availability
 */
enum class Type { NUMERICAL, AUTODIFF, CERES, ANALYTIC, DEFAULT };

/**
 * @brief Differentiation in local tangent space (right derivative)
 *
 * @tparam dm differentiation method to use
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 *
 * All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the Manifold concept.
 */
template<Type dm, typename _F, typename _Wrt>
auto dr(_F && f, _Wrt && x)
{
  if constexpr (dm == Type::NUMERICAL) {
    return detail::dr_numerical(std::forward<_F>(f), std::forward<_Wrt>(x));
  } else if constexpr (dm == Type::AUTODIFF) {
#ifdef SMOOTH_DIFF_AUTODIFF
    return dr_autodiff(std::forward<_F>(f), std::forward<_Wrt>(x));
#else
    static_assert(dm != Type::AUTODIFF, "compat/autodiff.hpp header not included");
#endif
  } else if constexpr (dm == Type::CERES) {
#ifdef SMOOTH_DIFF_CERES
    return dr_ceres(std::forward<_F>(f), std::forward<_Wrt>(x));
#else
    static_assert(dm != Type::CERES, "compat/ceres.hpp header not included");
#endif
  } else if constexpr (dm == Type::ANALYTIC) {
    return std::apply(f, std::forward<_Wrt>(x));
  } else if constexpr (dm == Type::DEFAULT) {
#ifdef SMOOTH_DIFF_AUTODIFF
    return dr_autodiff(std::forward<_F>(f), std::forward<_Wrt>(x));
#elif SMOOTH_DIFF_CERES
    return dr_ceres(std::forward<_F>(f), std::forward<_Wrt>(x));
#else
    return detail::dr_numerical(std::forward<_F>(f), std::forward<_Wrt>(x));
#endif
  }
}

/**
 * @brief Differentiation in local tangent space using default method
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 *
 * All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the Manifold concept.
 */
template<typename _F, typename _Wrt>
auto dr(_F && f, _Wrt && x)
{
  return dr<Type::DEFAULT>(std::forward<_F>(f), std::forward<_Wrt>(x));
}

}  // namespace diff
}  // namespace smooth

#endif  // SMOOTH__DIFF_HPP_

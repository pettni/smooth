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

#include "internal/utils.hpp"
#include "manifold.hpp"
#include "adapted_lie_group.hpp"
#include "tn.hpp"

namespace smooth {

/**
 * @brief Grouping of function arguments.
 *
 * A tuple of references is created from the input arguments,
 * which is the expected format in e.g. dr() and minimize().
 */
template<typename... _Args>
  requires(AdaptedManifold<std::decay_t<_Args>> &&...)
auto wrt(_Args &&... args) { return std::forward_as_tuple(std::forward<_Args>(args)...); }

// differentiation module
namespace diff {
namespace detail {

/**
 * @brief Numerical differentiation in tangent space.
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 *
 * @note All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the AdaptedLieGroup concept.
 */
template<typename _F, typename _Wrt>
auto dr_numerical(_F && f, _Wrt && x)
{
  using Result = decltype(std::apply(f, x));
  using Scalar = typename man<Result>::Scalar;

  static_assert(AdaptedManifold<Result>, "f(x) is not an AdaptedManifold");

  // arguments are modified below, so we create a copy of those that come in as const
  auto x_nc = utils::tuple_copy_if_const(std::forward<_Wrt>(x));

  // static sizes
  static constexpr Eigen::Index Nx = utils::tuple_dof<std::decay_t<_Wrt>>::value;
  static constexpr Eigen::Index Ny = man<Result>::Dof;

  const Scalar eps = std::sqrt(Eigen::NumTraits<Scalar>::epsilon());

  const Result val = std::apply(f, x_nc);

  // dynamic sizes
  Eigen::Index nx = std::apply(
    [](auto &&... args) { return (man<std::decay_t<decltype(args)>>::dof(args) + ...); }, x_nc);
  Eigen::Index ny = man<Result>::dof(val);

  // output variable
  Eigen::Matrix<Scalar, Ny, Nx> jac(ny, nx);

  Eigen::Index index_pos = 0;

  utils::static_for<std::tuple_size_v<std::decay_t<_Wrt>>>([&](auto i) {
    auto & w = std::get<i>(x_nc);
    using W  = std::decay_t<decltype(w)>;

    static constexpr Eigen::Index Nx_j = man<W>::Dof;
    const int nx_j                     = man<W>::dof(w);

    for (auto j = 0; j != nx_j; ++j) {
      Scalar eps_j = eps;
      if constexpr (std::is_base_of_v<Eigen::MatrixBase<W>, W>) {
        // scale step size if we are in Rn
        eps_j *= abs(w[j]);
        if (eps_j == 0.) { eps_j = eps; }
      } else if constexpr (std::is_base_of_v<smooth::TnBase<W>, W>) {
        // or Tn
        eps_j *= abs(w.rn()[j]);
        if (eps_j == 0.) { eps_j = eps; }
      }
      w = man<W>::rplus(w, (eps_j * Eigen::Matrix<Scalar, Nx_j, 1>::Unit(nx_j, j)).eval());
      jac.col(index_pos + j) = man<Result>::rsub(std::apply(f, x_nc), val) / eps_j;
      w = man<W>::rplus(w, (-eps_j * Eigen::Matrix<Scalar, Nx_j, 1>::Unit(nx_j, j)).eval());
    }
    index_pos += nx_j;
  });

  return std::make_pair(val, jac);
}

}  // namespace detail

/**
 * @enum smooth::diff::Type
 * @brief Differentiation methods
 */
enum class Type {
  NUMERICAL,  ///< Numerical (forward) derivatives
  AUTODIFF,   ///< Uses the autodiff (https://autodiff.github.io) library; requires  \p
              ///< compat/autodiff.hpp
  CERES,      ///< Uses the Ceres (http://ceres-solver.org) built-in autodiff; requires \p
              ///< compat/ceres.hpp
  ANALYTIC,   ///< Hand-coded derivative, requires that function returns \p std::pair \f$(f(x),
              ///< \mathrm{d}^r f_x) \f$
  DEFAULT     ///< Automatically select type based on availability
};

static constexpr Type DefaultType =
#ifdef SMOOTH_DIFF_AUTODIFF
  Type::AUTODIFF;
#elif defined SMOOTH_DIFF_CERES
  Type::CERES;
#else
  Type::NUMERICAL;
#endif

/**
 * @brief Differentiation in tangent space
 *
 * @tparam dm differentiation method to use
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 *
 * @note All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the AdaptedLieGroup concept.
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
    return dr<DefaultType>(std::forward<_F>(f), std::forward<_Wrt>(x));
  }
}

/**
 * @brief Differentiation in tangent space using default method
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 *
 * @note All arguments in x as well as the return type \f$f(x)\f$ must satisfy
 * the AdaptedLieGroup concept.
 */
template<typename _F, typename _Wrt>
auto dr(_F && f, _Wrt && x)
{
  return dr<Type::DEFAULT>(std::forward<_F>(f), std::forward<_Wrt>(x));
}

}  // namespace diff
}  // namespace smooth

#endif  // SMOOTH__DIFF_HPP_

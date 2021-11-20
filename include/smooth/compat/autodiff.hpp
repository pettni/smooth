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

#ifndef SMOOTH__COMPAT__AUTODIFF_HPP_
#define SMOOTH__COMPAT__AUTODIFF_HPP_

/**
 * @file
 * @brief autodiff compatability header.
 */

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#define SMOOTH_DIFF_AUTODIFF

#include "smooth/internal/utils.hpp"
#include "smooth/manifold.hpp"
#include "smooth/wrt.hpp"

namespace smooth {

/// @brief Specialize trait to make autodiff type a Manifold
template<std::floating_point F>
struct traits::scalar_trait<autodiff::Dual<F, F>>
{
  static constexpr bool value = true;
};

namespace diff {

/**
 * @brief Automatic differentiation in tangent space using the autodiff library.
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 */
template<typename _F, typename _Wrt>
auto dr_autodiff(_F && f, _Wrt && x)
{
  using Result = decltype(std::apply(f, x));
  using Scalar = ::smooth::Scalar<Result>;

  static_assert(Manifold<Result>, "f(x) is not a Manifold");

  using AdScalar = autodiff::Dual<Scalar, Scalar>;

  Result fval = std::apply(f, x);

  static constexpr Eigen::Index Nx = wrt_dof<_Wrt>();
  static constexpr Eigen::Index Ny = Dof<Result>;
  const Eigen::Index nx = std::apply([](auto &&... args) { return (dof(args) + ...); }, x);

  // cast fval and x to ad types
  const auto x_ad                       = wrt_cast<AdScalar>(x);
  const CastT<AdScalar, Result> fval_ad = cast<AdScalar>(fval);

  // zero-valued tangent element
  Eigen::Matrix<AdScalar, Nx, 1> a_ad = Eigen::Matrix<AdScalar, Nx, 1>::Zero(nx);

  Eigen::Matrix<Scalar, Ny, Nx> jac = autodiff::jacobian(
    [&f, &fval_ad, &x_ad](Eigen::Matrix<AdScalar, Nx, 1> & var) -> Eigen::Matrix<AdScalar, Ny, 1> {
      return rminus<CastT<AdScalar, Result>>(std::apply(f, wrt_rplus(x_ad, var)), fval_ad);
    },
    autodiff::wrt(a_ad),
    autodiff::at(a_ad));

  return std::make_pair(std::move(fval), std::move(jac));
}

}  // namespace diff
}  // namespace smooth

#endif  // SMOOTH__COMPAT__AUTODIFF_HPP_

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

// TODO switch to autodiff::Real when it supports atan2
// https://github.com/autodiff/autodiff/issues/185

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#define SMOOTH_DIFF_AUTODIFF

#include "smooth/internal/utils.hpp"
#include "smooth/manifold.hpp"
#include "smooth/wrt.hpp"

namespace smooth {

/// @brief Specialize trait to make autodiff type a Manifold
template<typename T>
struct traits::scalar_trait<autodiff::Dual<T, T>>
{
  // \cond
  static constexpr bool value = true;
  // \endcond
};

namespace diff {

/**
 * @brief Automatic differentiation in tangent space using the autodiff library.
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 */
template<std::size_t K = 1>
  requires(K >= 1 && K <= 2)
auto dr_autodiff(auto && f, auto && x)
{
  using Result = decltype(std::apply(f, x));
  using Scalar = ::smooth::Scalar<Result>;
  using Eigen::Matrix;

  static_assert(Manifold<Result>, "f(x) is not a Manifold");

  using AdScalar = autodiff::HigherOrderDual<K, Scalar>;

  Result F = std::apply(f, x);

  static constexpr Eigen::Index Nx = wrt_Dof<decltype(x)>();
  static constexpr Eigen::Index Ny = Dof<Result>;
  const Eigen::Index nx = std::apply([](auto &&... args) { return (dof(args) + ...); }, x);
  const Eigen::Index ny = dof(F);

  // cast F and x to ad types
  const auto x_ad              = wrt_cast<AdScalar>(x);
  CastT<AdScalar, Result> F_ad = cast<AdScalar>(F);

  if constexpr (K == 1) {
    // zero-valued tangent element
    Matrix<AdScalar, Nx, 1> a_ad = Matrix<AdScalar, Nx, 1>::Zero(nx);

    // function to differentiate
    const auto f_ad = [&](Matrix<AdScalar, Nx, 1> & var) -> Matrix<AdScalar, Ny, 1> {
      return rminus<CastT<AdScalar, Result>>(std::apply(f, wrt_rplus(x_ad, var)), F_ad);
    };

    Matrix<Scalar, Ny, Nx> J(ny, nx);

    J = autodiff::jacobian(f_ad, autodiff::wrt(a_ad), autodiff::at(a_ad));
    return std::make_pair(std::move(F), std::move(J));
  } else if constexpr (K == 2) {
    // function to differentiate
    const auto f_ad = [&f, &x_ad, &F_ad](
                        Matrix<AdScalar, Nx, 1> & var1,
                        Matrix<AdScalar, Nx, 1> & var2) -> Matrix<AdScalar, Ny, 1> {
      return rminus(std::apply(f, wrt_rplus(wrt_rplus(x_ad, var1), var2)), F_ad);
    };

    Matrix<Scalar, Ny, Nx> J(ny, nx);
    Matrix<Scalar, Nx, std::min(Nx, Ny) == -1 ? -1 : Nx * Ny> H(nx, nx * ny);

    // zero-valued tangent elements
    Matrix<AdScalar, Nx, 1> a_ad1 = Matrix<AdScalar, Nx, 1>::Zero(nx);
    Matrix<AdScalar, Nx, 1> a_ad2 = Matrix<AdScalar, Nx, 1>::Zero(nx);

    const auto a_wrt1 = autodiff::wrt(a_ad1);
    const auto a_wrt2 = autodiff::wrt(a_ad2);

    autodiff::detail::ForEachWrtVar(
      a_wrt1, [&](auto && i, auto && xi) constexpr {
        autodiff::detail::ForEachWrtVar(
          a_wrt2, [&](auto && j, auto && xj) constexpr {
            const auto u =
              autodiff::detail::eval(f_ad, autodiff::at(a_ad1, a_ad2), autodiff::wrt(xi, xj));
            for (auto k = 0u; k < ny; ++k) {
              J(k, i)          = static_cast<double>(autodiff::detail::derivative<1>(u[k]));
              H(j, k * nx + i) = autodiff::detail::derivative<2>(u[k]);
            }
          });
      });

    return std::make_tuple(std::move(F), std::move(J), std::move(H));
  }
}

}  // namespace diff
}  // namespace smooth

#endif  // SMOOTH__COMPAT__AUTODIFF_HPP_

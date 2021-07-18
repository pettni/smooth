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

#include "smooth/concepts.hpp"
#include "smooth/internal/utils.hpp"

namespace smooth::diff {

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
  using Result   = typename decltype(std::apply(f, x))::PlainObject;
  using Scalar   = typename Result::Scalar;
  using AdScalar = autodiff::Dual<Scalar, Scalar>;

  // determine sizes if input and output
  constexpr auto Nx        = utils::tuple_dof<_Wrt>::value;
  auto nx                  = std::apply([](auto &&... args) { return (args.size() + ...); }, x);
  static constexpr auto Ny = Result::SizeAtCompileTime;

  const Result val = std::apply(f, x);

  // cast val and x to ad types
  auto x_ad = utils::tuple_cast<AdScalar>(x);
  const typename decltype(val.template cast<AdScalar>())::PlainObject val_ad =
    val.template cast<AdScalar>();

  // zero-valued tangent element
  Eigen::Matrix<AdScalar, Nx, 1> a_ad = Eigen::Matrix<AdScalar, Nx, 1>::Zero(nx);

  Eigen::Matrix<Scalar, Ny, Nx> jac = autodiff::jacobian(
    [&f, &val_ad, &x_ad](Eigen::Matrix<AdScalar, Nx, 1> & var) -> Eigen::Matrix<AdScalar, Ny, 1> {
      return std::apply(f, utils::tuple_plus(x_ad, var)) - val_ad;
    },
    autodiff::wrt(a_ad),
    autodiff::at(a_ad));

  return std::make_pair(val, jac);
}

}  // namespace smooth::diff

#endif  // SMOOTH__COMPAT__AUTODIFF_HPP_

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

#ifndef SMOOTH__COMPAT__CERES_HPP_
#define SMOOTH__COMPAT__CERES_HPP_

/**
 * @file
 * @brief ceres compatability header.
 */

#include <ceres/autodiff_local_parameterization.h>
#include <ceres/internal/autodiff.h>

#define SMOOTH_DIFF_CERES

#include "smooth/internal/utils.hpp"
#include "smooth/lie_group.hpp"
#include "smooth/manifold.hpp"
#include "smooth/map.hpp"

namespace smooth {

// \cond
template<NativeLieGroup G>
struct CeresParamFunctor
{
  template<typename Scalar>
  bool operator()(const Scalar * x, const Scalar * delta, Scalar * x_plus_delta) const
  {
    using GCast = typename G::template PlainObjectCast<Scalar>;

    smooth::Map<const GCast> mx(x);
    Eigen::Map<const Tangent<GCast>> mdelta(delta);
    smooth::Map<GCast> mx_plus_delta(x_plus_delta);

    mx_plus_delta = mx * GCast::exp(mdelta);
    return true;
  }
};
// \endcond

/**
 * @brief Parameterization for on-manifold optimization with Ceres.
 */
template<Manifold G>
class CeresLocalParameterization
    : public ceres::AutoDiffLocalParameterization<CeresParamFunctor<G>, G::RepSize, G::Dof>
{};

/**
 * @brief Automatic differentiation in tangent space
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 */
template<typename _F, typename _Wrt>
auto dr_ceres(_F && f, _Wrt && x)
{
  // There is potential to improve thie speed of this by reducing casting.
  // The ceres Jet type supports binary operations with e.g. double, but currently
  // the Lie operations require everything to have a uniform scalar type. Enabling
  // plus and minus for different scalars would thus save some casts.
  using Result = decltype(std::apply(f, x));
  using Scalar = ::smooth::Scalar<Result>;

  static_assert(Manifold<Result>, "f(x) is not an Manifold");

  Result fval = std::apply(f, x);

  static constexpr Eigen::Index Nx = utils::tuple_dof<_Wrt>::value;
  static constexpr Eigen::Index Ny = Dof<Result>;
  const Eigen::Index nx = std::apply([](auto &&... args) { return (dof(args) + ...); }, x);
  const Eigen::Index ny = dof<Result>(fval);

  static_assert(Nx != -1, "Ceres autodiff does not support dynamic sizes");

  Eigen::Matrix<Scalar, Nx, 1> a = Eigen::Matrix<Scalar, Nx, 1>::Zero(nx);
  Eigen::Matrix<Scalar, Ny, 1> b(ny);
  Eigen::Matrix<Scalar, Ny, Nx, (Nx == 1) ? Eigen::ColMajor : Eigen::RowMajor> jac(ny, nx);

  const auto f_deriv = [&]<typename T>(const T * in, T * out) {
    Eigen::Map<const Eigen::Matrix<T, Nx, 1>> mi(in, nx);
    Eigen::Map<Eigen::Matrix<T, Ny, 1>> mo(out, ny);
    mo = rminus<CastT<T, Result>>(
      std::apply(f, utils::tuple_plus(utils::tuple_cast<T>(x), mi)), cast<T, Result>(fval));
    return true;
  };
  const Scalar * a_ptr[1] = {a.data()};
  Scalar * jac_ptr[1]     = {jac.data()};

  ceres::internal::AutoDifferentiate<Dof<Result>, ceres::internal::StaticParameterDims<Nx>>(
    f_deriv, a_ptr, b.size(), b.data(), jac_ptr);

  return std::make_pair(std::move(fval), Eigen::Matrix<Scalar, Ny, Nx>(jac));
}

}  // namespace smooth

#endif  // SMOOTH__COMPAT__CERES_HPP_

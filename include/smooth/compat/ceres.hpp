// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief ceres compatability header.
 */

#include <ceres/autodiff_manifold.h>
#include <ceres/internal/autodiff.h>

#define SMOOTH_DIFF_CERES

#include "smooth/detail/traits.hpp"
#include "smooth/detail/wrt_impl.hpp"
#include "smooth/lie_group_base.hpp"
#include "smooth/manifolds.hpp"
#include "smooth/wrt.hpp"

SMOOTH_BEGIN_NAMESPACE

// mark Jet as a valid scalar
template<typename T, int I>
struct detail::scalar_trait<ceres::Jet<T, I>>
{
  static constexpr bool value = true;
};

// \cond
template<Manifold G>
struct CeresParamFunctor
{
  template<typename Scalar>
  bool Plus(const Scalar * x, const Scalar * delta, Scalar * x_plus_delta) const
  {
    smooth::MapDispatch<const CastT<Scalar, G>> mx(x);
    Eigen::Map<const Tangent<CastT<Scalar, G>>> mdelta(delta);
    smooth::MapDispatch<CastT<Scalar, G>> mx_plus_delta(x_plus_delta);

    mx_plus_delta = rplus(mx, mdelta);

    return true;
  }

  template<typename Scalar>
  bool Minus(const Scalar * x, const Scalar * y, Scalar * x_minus_y) const
  {
    smooth::MapDispatch<const CastT<Scalar, G>> mx(x);
    smooth::MapDispatch<const CastT<Scalar, G>> my(y);
    Eigen::Map<Tangent<CastT<Scalar, G>>> m_x_minus_y(x_minus_y);

    m_x_minus_y = rminus<CastT<Scalar, G>, CastT<Scalar, G>>(mx, my);

    return true;
  }
};
// \endcond

/**
 * @brief Parameterization for on-manifold optimization with Ceres.
 */
template<Manifold G>
using CeresLocalParameterization = ceres::AutoDiffManifold<CeresParamFunctor<G>, G::RepSize, G::Dof>;

/**
 * @brief Automatic differentiation in tangent space
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 */
auto dr_ceres(auto && f, auto && x)
{
  // There is potential to improve thie speed of this by reducing casting.
  // The ceres Jet type supports binary operations with e.g. double, but currently
  // the Lie operations require everything to have a uniform scalar type. Enabling
  // plus and minus for different scalars would thus save some casts.
  using Result = decltype(std::apply(f, x));
  using Scalar = ::smooth::Scalar<Result>;

  static_assert(Manifold<Result>, "f(x) is not a Manifold");

  Result fval = std::apply(f, x);

  static constexpr Eigen::Index Nx = wrt_Dof<decltype(x)>();
  static constexpr Eigen::Index Ny = Dof<Result>;
  const Eigen::Index nx            = std::apply([](auto &&... args) { return (dof(args) + ...); }, x);
  const Eigen::Index ny            = dof<Result>(fval);

  static_assert(Nx > 0, "Ceres autodiff does not support dynamic sizes");

  Eigen::Matrix<Scalar, Nx, 1> a = Eigen::Matrix<Scalar, Nx, 1>::Zero(nx);
  Eigen::Matrix<Scalar, Ny, 1> b(ny);
  Eigen::Matrix<Scalar, Ny, Nx, (Nx == 1) ? Eigen::ColMajor : Eigen::RowMajor> jac(ny, nx);
  jac.setZero();

  const auto f_deriv = [&]<typename T>(const T * in, T * out) {
    Eigen::Map<const Eigen::Matrix<T, Nx, 1>> mi(in, nx);
    Eigen::Map<Eigen::Matrix<T, Ny, 1>> mo(out, ny);
    mo = rminus<CastT<T, Result>>(std::apply(f, wrt_rplus(wrt_cast<T>(x), mi)), cast<T, Result>(fval));
    return true;
  };
  const Scalar * a_ptr[1] = {a.data()};
  Scalar * jac_ptr[1]     = {jac.data()};

  ceres::internal::AutoDifferentiate<Dof<Result>, ceres::internal::StaticParameterDims<Nx>>(
    f_deriv, a_ptr, static_cast<int>(b.size()), b.data(), jac_ptr);

  return std::make_pair(std::move(fval), Eigen::Matrix<Scalar, Ny, Nx>(jac));
}

SMOOTH_END_NAMESPACE

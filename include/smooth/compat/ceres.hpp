#ifndef SMOOTH__COMPAT__CERES_HPP_
#define SMOOTH__COMPAT__CERES_HPP_

/**
 * @file
 * @brief ceres compatability header.
 */

#include <ceres/autodiff_local_parameterization.h>
#include <ceres/internal/autodiff.h>

#define SMOOTH_DIFF_CERES

#include "smooth/concepts.hpp"
#include "smooth/internal/utils.hpp"

namespace smooth {

/**
 * @brief Functor defining on-manifold plus for a Lie group type in Ceres format.
 */
template<LieGroup G>
struct ParameterizationFunctor
{
  /**
   * @brief Plus operation.
   */
  template<typename Scalar>
  bool operator()(const Scalar * x, const Scalar * delta, Scalar * x_plus_delta) const
  {
    using GScalar = decltype(G().template cast<Scalar>());

    Eigen::Map<const GScalar> mx(x);
    Eigen::Map<const typename GScalar::Tangent> mdelta(delta);
    Eigen::Map<GScalar> mx_plus_delta(x_plus_delta);

    mx_plus_delta = mx + mdelta;
    return true;
  }
};

/**
 * @brief Define a parameterization using ParameterizationFunctor and automatic differentiation.
 */
template<LieGroup G>
class LieGroupParameterization
    : public ceres::AutoDiffLocalParameterization<ParameterizationFunctor<G>, G::RepSize, G::Dof>
{};

/**
 * @brief Automatic differentiation in tangent space
 *
 * @param f function to differentiate
 * @param x reference tuple of function arguments
 * @return \p std::pair containing value and right derivative: \f$(f(x), \mathrm{d}^r f_x)\f$
 *
 * \note There is potential to improve thie speed of this by reducing casting.
 * The ceres Jet type supports binary operations with e.g. double, but currently
 * the Lie operations require everything to have a uniform scalar type. Enabling
 * + and - for different scalars would thus save some casts.
 */
template<typename _F, typename _Wrt>
auto dr_ceres(_F && f, _Wrt && x)
{
  using Result = decltype(std::apply(f, x));
  using Scalar = typename Result::Scalar;

  const auto fval = std::apply(f, x);

  static constexpr Eigen::Index Nx = utils::tuple_dof<_Wrt>::value;
  const auto nx = std::apply([](auto &&... args) { return (args.size() + ...); }, x);
  static constexpr Eigen::Index Ny = Result::SizeAtCompileTime;
  const auto ny                    = fval.size();

  Eigen::Matrix<Scalar, Nx, 1> a(nx);
  a.setZero();

  Eigen::Matrix<Scalar, Ny, 1> b(ny);

  Eigen::Matrix<Scalar, Ny, Nx, (Nx == 1) ? Eigen::ColMajor : Eigen::RowMajor> jac(ny, nx);

  const Scalar * prms[1] = {a.data()};
  double ** jac_rows     = new double *[ny];
  for (auto i = 0; i < ny; i++) { jac_rows[i] = jac.row(i).data(); }

  const auto f_deriv = [&]<typename T>(const T * in, T * out) {
    Eigen::Map<const Eigen::Matrix<T, Nx, 1>> m_in(in, nx);
    Eigen::Map<Eigen::Matrix<T, Ny, 1>> m_out(out, ny);

    m_out =
      std::apply(f, utils::tuple_plus(utils::tuple_cast<T>(x), m_in)) - fval.template cast<T>();

    return true;
  };

  ceres::internal::AutoDifferentiate<Result::SizeAtCompileTime,
    ceres::internal::StaticParameterDims<Nx>>(f_deriv, prms, b.size(), b.data(), jac_rows);

  Eigen::Matrix<Scalar, Ny, Nx> jac_ret = jac;

  return std::make_pair(fval, jac_ret);
}

}  // namespace smooth

#endif  // SMOOTH__COMPAT__CERES_HPP_

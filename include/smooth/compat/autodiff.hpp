#ifndef SMOOTH__COMPAT__AUTODIFF_HPP_
#define SMOOTH__COMPAT__AUTODIFF_HPP_

// clang-format off
#include <Eigen/Core>
#include <autodiff/forward/forward.hpp>
#include <autodiff/forward/eigen.hpp>
// clang-format on

#define SMOOTH_DIFF_AUTODIFF

#include "smooth/concepts.hpp"
#include "smooth/internal/utils.hpp"

namespace smooth::diff {

/**
 * @brief Automatic differentiation in tangent space
 *
 * @param f function to differentiate
 * @param X function arguments as tuple x...
 * @return pair( f(x...), dr f_(x...) )
 */
template<typename _F, typename _Wrt>
auto dr_autodiff(_F && f, _Wrt && x)
{
  using Result   = decltype(std::apply(f, x));
  using Scalar   = typename Result::Scalar;
  using AdScalar = autodiff::forward::Dual<Scalar, Scalar>;

  // determine sizes if input and output
  constexpr auto Nx        = utils::tuple_dof<_Wrt>::value;
  auto nx                  = std::apply([](auto &&... args) { return (args.size() + ...); }, x);
  static constexpr auto Ny = Result::SizeAtCompileTime;

  auto val = std::apply(f, x);

  // cast val and x to ad types
  auto x_ad = utils::tuple_cast<AdScalar>(x);
  typename decltype(val.template cast<AdScalar>())::PlainObject val_ad =
    val.template cast<AdScalar>();

  // zero-valued tangent element
  Eigen::Matrix<AdScalar, Nx, 1> a_ad = Eigen::Matrix<AdScalar, Nx, 1>::Zero(nx);

  Eigen::Matrix<Scalar, Ny, Nx> jac = autodiff::forward::jacobian(
    [&f, &val_ad, &x_ad](Eigen::Matrix<AdScalar, Nx, 1> & var) -> Eigen::Matrix<AdScalar, Ny, 1> {
      return std::apply(f, utils::tuple_plus(x_ad, var)) - val_ad;
    },
    wrt(a_ad),
    wrt(a_ad));

  return std::make_pair(val, jac);
}

}  // namespace smooth::diff

#endif  // SMOOTH__COMPAT__AUTODIFF_HPP_

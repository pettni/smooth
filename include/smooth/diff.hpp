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
 * @return pair( f(x...), dr f_(x...) )
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
  Eigen::Index nx = std::apply(
    []<typename... Args>(Args && ... args) { return (args.size() + ...); }, x);
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
 * @brief Available differentiation methods
 *
 * NUMERICAL calculates numerical (forward) derivatives
 *
 * AUTODIFF uses the autodiff (autodiff.github.io) library and requires the
 * autodiff compatability header
 *
 * CERES uses the Ceres (ceres-solver.org) built-in autodiff and requires the
 * Ceres compatability header
 *
 * ANALYTIC is meant for hand-coded derivatives, and requires that
 * the function returns a pair [f(x), dr f_x]
 *
 * DEFAULT selects from a priority list based on availability
 */
enum class Type { NUMERICAL, AUTODIFF, CERES, ANALYTIC, DEFAULT };

/**
 * @brief Differentiation in local tangent space (right derivative)
 *
 * @tparam dm differentiation method to use
 *
 * @param f function to differentiate
 * @param x... function arguments
 * @return pair( f(x...), dr f_(x...) )
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
 * @param x... function arguments
 * @return pair( f(x...), dr f_(x...) )
 */
template<typename _F, typename _Wrt>
auto dr(_F && f, _Wrt && x)
{
  return dr<Type::DEFAULT>(std::forward<_F>(f), std::forward<_Wrt>(x));
}

}  // namespace diff
}  // namespace smooth

#endif  // SMOOTH__DIFF_HPP_

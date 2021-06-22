#ifndef SMOOTH__DIFF_HPP_
#define SMOOTH__DIFF_HPP_

#include <Eigen/Core>
#include <type_traits>

#include "concepts.hpp"

namespace smooth::diff {

namespace detail {

/**
 * @brief Numerical differentiation in tangent space
 *
 * @param f function to differentiate
 * @param wrt... function arguments
 * @return pair( f(wrt...), dr f_(wrt...) )
 */
template<typename _F, typename... _Wrt>
  requires (Manifold<std::decay_t<_Wrt>> &&...)
  && (Manifold<std::invoke_result_t<_F, _Wrt...>>)
auto dr_numerical(_F && f, _Wrt &&... wrt)
{
  using Result = std::invoke_result_t<_F, _Wrt...>;
  using Scalar = typename Result::Scalar;

  // static sizes
  static constexpr Eigen::Index Nx =
    std::min<Eigen::Index>({std::decay_t<_Wrt>::SizeAtCompileTime...}) == -1
      ? -1
      : (std::decay_t<_Wrt>::SizeAtCompileTime + ...);
  static constexpr Eigen::Index Ny = Result::SizeAtCompileTime;

  const Scalar eps = std::sqrt(Eigen::NumTraits<Scalar>::epsilon());

  auto val = f(wrt...);

  // dynamic sizes
  Eigen::Index nx = (wrt.size() + ... + 0);
  Eigen::Index ny = val.size();

  // output variable
  Eigen::Matrix<Scalar, Ny, Nx> jac(ny, nx);

  Eigen::Index index_pos = 0;

  const auto f_iter = [&](auto && w) {
    static constexpr Eigen::Index Nx_j = std::decay_t<decltype(w)>::SizeAtCompileTime;
    const int nx_j                     = w.size();
    for (auto j = 0; j != nx_j; ++j) {
      Scalar eps_j = eps;
      if constexpr (RnLike<std::decay_t<decltype(w)>>) {
        // scale step size if we are in Rn
        eps_j *= abs(w[j]);
        if (eps_j == 0.) { eps_j = eps; }
      }
      w += (eps_j * Eigen::Matrix<Scalar, Nx_j, 1>::Unit(nx_j, j));
      jac.col(index_pos + j) = (f(wrt...) - val) / eps_j;
      w += (-eps_j * Eigen::Matrix<Scalar, Nx_j, 1>::Unit(nx_j, j));
    }
    index_pos += nx_j;
  };

  (f_iter(wrt), ...);

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
 * @param wrt... function arguments
 * @return pair( f(wrt...), dr f_(wrt...) )
 */
template<Type dm, typename _F, typename... _Wrt>
auto dr(_F && f, _Wrt &&... wrt)
{
  if constexpr (dm == Type::NUMERICAL) {
    return detail::dr_numerical(std::forward<_F>(f), std::forward<_Wrt>(wrt)...);
  } else if constexpr (dm == Type::AUTODIFF) {
#ifdef SMOOTH_DIFF_AUTODIFF
    return dr_autodiff(std::forward<_F>(f), std::forward<_Wrt>(wrt)...);
#else
    static_assert(dm != Type::AUTODIFF, "compat/autodiff.hpp header not included");
#endif
  } else if constexpr (dm == Type::CERES) {
#ifdef SMOOTH_DIFF_CERES
    return dr_ceres(std::forward<_F>(f), std::forward<_Wrt>(wrt)...);
#else
    static_assert(dm != Type::CERES, "compat/ceres.hpp header not included");
#endif
  } else if constexpr (dm == Type::ANALYTIC) {
    return f(std::forward<_Wrt>(wrt)...);
  } else if constexpr (dm == Type::DEFAULT) {
#ifdef SMOOTH_DIFF_AUTODIFF
    return dr_autodiff(std::forward<_F>(f), std::forward<_Wrt>(wrt)...);
#elif SMOOTH_DIFF_CERES
    return dr_ceres(std::forward<_F>(f), std::forward<_Wrt>(wrt)...);
#else
    return detail::dr_numerical(std::forward<_F>(f), std::forward<_Wrt>(wrt)...);
#endif
  }
}

/**
 * @brief Differentiation in local tangent space using default method
 *
 * @param f function to differentiate
 * @param wrt... function arguments
 * @return pair( f(wrt...), dr f_(wrt...) )
 */
template<typename _F, typename... _Wrt>
auto dr(_F && f, _Wrt &&... wrt)
{
  return dr<Type::DEFAULT>(std::forward<_F>(f), std::forward<_Wrt>(wrt)...);
}

}  // namespace smooth::diff

#endif  // SMOOTH__DIFF_HPP_

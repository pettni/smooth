#ifndef SMOOTH__DIFF_HPP_
#define SMOOTH__DIFF_HPP_

#include <Eigen/Core>
#include <type_traits>

#include "smooth/common.hpp"

namespace smooth {

/**
 * @brief Numerical differentiation in tangent space
 *
 * @param f function to differentiate
 * @param wrt... function arguments
 * @return pair( f(wrt...), dr f_(wrt...) )
 */
template<typename _F, typename... _Wrt>
auto jacobian(_F && f, _Wrt &&... wrt)
{
  using Result     = std::invoke_result_t<_F, _Wrt...>;
  using Scalar     = typename Result::Scalar;
  const Scalar eps = std::sqrt(Eigen::NumTraits<Scalar>::epsilon());

  // static sizes
  static constexpr int Nx = std::min<int>({lie_info<std::decay_t<_Wrt>>::lie_dof...}) == -1
                            ? -1
                            : (lie_info<std::decay_t<_Wrt>>::lie_dof + ...);
  static constexpr int Ny = lie_info<Result>::lie_dof;

  auto val = f(wrt...);

  // dynamic sizes
  int nx = (lie_info<std::decay_t<_Wrt>>::lie_dof_dynamic(wrt) + ... + 0);
  int ny = lie_info<Result>::lie_dof_dynamic(val);

  // output variable
  Eigen::Matrix<Scalar, Ny, Nx> jac(ny, nx);

  int index_pos = 0;

  const auto f_iter = [&](auto && w) {
    static constexpr int Nx_j = lie_info<std::decay_t<decltype(w)>>::lie_dof;
    const int nx_j            = lie_info<std::decay_t<decltype(w)>>::lie_dof_dynamic(w);
    for (auto j = 0; j != nx_j; ++j) {
      Scalar eps_j = eps;
      if constexpr (EnLike<std::decay_t<decltype(w)>>) {
        // scale step size if we are in Rn
        eps_j *= abs(w[j]);
        if (eps_j == 0.) { eps_j = eps; }

        w[j] += eps_j;
        jac.col(index_pos + j) = (f(wrt...) - val) / eps_j;
        w[j] -= eps_j;
      } else {
        w += (eps_j * Eigen::Matrix<Scalar, Nx_j, 1>::Unit(nx_j, j));
        jac.col(index_pos + j) = (f(wrt...) - val) / eps_j;
        w += (-eps_j * Eigen::Matrix<Scalar, Nx_j, 1>::Unit(nx_j, j));
      }
    }
    index_pos += nx_j;
  };

  (f_iter(wrt), ...);

  return std::make_pair(val, jac);
}

}  // namespace smooth

#endif  // SMOOTH__DIFF_HPP_
